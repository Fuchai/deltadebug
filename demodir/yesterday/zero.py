# first of all, I think we still need a policy network independently from value. Somehow I think this is the best
# We run MCTS until the end
# We then sample the MCTS timesteps to train the neural network (how many times?)
# Then we repeat. (with new MCTS tree? right?)
# We need to design the asynchronous behavior.
from collections import deque
from multiprocessing.pool import ThreadPool, Pool
from mcts import MCTS, TimeStep
from neuralnetwork import NoPolicy, PaperLoss, YesPolicy, SharedPolicy
import random
import torch
import torch.optim
import torch.nn as nn
import os
from pathlib import Path
from os.path import abspath
import datetime
from neuralnetwork import states_to_batch_tensor
import numpy as np
import time
import queue
import threading
import pickle
import math


class AlphaZero:
    """
    Use MCTS to generate g number of games
    Train based on these games with repeatedly sampled time steps
    Refresh the g number of games, repeat
    """

    def __init__(self, model_name, is_cuda=True):
        # NEURAL NETWORK
        self.model_name = model_name
        self.scale = 128

        self.nn = SharedPolicy(self.scale)
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.nn = self.nn.cuda()
        self.loss_fn = PaperLoss()
        self.optim = torch.optim.Adam(self.nn.parameters(), weight_decay=0.01)
        # control how many time steps each loss.backwards() is called for.
        # controls the GPU memory allocation
        self.time_step_sample_size = 1024
        # controls how many boards is fed into a neural network at once
        # controls the GPU utilization.
        self.nn_feeding_batch_size = 512
        # time steps contain up to self.game_size different games.
        self.training_time_steps = []
        self.validation_time_steps = []
        # determines the CPU memory consumption
        game_sacle = 2
        self.training_games_per_refresh = 7 * game_sacle
        self.validation_games_per_refresh = game_sacle
        # controls the variance versus the training speed,
        # higher means lower variance but slower training convergence due to bias
        print("Total threads: ", self.training_games_per_refresh+self.validation_games_per_refresh)

        # variance is too big. The epochs oscillate between modes
        self.replace_ratio_per_refresh = 1 / 10
        self.value_policy_backward_coeff = 1

        self.total_game_refresh = 200
        # if training too much, the model might diverge.
        self.sample_batches_per_epoch = 102400 // self.time_step_sample_size * game_sacle
        self.validation_period = 2000
        self.total_validation_batches = 40
        self.print_period = 10
        self.save_period = 1000
        self.log_file = "log/" + self.model_name + "_" + datetime_filename() + ".txt"
        self.log_file = Path(self.log_file)

        # Pass to MCTS and other methods
        self.max_game_length = 200
        self.peace = 100
        self.simulations_per_play = 200
        # this is a tuned parameter, do not change
        # 4096 memory bound
        self.eval_batch_size = 819200 // self.simulations_per_play
        print("GPU memory batches:", self.eval_batch_size)
        self.debug = True
        self.max_queue_size = self.eval_batch_size * 2

        # if seeds are set incorrectly, the model might diverge
        # self.seed = 123
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

        self.fast = False
        if self.fast:
            self.fast_settings()

        self.starting_epoch = 0
        self.starting_iteration = 0
        if not self.fast:
            self.load_model()

    def fast_settings(self):
        self.max_game_length = 4
        self.simulations_per_play = 10
        self._tsss = self.time_step_sample_size
        self.time_step_sample_size = 4

    def mcts_add_game(self, epoch):
        with torch.no_grad():
            self.nn.eval()
            new_train_time_steps = []
            new_validation_time_steps = []
            nn_thread_edge_queue = queue.Queue(maxsize=self.max_queue_size)
            # def gpu_thread_worker(nn, queue, eval_batch_size, is_cuda):
            gpu_thread = threading.Thread(target=gpu_thread_worker,
                                          args=(self.nn, nn_thread_edge_queue, self.eval_batch_size, self.is_cuda))
            gpu_thread.start()

            # 8 thread MCTS search
            ars = []
            mcts_pool = ThreadPool(processes=self.training_games_per_refresh + self.validation_games_per_refresh)
            # mcts_search_worker(nn_thread_edge_queue,
            #                    self.nn, self.is_cuda,
            #                    self.max_game_length,
            #                    self.peace,
            #                    self.simulations_per_play,
            #                    self.debug, epoch, new_train_time_steps)
            for i in range(self.training_games_per_refresh):
                async_result = mcts_pool.apply_async(mcts_search_worker, args=(nn_thread_edge_queue,
                                                                               self.nn, self.is_cuda,
                                                                               self.max_game_length,
                                                                               self.peace,
                                                                               self.simulations_per_play,
                                                                               self.debug, epoch, new_train_time_steps))
                ars.append(async_result)
            for i in range(self.validation_games_per_refresh):
                async_result = mcts_pool.apply_async(mcts_search_worker, args=(nn_thread_edge_queue,
                                                                               self.nn, self.is_cuda,
                                                                               self.max_game_length,
                                                                               self.peace,
                                                                               self.simulations_per_play,
                                                                               self.debug, epoch,
                                                                               new_validation_time_steps))
                ars.append(async_result)

            mcts_pool.close()
            for ar in ars:
                ar.wait()
            mcts_pool.join()
            print("MCTS pool has joined")

            nn_thread_edge_queue.put(None)
            print("Terminal sentinel is put on queue")
            nn_thread_edge_queue.join()
            if self.debug:
                print("Queue has joined")
            gpu_thread.join()
            if self.debug:
                print("GPU Thread has joined")
            # new_time_steps += mcts.time_steps
            print("Successful generation of", self.validation_games_per_refresh + self.training_games_per_refresh,
                  "games")
            print("Queue empty:", nn_thread_edge_queue.empty())
            # check if any time step do not have children
            new_train_time_steps = [ts for ts in new_train_time_steps if len(ts.children_states) != 0]
            new_validation_time_steps = [ts for ts in new_validation_time_steps if len(ts.children_states) != 0]

            # perform validation and training split
            # all_indices=list(range(len(new_train_time_steps)))
            # random.shuffle(all_indices)
            # total_valid_points=int(len(new_train_time_steps)*self.validation_split)
            # new_valid_indices=all_indices[0:total_valid_points]
            # new_train_indices=all_indices[total_valid_points:]
            # new_valid_points=[new_train_time_steps[i] for i in new_valid_indices]
            # new_train_points=[new_train_time_steps[i] for i in new_train_indices]

            # append training
            self.training_time_steps = self.refresh_helper(new_train_time_steps, self.training_time_steps)
            # old_remove= len(new_train_points) + len(self.training_time_steps) - self.replace_ratio_per_refresh
            # if old_remove<0:
            #     # always remove 10% of the games
            #     # running keep 160 games per sampling population
            #     old_remove= len(self.training_time_steps) // 10
            # old_retain= len(self.training_time_steps) - old_remove
            # self.training_time_steps=random.sample(self.training_time_steps, k=old_retain)
            # self.training_time_steps= self.training_time_steps + new_train_points

            # append validation
            self.validation_time_steps = self.refresh_helper(new_validation_time_steps, self.validation_time_steps)

            if not self.fast:
                self.save_games()

    def refresh_helper(self, new_points, old_points):
        # always remove 10% of the games
        # running keep 160 games per sampling population
        old_remove = int(len(old_points) * self.replace_ratio_per_refresh)
        old_retain_num = len(old_points) - old_remove
        old_points = random.sample(old_points, k=old_retain_num)
        old_points = old_points + new_points
        return old_points

    def save_games(self):
        if len(self.training_time_steps)==0:
            raise ValueError()
        with open("training_timesteps", "wb") as f:
            pickle.dump(self.training_time_steps, f)

        with open("validation_timesteps", "wb") as f:
            pickle.dump(self.validation_time_steps, f)

    def load_games(self):
        try:
            with open("training_timesteps", "rb") as f:
                self.training_time_steps = pickle.load(f)
                print("games loaded")
        except FileNotFoundError:
            print("Training timestep absent from loading")

        try:
            with open("validation_timesteps", "rb") as f:
                self.validation_time_steps = pickle.load(f)
        except FileNotFoundError:
            print("Validation timestep absent from loading")

    def train(self):
        dqlen = 50
        vdq = deque(maxlen=dqlen)
        ptq = deque(maxlen=dqlen)
        pdiffdq = deque(maxlen=dqlen)
        first_run = True
        for epoch in range(self.starting_epoch, self.total_game_refresh):
            if not self.fast:
                self.load_games()
                # if not first_run:
                self.mcts_add_game(epoch)
            else:
                self.load_games()
            for ti in range(self.starting_iteration, self.sample_batches_per_epoch):
                train_vloss, train_ploss, pdiff = self.train_one_round()
                vdq.append(train_vloss)
                ptq.append(train_ploss)
                pdiffdq.append(pdiff)
                if ti % self.print_period == 0:
                    self.log_print(
                        "%14s " % self.model_name +
                        "train epoch %4d, resampling %4d. running value loss: %.5f. running policy loss: %.5f. "
                        "running p diff: %.5f" %
                        (epoch, ti, sum(vdq) / len(vdq), sum(ptq) / len(ptq), sum(pdiffdq) / len(pdiffdq)))
                if ti % self.validation_period == 0:
                    valid_vloss, valid_ploss, valid_pdiff = self.validate()
                    self.log_print(
                        "%14s " % self.model_name +
                        "valid epoch %4d, resampling %4d. validation value loss: %.5f. validation policy loss: %.5f "
                        "validation p diff: %.5f" %
                        (epoch, ti, valid_vloss, valid_ploss, valid_pdiff))
                if ti % self.save_period == 0:
                    self.save_model(epoch, ti)
            self.starting_iteration = 0
            first_run = False

    def run_one_round(self, sampled_tss):
        # compile value tensor
        values = {}

        value_batches = math.ceil(len(sampled_tss) / self.nn_feeding_batch_size)
        for batch_idx in range(value_batches):
            batch_tss = sampled_tss[
                        batch_idx * self.nn_feeding_batch_size: (batch_idx + 1) * self.nn_feeding_batch_size]
            value_inputs = [ts.checker_state for ts in batch_tss]
            value_tensor = states_to_batch_tensor(value_inputs, is_cuda=self.is_cuda)
            _, value_output = self.nn(value_tensor)
            for tsidx, ts in enumerate(batch_tss):
                # ts.v = value_output[tsidx]
                values[ts] = value_output[tsidx]

        # compile policy tensor
        # queue up children_states
        # slice output tensor
        # tss_policy_output = {}

        policy_inputs_queue = []
        dim_ts = []
        logits = {}
        for tsidx, ts in enumerate(sampled_tss):
            logits[ts] = []
            # if ts is az.training_time_steps[3336]:
            #     print("Stop here")
            for child in ts.children_states:
                if len(policy_inputs_queue) != self.nn_feeding_batch_size + 1:
                    # queue up
                    dim_ts.append(ts)
                    policy_inputs_queue.append(child)
                else:
                    ### process
                    self.get_policy_logits(policy_inputs_queue, dim_ts, logits)
                    policy_inputs_queue = []
                    dim_ts = []
                    dim_ts.append(ts)
                    policy_inputs_queue.append(child)
        # remnant in the queue
        if len(policy_inputs_queue) != 0:
            self.get_policy_logits(policy_inputs_queue, dim_ts, logits)

        # policy transpose
        for ts in sampled_tss:
            logits[ts] = torch.cat(logits[ts])
        #     try:
        #         assert(ts.logits.shape[0]==len(ts.children_states))
        #     except AssertionError:
        #         for tsidx, ts in enumerate(az.training_time_steps):
        #             if ts.logits is not None:
        #                 if ts.logits.shape[0]!=len(ts.children_states):
        #                     print(tsidx)
        #                     problem=True
        # if problem:
        #     assert False

        # loss calculation
        vloss, ploss, pdiff = 0, 0, 0
        for ts in sampled_tss:
            # should we reinitialize every time or store them?
            z = torch.Tensor([ts.z])
            pi = torch.Tensor(ts.pi)
            if self.is_cuda:
                z = z.cuda()
                pi = pi.cuda()

            ret = self.loss_fn(values[ts], z, logits[ts], pi)
            vloss += ret[0]
            ploss += ret[1]
            pdiff += ret[2]
        vloss = vloss / len(sampled_tss)
        ploss = ploss / len(sampled_tss)
        pdiff = pdiff / len(sampled_tss)
        return vloss, ploss, pdiff

    def get_policy_logits(self, policy_inputs_queue, dim_ts, logits):
        """

        :param policy_inputs_queue: a list of checker states that need policy logits
        :param dim_ts:
        :return:
        """
        assert (len(policy_inputs_queue) == len(dim_ts))
        policy_tensor = states_to_batch_tensor(policy_inputs_queue, self.is_cuda)
        policy_output, _ = self.nn(policy_tensor)
        for tsidx, ts in enumerate(dim_ts):
            logits[ts].append(policy_output[tsidx, :])

        # policy_tensor = states_to_batch_tensor(policy_inputs_queue, self.is_cuda)
        # policy_output, _ = self.nn(policy_tensor)
        # # slice and append
        # last_ts = None
        # tsbegin = None
        # assert (len(policy_inputs_queue)==len(dim_ts))
        # for tsidx, ts in enumerate(dim_ts):
        #     if ts != last_ts:
        #         if last_ts is not None:
        #             # slice the policy output
        #             # not including tsidx
        #             last_ts.logits.append(policy_output[tsbegin: tsidx, :])
        #             # tss_policy_output[ts].append(policy_output[tsbegin: ts, :, :])
        #         last_ts = ts
        #         tsbegin = tsidx
        # # take care of the last ones
        # # tss_policy_output[ts].append(policy_output[tsbegin: ts, :, :])
        # # including tsidx
        # assert (tsidx + 1 == len(dim_ts))
        # ts.logits.append(policy_output[tsbegin: tsidx + 1, :])

    def train_one_round(self):
        self.nn.train()
        # sample self.batch_size number of time steps, bundle them together
        try:
            sampled_tss = random.sample(self.training_time_steps, k=self.time_step_sample_size)
        except ValueError:
            sampled_tss = self.training_time_steps
        vloss, ploss, pdiff = self.run_one_round(sampled_tss)

        loss = self.value_policy_backward_coeff * vloss + ploss
        loss.backward()
        self.optim.step()
        return vloss.item(), ploss.item(), pdiff

    def validate(self):
        vls = []
        pls = []
        pdiff = []
        for i in range(self.total_validation_batches):
            # if i % self.print_period==0:
            #     print("Validating batch", i)
            vl, pl, pd = self.validate_one_round()
            vls.append(vl)
            pls.append(pl)
            pdiff.append(pd)
        return np.sum(vls) / self.total_validation_batches, \
               np.sum(pls) / self.total_validation_batches, \
               np.sum(pdiff) / self.total_validation_batches

    def validate_one_round(self):
        with torch.no_grad():
            self.nn.eval()
            # sample self.batch_size number of time steps, bundle them together
            try:
                sampled_tss = random.sample(self.validation_time_steps, k=self.time_step_sample_size)
            except ValueError:
                sampled_tss = self.validation_time_steps
            vloss, ploss, pdiff = self.run_one_round(sampled_tss)
        return vloss.item(), ploss.item(), pdiff

    def log_print(self, message):
        string = str(message)
        if self.log_file is not None and self.log_file != False:
            with self.log_file.open("a") as handle:
                handle.write(string + '\n')
        print(string)

    def save_model(self, epoch, iteration):
        if not self.fast:
            epoch = int(epoch)
            task_dir = os.path.dirname(abspath(__file__))
            if not os.path.isdir(Path(task_dir) / "saves"):
                os.mkdir(Path(task_dir) / "saves")

            pickle_file = Path(task_dir).joinpath(
                "saves/" + self.model_name + "_" + str(epoch) + "_" + str(iteration) + ".pkl")
            with pickle_file.open('wb') as fhand:
                torch.save((self.nn.state_dict(), self.optim, epoch, iteration), fhand)

            print("saved model", self.model_name, "at", pickle_file)

    def load_model(self):
        """
        if starting epoch and iteration are zero, it loads the newest model
        :return:
        """
        starting_epoch = self.starting_epoch
        starting_iteration = self.starting_iteration
        task_dir = os.path.dirname(abspath(__file__))
        save_dir = Path(task_dir) / "saves"
        highest_epoch = 0
        highest_iter = 0

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for child in save_dir.iterdir():
            if child.name.split("_")[0] == self.model_name:
                epoch = child.name.split("_")[1]
                iteration = child.name.split("_")[2].split('.')[0]
                iteration = int(iteration)
                epoch = int(epoch)
                # some files are open but not written to yet.
                if child.stat().st_size > 20480:
                    if epoch > highest_epoch or (iteration > highest_iter and epoch == highest_epoch):
                        highest_epoch = epoch
                        highest_iter = iteration

        if highest_epoch == 0 and highest_iter == 0:
            print("nothing to load")
            return

        if starting_epoch == 0 and starting_iteration == 0:
            pickle_file = Path(task_dir).joinpath(
                "saves/" + self.model_name + "_" + str(highest_epoch) + "_" + str(highest_iter) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                computer, optim, epoch, iteration = torch.load(pickle_file,map_location=torch.device('cpu'))
            print('Loaded model at epoch ', highest_epoch, 'iteration', highest_iter)
        else:
            pickle_file = Path(task_dir).joinpath(
                "saves/" + self.model_name + "_" + str(starting_epoch) + "_" + str(starting_iteration) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                computer, optim, epoch, iteration = torch.load(pickle_file,map_location=torch.device('cpu'))
            print('Loaded model at epoch ', starting_epoch, 'iteration', starting_iteration)

        self.nn.load_state_dict(computer)
        self.optim = optim
        self.starting_epoch = highest_epoch
        self.starting_iter = highest_iter


def datetime_filename():
    return datetime.datetime.now().strftime("%m-%d-%H-%M-%S")


def gpu_thread_worker(nn, edge_queue, eval_batch_size, is_cuda):
    while True:
        with torch.no_grad():
            nn.eval()
            edges = []
            last_batch = False
            for i in range(eval_batch_size):
                if edge_queue.empty():
                    break
                try:
                    edge = edge_queue.get_nowait()
                    if edge is None:
                        last_batch = True
                        print("Sentinel received. GPU will process this batch and terminate afterwards")
                    else:
                        edges.append(edge)
                except queue.Empty:
                    pass

            if len(edges) != 0:
                # print("batch size:", len(edges))

                # batch process
                states = [edge.to_node.checker_state for edge in edges]
                input_tensor = states_to_batch_tensor(states, is_cuda)
                # this line is the bottleneck
                if isinstance(nn, YesPolicy) or isinstance(nn, SharedPolicy):
                    value_tensor, logits_tensor = nn(input_tensor)

                else:
                    value_tensor = nn(input_tensor)

                if isinstance(nn, YesPolicy) or isinstance(nn, SharedPolicy):
                    logits_tensor = value_tensor

                for edx, edge in enumerate(edges):
                    edge.value = value_tensor[edx, 0]
                    edge.logit = logits_tensor[edx, 0]
                    edge_queue.task_done()
                    edge.from_node.unassigned -= 1
                    if edge.from_node.unassigned == 0:
                        edge.from_node.lock.release()
            else:
                time.sleep(0.1)

            if last_batch:
                edge_queue.task_done()
                print("Queue task done signal sent. Queue will join. Thread may still be running.")
                return


def mcts_search_worker(nn_thread_edge_queue, nn, is_cuda, max_game_length, peace, simulations_per_play,
                       debug, epoch, new_time_steps):
    mcts = MCTS(nn_thread_edge_queue, nn, is_cuda,
                max_game_length, peace, simulations_per_play,
                debug)
    mcts.play_until_terminal()
    new_time_steps += mcts.time_steps

if __name__ == '__main__':
    az = AlphaZero("highpuctshortgame28", is_cuda=False)
    az.train()

# fork alternate is longer games, value coef = 1
# longer game does not seem to work. All the games now look the same? The loss diverges?

# high puct highpuctshortgame28
# low puct = 1/2
# alternate =4

# first of all, I think we still need a policy network independently from value. Somehow I think this is the best
# We run MCTS until the end
# We then sample the MCTS timesteps to train the neural network (how many times?)
# Then we repeat. (with new MCTS tree? right?)
# We need to design the asynchronous behavior.

from mcts import MCTS, TimeStep
from neuralnetwork import NoPolicy, PaperLoss, YesPolicy
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


class AlphaZero:
    """
    Use MCTS to generate g number of games
    Train based on these games with repeatedly sampled time steps
    Refresh the g number of games, repeat
    """

    def __init__(self, model_name, is_cuda=True):
        # NEURAL NETWORK
        self.model_name = model_name
        self.nn = YesPolicy()
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.nn = self.nn.cuda()
        # TODO
        self.loss_fn = PaperLoss()
        self.optim = torch.optim.Adam(self.nn.parameters(), weight_decay=0.01)
        # control how many time steps each loss.backwards() is called for.
        # controls the GPU memory allocation
        self.training_time_step_batch_size = 2  # 16
        # controls how many boards is fed into a neural network at once
        # controls the speed of gpu computation.
        self.neural_network_batch_size = 128
        # time steps contain up to self.game_size different games.
        self.time_steps = []
        self.game_size = 4
        self.total_game_refresh = 20
        self.reuse_game_interval = 2000
        self.validation_period = 100
        self.print_period = 10
        self.save_period = 1000
        self.log_file = "log/" + self.model_name + "_" + datetime_filename() + ".txt"
        self.log_file = Path(self.log_file)
        self.refresh_period = 10

        self.starting_epoch=0
        self.starting_iteration=0
        self.load_model()

        # Pass to MCTS and other methods
        self.max_game_length = 200
        self.simulations_per_play = 200
        # this is a tuned parameter, do not change
        self.eval_batch_size = 204800 // self.simulations_per_play
        self.debug = True
        self.max_queue_size = self.eval_batch_size*2

        # self.fast_settings()

    def fast_settings(self):
        self.max_game_length=4
        self.simulations_per_play=10
        self.game_size=2

    def mcts_refresh_game(self):
        with torch.no_grad():
            self.nn.eval()
            self.time_steps = []
            for i in range(self.game_size):
                nn_thread_edge_queue = queue.Queue(maxsize=self.max_queue_size)
                # def gpu_thread_worker(nn, queue, eval_batch_size, is_cuda):
                gpu_thread = threading.Thread(target=gpu_thread_worker,
                                              args=(self.nn, nn_thread_edge_queue, self.eval_batch_size, self.is_cuda))
                gpu_thread.start()
                mcts = MCTS(nn_thread_edge_queue, self.nn, self.is_cuda,
                            self.max_game_length, self.simulations_per_play,
                            self.debug)
                mcts.play_until_terminal()
                nn_thread_edge_queue.put(None)
                # print("Terminal sentinel is put on queue")
                nn_thread_edge_queue.join()
                if self.debug:
                    print("Queue has joined")
                gpu_thread.join()
                if self.debug:
                    print("Thread has joined")
                self.time_steps += mcts.time_steps
                print("Successful generation of one game")
                print("Queue empty:", nn_thread_edge_queue.empty())

    def train(self):
        for epoch in range(self.starting_epoch, self.total_game_refresh):
            self.mcts_refresh_game()
            for ti in range(self.starting_iteration, self.reuse_game_interval):
                train_vloss, train_ploss = self.train_one_round()
                if ti % self.print_period == 0:
                    self.log_print(
                        "%14s " % self.model_name +
                        "train epoch %4d, batch %4d. running value loss: %.5f. running policy loss: %.5f" %
                        (epoch, ti, train_vloss, train_ploss))
                if ti % self.validation_period == 0:
                    valid_vloss, valid_ploss = self.validate_one_round()
                    self.log_print(
                        "%14s " % self.model_name +
                        "valid epoch %4d, batch %4d. validation value loss: %.5f. validation policy loss: %.5f" %
                        (epoch, ti, valid_vloss, valid_ploss))
                if ti % self.save_period == 0:
                    self.save_model(epoch, ti)

    def run_one_round(self, sampled_tss):
        # compile value tensor
        value_batches = len(sampled_tss) // self.neural_network_batch_size + 1
        for batch_idx in range(value_batches):
            batch_tss = sampled_tss[
                        batch_idx * self.neural_network_batch_size: (batch_idx + 1) * self.neural_network_batch_size]
            value_inputs = [ts.checker_state for ts in batch_tss]
            value_tensor = states_to_batch_tensor(value_inputs, is_cuda=self.is_cuda)
            _, value_output = self.nn(value_tensor)
            for tsidx, ts in enumerate(batch_tss):
                ts.v = value_output[tsidx]

        # compile policy tensor
        # queue up children_states
        # slice output tensor
        # tss_policy_output = {}
        #
        policy_inputs_queue = []
        dim_ts = []
        for ts in sampled_tss:
            ts.logits = []
            for child in ts.children_states:
                if len(policy_inputs_queue) != self.neural_network_batch_size:
                    # queue up
                    dim_ts.append(ts)
                    policy_inputs_queue.append(child)
                else:
                    ### process
                    self.policy_bonanza(policy_inputs_queue, dim_ts)
                    policy_inputs_queue = []
                    dim_ts = []
        # remnant in the queue
        self.policy_bonanza(policy_inputs_queue, dim_ts)

        # policy transpose
        for ts in sampled_tss:
            ts.logits = torch.cat(ts.logits)

        # loss calculation
        vloss, ploss = 0, 0
        for ts in sampled_tss:
            # should we reinitialize every time or store them?
            z = torch.Tensor([ts.z])
            pi = torch.Tensor(ts.pi)
            if self.is_cuda:
                z = z.cuda()
                pi = pi.cuda()

            ret= self.loss_fn(ts.v, z, ts.logits, pi)
            vloss+=ret[0]
            ploss+=ret[1]
        vloss=vloss/ self.neural_network_batch_size
        ploss=ploss/self.neural_network_batch_size
        return vloss, ploss

    def policy_bonanza(self, policy_inputs_queue, dim_ts):
        policy_tensor = states_to_batch_tensor(policy_inputs_queue, self.is_cuda)
        policy_output, _ = self.nn(policy_tensor)
        # slice and append
        last_ts = None
        tsbegin = None
        for tsidx, ts in enumerate(dim_ts):
            if ts != last_ts:
                if last_ts is not None:
                    # slice the policy output
                    # not including tsidx
                    last_ts.logits.append(policy_output[tsbegin: tsidx, :])
                    # tss_policy_output[ts].append(policy_output[tsbegin: ts, :, :])
                last_ts = ts
                tsbegin = tsidx
        # take care of the last ones
        # tss_policy_output[ts].append(policy_output[tsbegin: ts, :, :])
        # including tsidx
        assert (tsidx + 1 == len(dim_ts))
        ts.logits.append(policy_output[tsbegin: tsidx + 1, :])

    def train_one_round(self):
        self.nn.train()
        # sample self.batch_size number of time steps, bundle them together
        sampled_tss = random.sample(self.time_steps, k=self.training_time_step_batch_size)
        vloss, ploss = self.run_one_round(sampled_tss)
        loss=vloss+ploss
        loss.backward()
        self.optim.step()
        return vloss.item(), ploss.item()

    def validate_one_round(self):
        with torch.no_grad():
            self.nn.eval()
            # sample self.batch_size number of time steps, bundle them together
            sampled_tss = random.sample(self.time_steps, k=self.training_time_step_batch_size)
            vloss, ploss = self.run_one_round(sampled_tss)
            loss = vloss + ploss
        return vloss.item(), ploss.item()

    def time_steps_to_tensor(self, batch_tss):
        # what is the strategy for training time neural network call?
        # I suppose we should do one pass on all values, then we do passes over the probabilities
        # sometimes a batch represent different time steps
        # sometimes a batch represents the same time step but different children of the time step
        # cast and view bonanza.
        # keep track of the loss coefficient.
        # beware not to keep too many boards in the memory at once: when doing probability, do n boards at once

        # compatible with our next design that the probability head is independent from value head, but no other
        # design compatibility

        return None

    def log_print(self, message):
        string = str(message)
        if self.log_file is not None and self.log_file != False:
            with self.log_file.open("a") as handle:
                handle.write(string + '\n')
        print(string)

    def save_model(self, epoch, iteration):

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
        starting_epoch=self.starting_epoch
        starting_iteration=self.starting_iteration
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
                computer, optim, epoch, iteration = torch.load(pickle_file)
            print('Loaded model at epoch ', highest_epoch, 'iteration', highest_iter)
        else:
            pickle_file = Path(task_dir).joinpath(
                "saves/" + self.model_name + "_" + str(starting_epoch) + "_" + str(starting_iteration) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                computer, optim, epoch, iteration = torch.load(pickle_file)
            print('Loaded model at epoch ', starting_epoch, 'iteration', starting_iteration)

        self.nn.load_state_dict(computer)
        self.optim=optim
        self.starting_epoch=highest_epoch
        self.starting_iter=highest_iter


def datetime_filename():
    return datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

# spread more work to main thread
# def gpu_thread_worker(nn, edge_queue, eval_batch_size, is_cuda):
#     while True:
#         with torch.no_grad():
#             nn.eval()
#             edges = []
#             last_batch = False
#             for i in range(eval_batch_size):
#                 if edge_queue.empty():
#                     break
#                 try:
#                     edge = edge_queue.get_nowait()
#                     if edge is None:
#                         last_batch = True
#                         # print("Sentinel received. GPU will process this batch and terminate afterwards")
#                     else:
#                         edges.append(edge)
#                 except queue.Empty:
#                     pass
#
#             if len(edges) != 0:
#                 # batch process
#                 states = [edge.to_node.checker_state for edge in edges]
#                 input_tensor = states_to_batch_tensor(states, is_cuda)
#                 # this line is the bottleneck
#                 if isinstance(nn, YesPolicy):
#                     value_tensor, logits_tensor = nn(input_tensor)
#
#                 else:
#                     value_tensor = nn(input_tensor)
#
#                 # # value
#                 # value_array = value_tensor.cpu().numpy()
#                 # value_array = np.squeeze(value_array, axis=1)
#                 # value_list = value_array.tolist()
#                 # if isinstance(nn, YesPolicy):
#                 #     # logits
#                 #     logit_array = logits_tensor.cpu().numpy()
#                 #     logit_array = np.squeeze(logit_array, axis=1)
#                 #     logit_list = logit_array.tolist()
#                 # else:
#                 #     logit_list = value_list
#
#                 for edx, edge in enumerate(edges):
#                     edge.value = value_tensor[edx, 0].item()
#                     if isinstance(nn, YesPolicy):
#                         edge.logit = logits_tensor[edx,0].item()
#                     else:
#                         edge.logit = value_tensor[edx,0].item()
#                     edge_queue.task_done()
#                     edge.from_node.unassigned -= 1
#                     if edge.from_node.unassigned == 0:
#                         edge.from_node.lock.release()
#             else:
#                 time.sleep(0.1)
#
#             if last_batch:
#                 edge_queue.task_done()
#                 # print("Queue task done signal sent. Queue will join. Thread may still be running.")
#                 return



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
                        # print("Sentinel received. GPU will process this batch and terminate afterwards")
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
                if isinstance(nn, YesPolicy):
                    value_tensor, logits_tensor = nn(input_tensor)

                else:
                    value_tensor = nn(input_tensor)

                if isinstance(nn, YesPolicy):
                    logits_tensor=value_tensor

                # value
                # value_array = value_tensor.cpu().numpy()
                # value_array = np.squeeze(value_array, axis=1)
                # value_list = value_array.tolist()
                # if isinstance(nn, YesPolicy):
                #     # logits
                #     logit_array = logits_tensor.cpu().numpy()
                #     logit_array = np.squeeze(logit_array, axis=1)
                #     logit_list = logit_array.tolist()
                # else:
                #     logit_list = value_list

                for edx, edge in enumerate(edges):
                    edge.value = value_tensor[edx,0]
                    edge.logit = logits_tensor[edx,0]
                    edge_queue.task_done()
                    edge.from_node.unassigned -= 1
                    if edge.from_node.unassigned == 0:
                        edge.from_node.lock.release()
            else:
                time.sleep(0.1)

            if last_batch:
                edge_queue.task_done()
                # print("Queue task done signal sent. Queue will join. Thread may still be running.")
                return

if __name__ == '__main__':
    az = AlphaZero("alpha", is_cuda=True)
    az.train()

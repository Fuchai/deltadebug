import time
from checker import Checker
import numpy as np
import math
from permatree import PermaTree
import random
from queue import Empty
from neuralnetwork import states_to_batch_tensor
import torch

# tree search does not check same node
# algorithm
# Input parameters: node (; node of a tree), tree structure, neural net
class MCTS:
    """
    Selects from root until a leaf node. A leaf node denotes a leaf node in the PermaTree, not the end game (ref Zero
    page 2 right column paragraph 1).
    """

    def __init__(self, nn_execution_queue, nn, is_cuda, max_game_length, simulations_per_play, debug):
        self.checker = Checker()
        self.permaTree = PermaTree(self.checker, is_cuda)
        self.nn_queue = nn_execution_queue
        self.nn=nn
        # changes to False after the first 30 moves
        self.temperature = True
        self.temperature_change_at = 30
        self.puct = 10000
        self.max_game_length = max_game_length
        self.time_steps = []
        self.is_cuda = is_cuda
        self.simulations_per_play = simulations_per_play
        self.debug=debug

    def play_until_terminal(self):
        """
        plays the actual game.
        play until terminal (including resignation)
        when terminal, the winner is calculated, and all data points get an extra value
        :return:
        """
        for step in range(self.max_game_length):
            # if step % 10 == 0:
            print("Game step " + str(step) + " /" + str(self.max_game_length))
            if step == self.temperature_change_at:
                self.temperature = False
            if self.debug:
                t0 = time.time()
            for simulation in range(self.simulations_per_play):
                self.simulation()
                # if simulation % 40 == 0 and self.debug:
                    # print("Simulation " + str(simulation) + " /" + str(self.simulations_per_play))
            if self.debug:
                t1 = time.time()
                print("Time per play: " + str(t1 - t0))
                self.permaTree.root.checker_state.print_board()
            terminal = self.play()
            if terminal:
                print("Terminated at step", step)
                break

        # there will be an outcome whether the game reaches terminal or not
        final_state = self.permaTree.root.checker_state
        outcome = final_state.evaluate()

        # TODO not continuous outcome currently
        if outcome == 0:
            z = 0
        else:
            # truth table:
            #      flipped  not flipped
            # o>0    -1          1
            # o<0     1         -1

            a = final_state.flipped
            b = outcome > 0
            z = a ^ b
            z = z * 2 - 1

        # assign z
        for ts in self.time_steps:
            if not ts.checker_state.flipped:
                ts.z = z
            else:
                ts.z = -z

    def play(self):
        """
        moves the root of the tree
        add a data point without the final outcome z
        final outcome will be assigned at terminal
        :return:
        0: normal, game continues
        1: game terminates with no moves from root
        2: draw due to excessive non-capturing
        """
        root = self.permaTree.root
        if len(root.edges) == 0:
            return 1
        scores = []
        for level_one_edge in root.edges:
            vc = level_one_edge.visit_count
            scores.append(vc)
        if self.temperature:
            sum_scores = sum(scores)
            pi = [score / sum_scores for score in scores]
            # samples instead

            sampled_action = random.choices(range(len(root.edges)), weights=pi, k=1)
            sampled_action = root.edges[sampled_action[0]]
            self.permaTree.move_root(sampled_action.to_node)
        else:
            maxscore = - float("inf")
            maxedx = None
            for edx, score in enumerate(scores):
                if score > maxscore:
                    maxedx = edx
                    maxscore = score
            pi = [0] * len(scores)
            pi[maxedx] = 1
            max_action = root.edges[maxedx]
            self.permaTree.move_root(max_action.to_node)
        if self.permaTree.last_capture == 40:
            print("Terminated due to peaceful activity")
            return 2

        self.time_steps.append(TimeStep(root.checker_state, root.get_children_checker_states(), pi))

    def simulation(self):
        """
        run select, expand, backup L-times on a root node
        :return: L: the length of simulation from root to leaf node
        """
        current_node = self.permaTree.root
        l = 1
        while not current_node.is_leaf():
            selected_edge = self.select(current_node)
            current_node = selected_edge.to_node
            l += 1
        self.expand(current_node)
        if current_node.is_root():
            v = 0
        else:
            v = current_node.from_edge.value
        self.backup(current_node, v)
        return l

    # def temperature_adjust(self, count):
    #     return count ** (1 / self.temperature)

    def select(self, node):
        """
        select function is called from the root to the leaf node in perma_tree for each simulation of L simulations
        :param node:
        :return:
        """
        # if the node is being expanded, then all of its children do not have value and probability, thus
        # select() must wait
        # block if the first acquire does not open
        # unblock if no lock on semaphore, where the values of the node has been assigned

        # this acquire blocks for quite a while to wait for the gpu thread to return answer,
        # we should share some of the tasks
        # let's release the lock when logits are ready, but probability might not be computed
        node.lock.acquire()
        if not node.probability_ready:
            # COMMENT THIS OUT TO USE OLD FUNCTION
            for edge in node.edges:
                edge.value=edge.value.item()
                edge.logit=edge.logit.item()
            # COMMENT THIS OUT TO USE OLD FUNCTION

            # probability
            prob_tensor = self.nn.logits_to_probability(torch.Tensor([edge.logit for edge in node.edges]))
            prob_array = prob_tensor.numpy()
            prob_list = prob_array.tolist()
            for ie, edge in enumerate(node.edges):
                edge.prior_probability = prob_list[ie]
            node.probability_ready=True
        node.lock.release()

        # argmax_input_list = contains a list of [Q(s,a)+U(s,a)] vals of all outbound edges of a node
        QU = []
        # every node/node should have a list of next possible actions
        # loop through all the child edges of a node
        if node.is_root():
            sumnsb = 1
            for edge in node.edges:
                sumnsb += edge.visit_count
        else:
            sumnsb = node.from_edge.visit_count

        for edge in node.edges:
            Nsa = edge.visit_count
            # Nsb = sum(edge.sibling_visit_count_list)  # does Nsb mean edge sibling visit count ? <verify>
            # u(s,a) = controls exploration
            if edge.prior_probability is None:
                print("RACE CONDITION")
                print(node.is_root())
            Usa = (self.puct * edge.prior_probability * math.sqrt(sumnsb)) / (1 + Nsa)
            QU.append(edge.mean_action_value + Usa)

        # pick the edge that is returned by the argmax and return it
        # make it node
        maxqu = -float('inf')
        maxedx = None
        for edx, qu in enumerate(QU):
            if qu > maxqu:
                maxedx = edx
                maxqu = qu
        selected_edge = node.edges[maxedx]
        return selected_edge

    def expand(self, leaf_node):
        """
        expand and evaluate the leaf node sl of the selected edge
        value and prior probability are assigned
        :param leaf_node:
        :return:
        """

        # will call assign value for each child
        terminal = leaf_node.construct_edges()
        if terminal:
            return

        # # initialize basic statistics
        # for idx, edge in enumerate(leaf_node.edges):
        #     edge.visit_count = 0
        #     edge.action_value = 0
        #     edge.mean_action_value = 0
        #     # update perma tree with current edge
        #     # self.permaTree.update(edge)

        # initialize value and probability of all children
        leaf_node.put_children_on_nn_queue(self.nn_queue)

    def backup(self, leaf_node, v):
        """
        trace back the whole path from given node till root node while updating edges on the path

        :param leaf_node:
        :return:
        """
        # parent of root node is null
        current_node = leaf_node
        while current_node.parent is not None:
            edge = current_node.from_edge
            edge.visit_count = edge.visit_count + 1
            edge.total_action_value = edge.total_action_value + v
            edge.mean_action_value = edge.total_action_value / edge.visit_count
            current_node = edge.from_node

    def print_root(self):
        self.permaTree.root.checker_state.print_board()

# interfaces with the Zero
class TimeStep:
    def __init__(self, checker_state, children_states, mcts_pi):
        self.checker_state = checker_state
        self.children_states = children_states
        self.pi = mcts_pi
        self.z=None

        # neural network output ref holder
        self.v=None
        self.logits=None

# load the model, make decision
from alphabeta import AlphaBeta
from mcts import MCTS
from alphabeta import *
from checker import *
from zero import *


class Agent:
    def respond(self, opponent_state):
        """
        Override this method

        :param opponent_state: the opponent resulting state
        :return: agent action
        """
        action=None
        return action

class MinimaxAgent(Agent, AlphaBeta):
    def __init__(self, first, *args, **kwargs):
        super(MinimaxAgent, self).__init__(human_first= not first, human_second=first, *args, **kwargs)
        current_round=0

    def respond(self, opponent_state):
        """
        takes a unflipped opponent state
        responds, do not flip back
        :param opponent_state:
        :return:
        """
        self.human_play(opponent_state)
        ret = self.auto_play(verbose=False)
        # self.state.print_board()
        return self.state

    def versus(self):

        for i in range(self.max_rounds):
            ret=True
            if i%2==0:
                if self.human_first:
                    self.verbose_human_play()
                else:
                    ret=self.auto_play()
                    print("Alpha played")
                    self.state.get_flipped_state().print_board(player_view=False)
            else:
                if self.human_second:
                    self.verbose_human_play()
                else:
                    ret=self.auto_play()
                    print("Beta played")
                    self.state.get_flipped_state().print_board(player_view=False)
            if not ret:
                print("Game finished")
                break
        if ret:
            print("Game reached max rounds")
        self.print_winner()

class NeuralAgent(Agent):
    def __init__(self,mcts):
        self.mcts=mcts
        self.step=0

    def respond(self, opponent_state):
        """
        similar to play_until_terminal except this only play once
        :param opponent_state: opponent's resulting state
        :return: agent's resulting state, terminal
        """
        self.receive(opponent_state)
        return self.play()

    def receive(self, opponent_state):
        self.run_simulation()
        found_child = self.mcts.permaTree.root.find_child(opponent_state)
        self.mcts.permaTree.move_root(found_child)

    def play(self):
        self.run_simulation()
        terminal = self.mcts.play()
        if terminal:
            print("Terminated at step", self.step)
        # return root aka board state after agent's action
        self.step += 2
        return self.mcts.permaTree.root.checker_state, terminal

    def run_simulation(self):
        if self.step == self.mcts.temperature_change_at:
            self.mcts.temperature = False
        for idx in range(self.mcts.simulations_per_play):
            self.mcts.simulation()


def alphazero_versus_alphazero(m1="alternate", m2="alternate", e1=28, e2=42):
    """

    :param m1:
    :param m2:
    :param e1:
    :param e2:
    :return: first player won
    """
    # first player is AlphaZero
    # second player is AlphaBeta

    # load in neural network

    print("AlphaChecker Zero epoch", e1, "is first player")
    print("AlphaChecker Zero epoch", e2, "is second player")

    az1, gq1, gpu1=alphazero_factory(m1, e1)
    az2, gq2, gpu2=alphazero_factory(m2,e2)

    for i in range(az1.mcts.max_game_length):
        print("Round", i)
        if i==0:
            action, terminal=az1.play()
            print("First player epoch", e1, "played:")
            az1.mcts.print_root()
        else:
            if i % 2 == 0:
                action, terminal = az1.respond(action)
                print("First player epoch", e1, "played:")
                az1.mcts.print_root()
            else:
                action, terminal = az2.respond(action)
                print("Second player epoch", e2, "played:")
                az2.mcts.print_root()
        if terminal:
            print("Game finished")
            break
    if not terminal:
        print("Game reached max rounds")
    gq1.put(None)
    gq2.put(None)
    print("Terminal sentinel is put on queue")
    gq1.join()
    gq2.join()
    gpu1.join()
    gpu2.join()
    return az1.mcts.find_winner(az1.mcts.permaTree.root), az1.mcts.permaTree.root.checker_state

def mass_play(az, ret_actions, ret_terminals, idx):
    ret_actions[idx], ret_terminals[idx]=az.play()

def mass_respond(az, actions, terminals, ret_actions, ret_terminals, idx):
    if not terminals[idx]:
        ret_actions[idx], ret_terminals[idx]=az.respond(actions[idx])


def alphazero_factory(model_name="alternate", epoch=0, is_cuda=True):
    # load in neural network
    alphazero = AlphaZero(model_name, is_cuda=is_cuda)
    alphazero.starting_epoch = epoch
    alphazero.load_model()
    nn_thread_edge_queue = queue.Queue(maxsize=alphazero.max_queue_size)
    gpu_thread = threading.Thread(target=gpu_thread_worker,
                                  args=(alphazero.nn, nn_thread_edge_queue,
                                        alphazero.eval_batch_size, alphazero.is_cuda))
    gpu_thread.start()
    mcts = MCTS(nn_thread_edge_queue, nn=alphazero.nn, is_cuda=alphazero.is_cuda,
                max_game_length=alphazero.max_game_length, peace=alphazero.peace,
                simulations_per_play=alphazero.simulations_per_play, debug=False)

    # construct agents
    agz = NeuralAgent(mcts)
    return agz, nn_thread_edge_queue, gpu_thread

def alphazero_versus_alphabeta():
    agz, queue, thread=alphazero_factory("alternate", is_cuda=True)
    print("AlphaChecker Zero is first player")
    ab = MinimaxAgent(first=False, alpha_depth=2, beta_depth=2)
    print("AlphaBeta Pruning is second player")

    # play game
    print("Initial board")
    agz.mcts.print_root()
    action = agz.mcts.permaTree.root
    for i in range(agz.mcts.max_game_length):
        ret = True
        if i==0:
            action, end = agz.play()
            print("Zero played:")
            agz.mcts.print_root()
        else:
            if i % 2 == 0:
                action, end = agz.respond(action)
                ret = not end
                print("Zero played:")
                agz.mcts.print_root()
            else:
                action = ab.respond(action.get_flipped_state())
                print("Beta played:")
                # flip it for alphago because it does not know how to flip it
                action.print_board(player_view=False)
        if not ret:
            print("Game finished")
            break
    if ret:
        print("Game reached max rounds")
    ab.print_winner()


def repeat_experiments(e1, e2, total):
    m1 = "alternate"
    m2 = "alternate"
    first_player_winning_count=0
    draw=0
    lose=0
    for i in range(total):
        winner, final_state=alphazero_versus_alphazero(m1, m2, e1, e2)
        if winner==1:
            first_player_winning_count+=1
        elif winner == 0:
            draw+=1
        else:
            lose+=1
        final_state.print_board()
        print("first player epoch", e1,  "winning:", first_player_winning_count)
        print("draw:" , draw)
        print("second player epoch", e2, "winning:", lose)
    return first_player_winning_count, draw, lose


if __name__ == '__main__':
    # res={}
    # e1, e2, total=28, 19, 8
    # win, draw, lose=repeat_experiments(e1, e2, total)
    # res[(e1, e2, total)]=(win,draw,lose)
    #
    # e1, e2, total=9, 19, 20
    # win, draw, lose=repeat_experiments(e1, e2, total)
    # res[(e1, e2, total)]=(win,draw,lose)
    #
    # e1, e2, total=42, 9, 20
    # win, draw, lose=repeat_experiments(e1, e2, total)
    # res[(e1, e2, total)]=(win,draw,lose)
    #
    # print(res)
    #
    # res={}
    # e1, e2, total=19, 9, 5
    # win, draw, lose=repeat_experiments(e1, e2, total)
    # res[(e1, e2, total)]=(win,draw,lose)
    #
    # e1, e2, total=28, 9, 20
    # win, draw, lose=repeat_experiments(e1, e2, total)
    # res[(e1, e2, total)]=(win,draw,lose)
    #
    # e1, e2, total=42, 19, 20
    # win, draw, lose=repeat_experiments(e1, e2, total)
    # res[(e1, e2, total)]=(win,draw,lose)
    #
    # print(res)

    res={}
    # e1, e2, total=42, 28, 20
    # win, draw, lose=repeat_experiments(e1, e2, total)
    # res[(e1, e2, total)]=(win,draw,lose)

    e1, e2, total=19, 27, 20
    win, draw, lose=repeat_experiments(e1, e2, total)
    res[(e1, e2, total)]=(win,draw,lose)
    #
    # e1, e2, total=9, 42, 20
    # win, draw, lose=repeat_experiments(e1, e2, total)
    # res[(e1, e2, total)]=(win,draw,lose)

    print(res)
import math
from checker import *

debug=True
class StateContainer:
    # hashed container for State lookup
    def __init__(self, state_list=None):
        if state_list is None:
            self.container = {}
        else:
            for state in state_list:
                if not self.has(state):
                    self.add(state)
                else:
                    print("state is already in container")

    def has(self, state):
        if state.board_hash() in self.container:
            return self.container[state.board_hash()]
        else:
            return False

    def __iter__(self):
        return iter(self.container.values())

    def __len__(self):
        return len(self.container)

    def __contains__(self, state):
        return self.has(state)

    def add(self, state):
        if not self.has(state):
            self.container[state.board_hash()] = state

    def remove(self, state):
        self.container.pop(state.board_hash())


class AlphaBeta(Checker):
    """
     convention: alpha is not flipped. beta is always flipped. first player is always alpha
    """

    # interacts with a checker game and finds the best move for the AI player
    # every board passed to AlphaBeta assumes optimization for positive player
    # you have to flip the board, basically.
    def __init__(self, alpha_depth=5, beta_depth=5, human_first=False, human_second=False, max_rounds=300):
        super(AlphaBeta, self).__init__()
        # currently no visited states check is implemented
        # the reason is because I do not know how to keep an even depth if multiple states wound up together
        # # the same position is drastically different given who plays first
        # # a king in check runs if the king plays, and mates if the checker plays
        # self.max_nodes = StateContainer()
        # self.min_nodes = StateContainer()

        # be careful that self.state and self.root_node.state should always be the same object
        if debug:
            assert not self.state.flipped
        self.root_node = AlphaNode(self.state, None)
        self.alpha_depth = alpha_depth
        self.beta_depth=beta_depth
        self.last_state = None
        self.human_first=human_first
        self.human_second=human_second
        self.max_rounds=max_rounds

    def start_game(self):
        print("Current board")
        self.print_board(player_view=False)
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

    def print_winner(self):
        pieces=self.root_node.evaluate()
        if pieces>0:
            print("First player wins")
        elif pieces<0:
            print("Second player wins")
        else:
            print("Draw")

    def auto_play(self, verbose=False):
        if debug:
            assert self.root_node.state is self.state
        if verbose:
            self.state.print_board()
        if self.root_node.is_alpha:
            self.root_node.alpha_prune(math.inf, self.alpha_depth)
        else:
            self.root_node.beta_prune(-math.inf, self.beta_depth)

        if self.root_node.choice is not None:
            # will be None if self.root_node is terminal and has no choice left
            self.last_state=self.root_node.state
            self.root_node = self.root_node.choice
            self.state = self.root_node.state
            if verbose:
                print("becomes")
                self.state.print_board()
            return True
        else:
            return False

    def verbose_human_play(self):
        state=self.ask_human_for_action(self.state)
        self.human_play(state)

    def human_play(self, human_action):
        if debug:
            assert self.root_node.state is self.state
        if len(self.root_node.children)==0:
            self.root_node.make_children()
        child = self.root_node.find_child(human_action.get_flipped_state())
        self.root_node = child
        self.state = self.root_node.state

    def print_board(self,player_view=False):
        self.root_node.state.print_board(player_view=player_view)


class AlphaBetaNode:
    """
    Modified significantly from wikipedia alpha beta pruning algorithm.
    The tree is permanent
    Alpha prune and beta prune is separate, which eliminates unused alpha param for alpha nodes, vice versa.
    Wikipedia has two max functions for alpha nodes, which is redundant.
    Every prune step holds a reference to the choice
    The logic from my pruning function is very visible.
    """

    def __init__(self, state, from_node):
        # self.is_alpha = is_alpha
        # # alpha value is the lower bound of the minimax value of this alpha node
        # if is_alpha:
        #     self.alpha = None
        # else:
        #     self.beta = None
        self.state = state
        self.parent = None
        self.children = []
        self.choice = None
        self.from_node=from_node

    def find_child(self, state):
        for child in self.children:
            if child.state.board_hash() == state.board_hash():
                return child

    def is_terminal(self):
        return (self.state.board >= 0).all() or (self.state.board <= 0).all()


class AlphaNode(AlphaBetaNode):
    def __init__(self, state, from_node):
        super(AlphaNode, self).__init__(state,from_node)
        self.is_alpha = True
        self._alpha = None

    def make_children(self):
        if debug:
            assert len(self.children) == 0
        states = self.state.get_legal_actions()[0]
        for state in states:
            opponent_state = state.get_flipped_state()
            child_node = BetaNode(opponent_state,self)
            self.children.append(child_node)

    def alpha_prune(self, parent_beta, depth):
        if debug:
            assert self.is_alpha
        if depth == 0 or self.is_terminal():
            return self.evaluate()

        if len(self.children) == 0:
            self.make_children()

        # the lower bound of what this max node can achieve
        lower_bound_alpha = -math.inf
        for child in self.children:
            # if minchild does not explore all grandchildren,
            # then the minchild has discovered an option with value lower than lower_bound_alpha
            # which will be returned, and it will not update lower_bound_alpha here
            child_value = child.beta_prune(lower_bound_alpha, depth - 1)
            if child_value > lower_bound_alpha:
                lower_bound_alpha = child_value
                self.choice = child
            if lower_bound_alpha >= parent_beta:
                # parent wants to minimize
                # parent has an option with less payoff
                # parent will not choose this alpha node
                # this alpha node will not need to explore its children
                break
        self._alpha = lower_bound_alpha
        return lower_bound_alpha

    def evaluate(self):
        # should not be called unless the depth is reached or the game ends
        # zero sum compliant
        assert not self.state.flipped
        return self.state.board.sum()


class BetaNode(AlphaBetaNode):
    def __init__(self, state, from_node):
        super(BetaNode, self).__init__(state, from_node)
        self.is_alpha = False
        self._beta = None

    def make_children(self):
        if debug:
            assert len(self.children) == 0
        states = self.state.get_legal_actions()[0]
        for state in states:
            if debug:
                assert ((self.state.board>0).sum()==(state.board>0).sum())
                assert ((self.state.board<0).sum()>=(state.board<0).sum())
            opponent_state = state.get_flipped_state()
            child_node = AlphaNode(opponent_state, self)
            self.children.append(child_node)

    def beta_prune(self, parent_alpha, depth):
        if debug:
            assert not self.is_alpha
        if depth == 0 or self.is_terminal():
            return self.evaluate()

        if len(self.children) == 0:
            self.make_children()

        upper_bound_beta = math.inf
        for child in self.children:
            child_value = child.alpha_prune(upper_bound_beta, depth - 1)
            if child_value < upper_bound_beta:
                upper_bound_beta = child_value
                self.choice = child
            if upper_bound_beta <= parent_alpha:
                break
        self._beta = upper_bound_beta
        return upper_bound_beta

    def evaluate(self):
        if debug:
            assert self.state.flipped
        return -self.state.board.sum()

    def evaluate_defensive(self):
        if debug:
            assert self.state.flipped
        board= (-self.state.board)
        board=board[board>0]
        return board.sum()

    def evaluate_offensive(self):
        if debug:
            assert self.state.flipped
        board= (-self.state.board)
        board=board[board<0]
        return board.sum()

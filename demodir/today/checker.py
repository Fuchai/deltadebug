import numpy as np
import random

debug = True


class Checker:
    def __init__(self):
        # just a bunch of interfaces added to the State object.
        # better not to inherit otherwise it's confusing.
        self.state = CheckerState(self.init_new_board(), flipped=False)

    @staticmethod
    def init_new_board():
        board = np.zeros((8, 8), dtype=np.int)
        for row in range(0, 3):
            for col in range(0, 8):
                if (row + col) % 2 == 0:
                    board[row, col] = 1
        for row in range(5, 8):
            for col in range(0, 8):
                if (row + col) % 2 == 0:
                    board[row, col] = -1
        return board

    def flip_board(self):
        self.state = self.state.get_flipped_state()

    def which_player(self):
        return "black to move" if self.state.flipped else "white to move"

    def print_board(self, player_view=False):
        print(self.which_player())
        self.state.print_board(player_view)

    @staticmethod
    def ask_human_for_action(from_state):
        """
        Incredible solution. only 40 lines. complete code reuse
        :param from_state:
        :return:
        """
        move_tree = from_state.get_legal_actions()[1]
        current_tree = move_tree
        print("Please enter your move, current board:")
        from_state.print_board(player_view=True)

        done = False
        while not done:
            try:
                from_row = int(input("What's the row number of the piece you want to move/jump? [0,8)"))
                from_col = int(input("What's the col number of the piece you want to move/jump? [0,8)"))
                to_row = int(input("What's the row number of the piece you want to move/jump to? [0,8)"))
                to_col = int(input("What's the col number of the piece you want to move/jump to? [0,8)"))

                if (from_row, from_col, to_row, to_col) in current_tree:
                    from_state, current_tree = current_tree[from_row, from_col, to_row, to_col]
                    if current_tree is None:
                        done = True
                    else:
                        # multi-jump
                        from_state.print_board(player_view=True)
                        # ask if you want another round
                        ask = True
                        while ask:
                            user_wants_jump = input("Do you want to jump another time? [y/n] Your piece is now at ("
                                                    + str(to_row) + "," + str(to_col) + ")")
                            if user_wants_jump.lower() == "n":
                                done = True
                                ask = False
                            elif user_wants_jump.lower() == "y":
                                done = False
                                ask = False
                            else:
                                ask = True
                else:
                    print("Invalid input")
            except ValueError:
                print("Invalid input")

        print("Your turn finished. Board now:")
        print(from_state.print_board(player_view=True))
        return from_state

    @staticmethod
    def player_play_fast_enter(state, path):
        """[(7, 7, 5, 5), (5, 5, 3, 3), (3, 3, 5, 1)]"""
        print(state)
        state, move_tree = state.get_legal_actions()
        current = []
        for action_tuple in path:
            if action_tuple in move_tree:
                current += [action_tuple]
                state, move_tree = move_tree[action_tuple]
                print("==========")
                print(current)
                print(state)
            else:
                raise ValueError("invalid path")
        return state

    @staticmethod
    def print_moves(move_tree, path=None):
        if path is None:
            path = []
        for move, value in move_tree.items():
            state, tree = value
            print("========")
            # print("-->".join([str(i) for i in path+[move]]))
            # better print this way becasue I can just copy and enter as code
            print(path + [move])
            print(state)
            if tree is not None:
                Checker.print_moves(tree, path + [move])


class CheckerState(dict):
    # Markov state. No path remembered. Checker is a Markov game.
    def __init__(self, board, flipped):
        super(CheckerState, self).__init__()
        # 1 for white, -1 for black. White moves first. I'm white.
        # 2 for crowned white, -2 for crowned black.
        # this is the cartesian notation, not the matrix notation.
        # row, column. 0,0 is the bottom left. row +1 is forward. column +1 is right.
        # np array
        self.board = board
        self.flipped = flipped
        self.is_action = False

    def get_legal_actions(self):
        """
        Your main function to get legal actions. It returns every legal actions together with a
        dict-of-dict representation of action tree, where the key is action and
        the value is (state, dict). The tree is
        only for the current player, not the complete game.

        It's critical to get legal jumps earlier than legal moves. This enhances the alpha-beta pruning procedure.
        :return:
        """
        if debug:
            self.assert_consistency()
        # this is not the recursive call. One call every player round of the game.
        act2, mt2 = self.get_legal_jumps()
        act1, mt1 = self.get_legal_moves()
        random.shuffle(act2)
        random.shuffle(act1)
        actions = act2 + act1
        mt2.update(mt1)

        if debug:
            for action in actions:
                # you are not allowed to flip the board when you search for the actions originating from the current node.
                assert (action.flipped == self.flipped)

        # move tree should be discarded as much as possible
        # the game itself is path independent
        # freeing up the pointers will save a lot of memory
        return actions, mt2

    def get_legal_moves(self):
        # literal moves, not jumps
        legal_moves = []
        immediate_children = {}
        # for each piece of the player, try move in both directions
        for row in range(0, 8):
            for column in range(0, 8):
                my_piece = self.board[row, column]
                if my_piece == 1:
                    # man moves
                    # move left forward
                    for c_delta in (-1, 1):
                        proposed_new_row = row + 1
                        proposed_new_column = column + c_delta
                        action = self.construct_action_after_move_if_legal(row, column, proposed_new_row,
                                                                           proposed_new_column)
                        if action is not None:
                            legal_moves.append(action)
                            immediate_children[(row, column, proposed_new_row, proposed_new_column)] = (action, None)
                if my_piece == 2:
                    # king moves
                    # move left forward
                    for r_delta in (-1, 1):
                        for c_delta in (-1, 1):
                            proposed_new_row = row + r_delta
                            proposed_new_column = column - c_delta
                            action = self.construct_action_after_move_if_legal(row, column, proposed_new_row,
                                                                               proposed_new_column)
                            if action is not None:
                                legal_moves.append(action)
                                immediate_children[(row, column, proposed_new_row, proposed_new_column)] = \
                                    (action, None)

        return legal_moves, immediate_children

    def construct_action_after_move_if_legal(self, from_row, from_col, proposed_new_row, proposed_new_column):
        piece = self.board[from_row, from_col]
        if piece == 1:
            r_delta = (1,)
            c_delta = (1, -1)
        else:
            r_delta = (1, -1)
            c_delta = (1, -1)
        if proposed_new_row - from_row not in r_delta:
            print("Invalid move")
            return None
        if proposed_new_column - from_col not in c_delta:
            print("Invalid move")
            return None

        # if the proposed location is on the board and is empty
        if self.is_on_board(proposed_new_row, proposed_new_column) and \
                self[proposed_new_row, proposed_new_column] == 0:
            new_board = np.copy(self.board)
            moved_piece = new_board[from_row, from_col]
            new_board[from_row, from_col] = 0
            if proposed_new_row == 7:
                new_board[proposed_new_row, proposed_new_column] = 2
            else:
                new_board[proposed_new_row, proposed_new_column] = moved_piece
            # return an action instead of a state, since an action holds a reference to previous action for repeated
            # jumps
            # in this case of moving, it's the same, but we prefer unified interface
            action = CheckerAction(self, new_board, self.flipped, False)
            if debug:
                assert ((self.board > 0).sum() == (action.board > 0).sum())
                assert ((self.board < 0).sum() >= (action.board < 0).sum())
            return action

    def get_legal_jumps(self):
        legal_jumps = []
        jump_tree = {}
        # for each piece of the player, try jump in both directions
        for row in range(0, 8):
            for column in range(0, 8):
                my_piece = self.board[row, column]
                if my_piece == 1:
                    # recursive helper function
                    # returns all legal jumps from this man
                    lj1, jt1 = self.get_legal_man_jump_actions(row, column)
                    legal_jumps += lj1
                    if jt1 is not None:
                        jump_tree.update(jt1)
                if my_piece == 2:
                    lj2, jt2 = self.get_legal_king_jump_actions(row, column)
                    legal_jumps += lj2
                    if jt2 is not None:
                        jump_tree.update(jt2)
        return legal_jumps, jump_tree

    def get_legal_man_jump_actions(self, from_row, from_col):
        """

        :param from_row:
        :param from_col:
        :return:
        legal_jumps: all reachable states starting from self state
        jump_tree: a dictionary {(from_row, from_col, proposed_new_row, proposed_new_column):
                                 (action_state, future_jump_tree)}

                   where key is every (from_row, from_col, to_row, to_col) actionable from
                   self state and its value is its resulting state and another jump tree dictionary,
                   None if no future actions from the resulting state.
        """
        # recursive call to try jump another time
        # return all recursive Actions
        # you must call recursive action not from this object, but the object returned from the action.

        # jump left forward
        legal_jumps = []
        immediate_children = {}

        # two possible direction
        for delta_c in (-2, 2):
            proposed_new_row = from_row + 2
            proposed_new_column = from_col + delta_c

            # get jump of this line
            jump = self.construct_action_after_man_jump_if_legal(from_row, from_col, proposed_new_row,
                                                                 proposed_new_column)
            # if jump is successful, jump again
            if jump is not None:
                legal_jumps += [jump]
                if jump[proposed_new_row, proposed_new_column] == 1:
                    future_jumps, future_immediate_children = \
                        jump.get_legal_man_jump_actions(proposed_new_row, proposed_new_column)
                    if debug:
                        if len(future_jumps) == 0:
                            assert future_immediate_children is None
                    immediate_children[(from_row, from_col, proposed_new_row, proposed_new_column)] = \
                        (jump, future_immediate_children)
                    legal_jumps += future_jumps
                    if len(future_jumps) != 0:
                        jump.is_terminal = False
                else:
                    immediate_children[(from_row, from_col, proposed_new_row, proposed_new_column)] = (jump, None)
                    # if a jumping man becomes the king, then it stops jumping
        if len(immediate_children) == 0:
            immediate_children = None
            if debug:
                assert len(legal_jumps) == 0
        return legal_jumps, immediate_children

    def construct_action_after_man_jump_if_legal(self, from_row, from_col, proposed_new_row, proposed_new_column):

        piece = self.board[from_row, from_col]
        if piece != 1:
            raise ValueError("Jumping a man that is not a man")
        r_delta = (2,)
        c_delta = (2, -2)
        if proposed_new_row - from_row not in r_delta:
            print("Invalid move")
            return None
        if proposed_new_column - from_col not in c_delta:
            print("Invalid move")
            return None

        taken_row = (from_row + proposed_new_row) // 2
        taken_col = (from_col + proposed_new_column) // 2

        # if the jump target is empty and on board and if the jump jumps over an enemy piece
        if self.is_on_board(proposed_new_row, proposed_new_column) and \
                self[taken_row, taken_col] < 0 and \
                self[proposed_new_row, proposed_new_column] == 0:
            new_board = np.copy(self.board)
            new_board[taken_row, taken_col] = 0
            new_board[from_row, from_col] = 0
            if proposed_new_row == 7:
                new_board[proposed_new_row, proposed_new_column] = 2
            else:
                new_board[proposed_new_row, proposed_new_column] = 1
            action = CheckerAction(self, new_board, self.flipped, True)
            if debug:
                assert ((self.board > 0).sum() == (action.board > 0).sum())
                assert ((self.board < 0).sum() >= (action.board < 0).sum())
            return action

    def get_legal_king_jump_actions(self, from_row, from_col):
        """

        :param from_row:
        :param from_col:
        :return:
        legal_jumps: all reachable states starting from self state
        immediate_children: a dictionary {(from_row, from_col, proposed_new_row, proposed_new_column):
                                 (action_state, future_immediate_children)}

                   where key is every (from_row, from_col, to_row, to_col) actionable from
                   self state and its value is its resulting state and another jump tree dictionary,
                   None if no future actions from the resulting state.
        """
        legal_jumps = []
        immediate_children = {}

        for delta_r in (-2, 2):
            for delta_c in (-2, 2):
                proposed_new_row = from_row + delta_r
                proposed_new_column = from_col + delta_c
                jump = self.construct_action_after_king_jump_if_legal(from_row, from_col, proposed_new_row,
                                                                      proposed_new_column)
                # if jump is successful, jump again
                if jump is not None:
                    legal_jumps += [jump]
                    future_jumps, future_immediate_children = \
                        jump.get_legal_king_jump_actions(proposed_new_row, proposed_new_column)
                    immediate_children[(from_row, from_col, proposed_new_row, proposed_new_column)] = \
                        (jump, future_immediate_children)
                    if debug:
                        if len(future_jumps) == 0:
                            assert future_immediate_children is None
                    legal_jumps += future_jumps
                    if len(future_jumps) == 0:
                        jump.is_terminal = True
                    else:
                        jump.is_terminal = False

        if len(immediate_children) == 0:
            immediate_children = None
            if debug:
                assert len(legal_jumps) == 0
        return legal_jumps, immediate_children

    def construct_action_after_king_jump_if_legal(self, from_row, from_col, proposed_new_row, proposed_new_column):

        piece = self.board[from_row, from_col]
        if piece != 2:
            raise ValueError("Jumping a king that is not a king")
        r_delta = (2, -2)
        c_delta = (2, -2)
        if proposed_new_row - from_row not in r_delta:
            print("Invalid move")
            return None
        if proposed_new_column - from_col not in c_delta:
            print("Invalid move")
            return None

        taken_row = (from_row + proposed_new_row) // 2
        taken_col = (from_col + proposed_new_column) // 2

        # if the jump does not go out of the board, and if the jump jumps over an enemy piece
        if self.is_on_board(proposed_new_row, proposed_new_column) and \
                self[taken_row, taken_col] < 0 and \
                self[proposed_new_row, proposed_new_column] == 0:
            new_board = np.copy(self.board)
            new_board[taken_row, taken_col] = 0
            new_board[from_row, from_col] = 0
            new_board[proposed_new_row, proposed_new_column] = 2
            action = CheckerAction(self, new_board, self.flipped, True)
            if debug:
                assert ((self.board > 0).sum() == (action.board > 0).sum())
                assert ((self.board < 0).sum() >= (action.board < 0).sum())
            return action

    @staticmethod
    def is_on_board(proposed_new_row, proposed_new_column):
        if (proposed_new_row + proposed_new_column) % 2 != 0:
            return False
        if proposed_new_column < 0 or proposed_new_column > 7:
            return False
        if proposed_new_row < 0 or proposed_new_row > 7:
            return False
        return True

    def assert_consistency(self):
        for row in range(0, 8):
            for col in range(0, 8):
                if self.board[row, col] != 0:
                    assert (row + col) % 2 == 0

    def is_multi_jump(self):
        return False

    def print_board(self, player_view=False):
        if player_view:
            if self.flipped:
                ret = str(np.flip(self.board, axis=0) * -1)
            else:
                ret = str(np.flip(self.board, axis=0))
        else:
            if self.flipped:
                board = np.rot90(self.board, 2) * -1
            else:
                board = self.board
            ret = str(np.flip(board, axis=0))

        ret = ret.replace("0", " ")
        ret = ret.split("\n")
        for i in range(len(ret)):
            ret[i] = str(7 - i) + ret[i]
        ret += ["    0  1  2  3  4  5  6  7 "]

        if player_view:
            ret = ["flipped: " + str(self.flipped)] + ret

        for s in ret:
            print(s)

    def __repr__(self):
        return repr(self.board)

    def __getitem__(self, row_col):
        assert (len(row_col) == 2)
        return self.board[row_col[0], row_col[1]]

    def __setitem__(self, row_col, v):
        assert (len(row_col) == 2)
        self.board[row_col[0], row_col[1]] = v

    def board_hash(self):
        # for if in collection call to avoid redundant searches from the same state
        return hash(str(self.board))

    def get_flipped_state(self):
        new_board = np.copy(self.board)
        # this is where the bug is, 03/28/2020
        new_board = np.rot90(new_board, 2)
        new_board *= -1
        new_state = CheckerState(new_board, flipped=not self.flipped)
        return new_state

    def evaluate(self):
        if not self.flipped:
            return self.board.sum()
        else:
            return -self.board.sum()


class CheckerAction(CheckerState):
    # Action is a state with a reference to the previous action (state)
    def __init__(self, old_state, new_board, flipped, capture):
        super(CheckerAction, self).__init__(new_board, flipped)
        # from what state did the action take place
        # if it is a multi jump, this old_state must be an Action object
        self.old_state = old_state
        # an action does not flip the board, the Checker object flips the board to alpha-beta prune
        assert (self.old_state.flipped == self.flipped)
        # default for moves, guarantees override by jumps
        self.is_terminal = True
        self.is_action = True
        self.is_capture = capture

    def is_multi_jump(self):
        if self.old_state.is_action:
            return True
        else:
            return False

from unittest import TestCase
from checker import *

class TestChecker(TestCase):
    def test_init_new_board(self):
        checker = Checker()
        checker.init_new_board()
        print(checker)

        checker.state[0, 0] = 0
        checker.state[0, 4] = 100
        print(checker)

        checker.flip_board()
        print(checker)
        # player_play_verbose(checker.state)
        print(checker.state.get_legal_actions())
        # player_play_verbose(checker.state)
        original_state=checker.state
        checker.flip_board()
        checker.flip_board()
        assert checker.state==original_state


    def test_multi_jump(self):
        # fake_board = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
        #                        [0, -1, 0, 0, 0, 0, 0, -1],
        #                        [1, 0, 0, 0, 0, 0, 0, 0],
        #                        [0, 0, 0, -1, 0, -1, 0, 0],
        #                        [0, 0, 0, 0, 0, 0, 0, 0],
        #                        [0, 0, 0, -1, 0, -1, 0, 0],
        #                        [0, 0, 0, 0, 0, 0, 0, 0],
        #                        [0, 0, 0, 0, 0, 0, 0, 0]])
        fake_board=np.array([[ 1, 0, 0, 0, 0, 0, 0, 0],
                             [ 0,-1, 0, 0, 0, 0, 0,-1],
                             [ 1, 0, 0, 0, 0, 0, 0, 0],
                             [ 0, 0, 0,-1, 0,-1, 0, 0],
                             [ 0, 0, 0, 0, 0, 0, 0, 0],
                             [ 0, 0, 0,-1, 0,-1, 0, 0],
                             [ 0, 0, 0, 0, 0, 0, 0, 0],
                             [ 0, 0, 0, 0, 0, 0, 0, 0]])
        checker = Checker()
        checker.state.board=fake_board
        print(checker)
        actions, move_tree=checker.state.get_legal_actions()
        checker.print_moves(move_tree)

    def test_man_becomes_king(self):
        fake_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, -1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, -1, 0, -1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, -1, 0, -1, 0, -1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
        checker = Checker()
        checker.state.board = fake_board
        print(checker)
        actions, move_tree = checker.state.get_legal_actions()
        checker.print_moves(move_tree)


    def test_king_multi_jump(self):
        fake_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, -1, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [-1, 0, 0, 0, 0, 0, -1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 2]])
        checker = Checker()
        checker.state.board = fake_board
        print(checker)
        actions, move_tree = checker.state.get_legal_actions()
        checker.print_moves(move_tree)

    def test_fast_inputs(self):
        fake_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, -1, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [-1, 0, 0, 0, 0, 0, -1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 2]])
        path = [(7, 7, 5, 5), (5, 5, 3, 3), (3, 3, 5, 1)]
        checker = Checker()
        checker.state.board = fake_board
        final = checker.player_play_fast_enter(checker.state, path)


    def test_fast_inputs(self):
        fake_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, -1, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [-1, 0, 0, 0, 0, 0, -1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 2]])
        path = [(7, 7, 5, 5), (5, 5, 3, 3), (3, 3, 5, 1)]
        checker = Checker()
        print(checker)
        checker.state.board = fake_board
        final = checker.ask_human_for_action(checker.state)

    def test_get_legal_actions(self):
        fake_board = np.array([[1, 0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0, 1],
                               [0, 0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, -1, 0, 0, 0, 0, 0],
                               [0, -1, 0, 0, 0,-1, 0, -1],
                               [-1, 0,-1, 0,-1, 0, -1, 0],
                               [0, -1, 0,-1, 0, 0, 0, -1]])
        checker = Checker()
        print(checker)
        checker.state.board = fake_board
        checker.ask_human_for_action(checker.state)
        checker.state.get_legal_actions()

if __name__ == '__main__':
    tc=TestChecker()
    tc.test_init_new_board()
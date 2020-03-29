from unittest import TestCase
from alphabeta import *

class TestAlphaBetaNode(TestCase):
    def test_evaluate(self):
        # with the board flipped, the evaluation should be the same
        fake_board = np.array([[1, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, -1, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [-1, 0, 0, 0, 0, 0, -1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
        state=CheckerState(fake_board, False)
        alpha=AlphaNode(state)
        val=alpha.evaluate()

        alpha.make_children()
        for child in alpha.children:
            self.assertTrue(child.evaluate()== val)

    def test_game(self):
        ab = AlphaBeta(4)
        ab.print_board()
        while True:
            ret = ab.auto_play(True)
            if not ret:
                break
        print("Game finished")
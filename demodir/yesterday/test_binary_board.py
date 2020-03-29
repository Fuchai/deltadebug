from unittest import TestCase
from neuralnetwork import batch_board
import numpy as np

class TestBinary_board(TestCase):
    def test_binary_board(self):
        fake_board = np.array([[1, 0, 1, 0, 0, 2, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, -1, 0, -1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [-1, 0, 0, 0, 0, 0, -1, 0],
                               [0, 0, 0, -2, 0, 0, 0, 0]])
        binary=batch_board(fake_board)
        print(binary)

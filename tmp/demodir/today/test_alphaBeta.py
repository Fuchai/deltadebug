from unittest import TestCase
from alphabeta import *

class TestAlphaBeta(TestCase):
    def test_first_player_advantage(self):
        ab = AlphaBeta(5, 3)
        ab.start_game()

    def test_second_player_advantage(self):
        ab = AlphaBeta(3, 5)
        ab.start_game()

    def test_human_first_player(self):
        ab = AlphaBeta(5, 3, human_first=True)
        ab.start_game()

    def test_human_second_player(self):
        ab = AlphaBeta(5, 3, human_second=True)
        ab.start_game()
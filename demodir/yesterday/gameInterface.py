from performance import *


def human_first_play():
    depth=5
    ab = AlphaBeta(0, depth, human_first=True)
    ab.start_game()

def human_second_play():
    depth=5
    ab = AlphaBeta(depth, 0, human_second=True)
    ab.start_game()

def play_with_your_friend():
    ab = AlphaBeta(0, 0, human_first=True, human_second=True)
    ab.start_game()

def machine_versus_machine():
    # first player (alpha) probably wins
    alpha_depth=5
    beta_depth=3
    ab = AlphaBeta(alpha_depth, beta_depth, max_rounds=300)
    ab.start_game()



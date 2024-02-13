from environment import GoBoard

import numpy as np

class Agent:
    def __init__(self, color, policy):
        self.color = color
        self.policy = policy
        if abs(color) != 1:
            raise ValueError("color must be 1 or -1")
    
    def act(self, board: GoBoard):
        # get legal moves
        legal_moves = board.get_legal_moves(self.color)

        # use policy to select a move
        move = self.policy(board, self.color)

        # check that the move is legal
        if move not in legal_moves:
            # warn the user
            print("Illegal move chosen: " + str(move))
            return "resign"
        
        # make the move
        return move
    
def random_move(board: GoBoard, color):
    """
    Choose a random legal move.
    """

    legal_moves = board.get_legal_moves(color)
    legal_moves.remove("resign")
    return np.random.choice(legal_moves)   

def capture_first(board: GoBoard, color):
    """
    If there is a move that captures a stone, play it. Otherwise, play randomly.
    Also, don't pass unless there are no other legal moves.
    """

    legal_moves = board.get_legal_moves(color)
    legal_moves.remove("resign")
    legal_moves.remove("pass")
    for move in legal_moves:
        x, y = board._coord_to_pos(move)
        if board.check_captures(x, y, color, check_only=True):
            return move
    if legal_moves:
        return np.random.choice(legal_moves)
    return "pass"

def random_diagonal(board: GoBoard, color):
    """
    Play randomly on every turn, but prioritize playing on coordinates to place stones on diagonals
    """
    legal_moves = board.get_legal_moves(color)
    legal_moves.remove("resign")
    diagonal_moves = []
    for move in legal_moves:
        if move == "pass":
            continue
        x, y = board._coord_to_pos(move)
        if x % 2 == y % 2:
            diagonal_moves.append(move)
    if diagonal_moves:
        return np.random.choice(diagonal_moves)
    return np.random.choice(legal_moves)
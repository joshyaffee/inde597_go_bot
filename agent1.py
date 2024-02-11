"""
Agent super class for Go game
subclasses will implement the policy function
"""

import environment1 as env
import numpy as np
import tensorflow as tf

class Agent:
    def __init__(self, color, policy = None, board: env.GoBoard = None):
        self.current_color = color
        self.board = board
        self.board_size = board.size
        if policy is None:
            policy = self._init_policy()
        self.policy = policy
        if abs(color) != 1:
            raise ValueError("color must be 1 or -1")
        
    
    def act(self, board: env.GoBoard):
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
    
    def _init_policy(self):
        pass # to be implemented by subclasses
    
    def _get_legal_moves(self, board):
        return board.get_legal_moves(self.color)
    
class RandomAgent(Agent):
    def _init_policy(self):
        legal_moves = self._get_legal_moves().remove("resign")
        return np.random.choice(legal_moves)
    
class CaptureFirstAgent(Agent):
    def _init_policy(self):
        legal_moves = self._get_legal_moves().remove("resign")
        for move in legal_moves:
            x, y = self.board._coord_to_pos(move)
            if self.board.check_captures(x, y, self.color, check_only=True):
                return move
            
class RandomDiagonalAgent(Agent):
    def _init_policy(self):
        legal_moves = self._get_legal_moves().remove("resign")
        diagonal_moves = []
        for move in legal_moves:
            x, y = self.board._coord_to_pos(move)
            if x % 2 == y % 2:
                diagonal_moves.append(move)
        if diagonal_moves:
            return np.random.choice(diagonal_moves)
        return np.random.choice(legal_moves)
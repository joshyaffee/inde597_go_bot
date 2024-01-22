########################################################################
#   IMPORTANT: SCORING USES CHINESE RULES AND ASSUMES NO DEAD STONES   #
#                                                                      #    
#   THIS CAN BE PROBLEMATIC SINCE IT IS NO LONGER DISCOURAGED TO PLAY  #
#   UNNECESSARY MOVES TO FILL IN YOUR OWN TERRITORY... OR YOUR         #
#   OPPONENT'S TERRITORY AFTER THE GAME IS RESOLVED STRATEGICALLY.     #
#                                                                      #
#   (The website--and most people--use Japanese rules, so this is a    #
#    serious issue, and should be addressed ASAP)                      #
########################################################################

"""
Code currently implemented for user to play against itself, but can be easily modified for bot self-play.
Adjust BOARD_SIZE to your liking. You can also adjust komi within the GoGame class instantiation, but it 
defaults to the website's default for level 1+ games.
"""

BOARD_SIZE = 9

import networkx as nx
import numpy as np
import re

class GoGame:
    def __init__(self, board_size: int, komi: float = -1):
        if board_size < 5 or board_size > 19:
            raise ValueError("board_size must be between 5 and 19")
        self.board_size = board_size
        if komi < 0:
            komi = 0 if self.board_size < 9 else 6.5
        self.komi = komi
        self.board = GoBoard(self.board_size, self)
        self.history = [([0]*self.board_size, [0]*self.board_size)]
        self.turn = 1 # 1: black, -1: white
        self.passes = 0
        self.captures = {1: 0, -1: 0}
        self.game_over = False
        self.black_score = None
        self.white_score = None

    def play_move(self, coord):
        coord = coord.upper()
        if coord == "PASS":
            self.passes += 1
            self.turn = -self.turn
            self.history.append((self.board.black_ints, self.board.white_ints))
            if self.passes == 2:
                self.black_score, self.white_score = self.board.get_score_chinese()
                print(f"Game over! Score is: Black {self.black_score} - White {self.white_score}")
                self.game_over = True
            return True
        
        elif coord == "RESIGN":
            self.history.append((self.board.black_ints, self.board.white_ints))
            print("Game over!")
            self.black_score = -1 * self.turn
            self.white_score = 0
            self.game_over = True
            return True
        
        else:
            if self.board.add_stone(coord, self.turn):
                self.passes = 0
                self.turn = -self.turn
                self.history.append((self.board.black_ints, self.board.white_ints))
                return True
            else:
                return False
            
    def user_play(self):
        colors = {1: "Black", -1: "White"}
        while not self.game_over:
            this_turn = self.turn
            self.board.print_board()
            while self.turn == this_turn and not self.game_over:
                coord = input(f"{colors[self.turn]}'s turn. Enter a valid move:")
                # use regex to check if coord is letter number, also check if it's on the board
                if re.match(r"[A-Za-z][0-9]+", coord) or coord.upper() == "PASS" or coord.upper() == "RESIGN":
                    try:
                        _ = self.board._coord_to_pos(coord)
                    except:
                        print("That is not on the board. Please enter a valid move such as 'A1' or 'pass' or 'resign'.")

                    if not self.play_move(coord):
                            print("Invalid move!")
                else:
                    print("Invalid input. Please enter a valid move such as 'A1' or 'pass' or 'resign'.")

        if self.black_score > self.white_score:
            print("Black wins!")
        elif self.white_score > self.black_score:
            print("White wins!")
        else:
            print("Tie!")
                
class GoBoard:
    def __init__(self, size: int, game: GoGame):
        self.size = size
        self.graph_board = self._init_graph_board()
        self.game = game
        self.matrix_board = np.zeros((size, size), dtype=np.int8)
        self.white_ints = [0] * size
        self.black_ints = [0] * size

    def __eq__(self, other):
        return self.white_ints == other.white_ints and self.black_ints == other.black_ints
        # unused, since we only store the ints in history, but could be useful

    def _init_graph_board(self):
        graph_board = nx.grid_2d_graph(self.size, self.size)
        for node in graph_board.nodes:
            graph_board.nodes[node]["color"] = 0
        return graph_board
    
    def get_color(self, coord):
        x, y = self._coord_to_pos(coord)
        return self.graph_board.nodes[(x, y)]["color"]

    def _coord_to_pos(self, coord):
        # A1 -> (0, 0), S1 -> (18, 0)
        x = ord(coord[0]) - ord("A")
        y = int(coord[1:]) - 1
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            raise ValueError("Invalid coordinate")
        return x, y
    
    def _pos_to_coord(self, x, y):
        # (0, 0) -> A1, (18, 0) -> S1
        return chr(x + ord("A")) + str(y + 1)

    def add_stone(self, coord, color):
        x, y = self._coord_to_pos(coord)
        capture_num = self.check_captures(x, y, color, check_only=False)
        self.game.captures[color] += capture_num
        if self.check_liberties(x, y, color)[0] or capture_num > 0:
            self.graph_board.nodes[(x, y)]["color"] = color
            self.matrix_board[y, x] = color
            self.update_ints()
            return True
        else:
            return False 
        
    def check_liberties(self, x, y, color):
        liberties = []
        visted = {(x, y)}
        queue = [(x, y)]
        visited_this_color = {(x, y)}
        while queue:
            x, y = queue.pop()
            visted.add((x, y))
            for neighbor in self.graph_board.neighbors((x, y)):
                if neighbor not in visted:
                    if self.graph_board.nodes[neighbor]["color"] == 0:
                        liberties.append(neighbor)
                    elif self.graph_board.nodes[neighbor]["color"] == color:
                        queue.append(neighbor)
                        visited_this_color.add(neighbor)
        return liberties, visited_this_color

    def check_captures(self, x, y, color, check_only=False):
        neigbors = self.graph_board.neighbors((x, y))
        capture_num = 0
        for neighbor in neigbors:
            if self.graph_board.nodes[neighbor]["color"] == -color: # this logic is wrong
                liberties, group = self.check_liberties(neighbor[0], neighbor[1], -color)
                if len(set(liberties) - {(x,y)}) == 0:
                    if not check_only:
                        capture_add = self.capture_stones(group, x, y)
                        capture_num += capture_add
                    else:
                        return 1
        return capture_num
      
    def capture_stones(self, group, x, y):
        # capture stones, check for ko rule, return number of captured stones
        possible_board = self.graph_board.copy()
        possible_matrix_board = self.matrix_board.copy()

        possible_board.nodes[(x, y)]["color"] = self.game.turn
        possible_matrix_board[y, x] = self.game.turn
        for stone in group:
            possible_board.nodes[stone]["color"] = 0
            possible_matrix_board[stone[1], stone[0]] = 0
            possible_ints = self.update_ints(board=possible_matrix_board, in_place=False)
        if possible_ints in self.game.history:
            return 0 # ko rule
        else:
            self.graph_board = possible_board
            self.matrix_board = possible_matrix_board
        return len(group)

    def update_ints(self, board = None, in_place = True):
        white_ints = [0] * self.size
        black_ints = [0] * self.size
        if board is None:
            board = self.matrix_board
        for i in range(self.size):
            for j in range(self.size):
                if board[i, j] == 1:
                    black_ints[i] += 1 << j
                elif board[i, j] == -1:
                    white_ints[i] += 1 << j
        if in_place:
            self.white_ints = white_ints
            self.black_ints = black_ints
        else:
            return black_ints, white_ints

    def get_legal_moves(self, color):
        legal_moves = []
        for node in self.graph_board.nodes:
            if self.graph_board.nodes[node]["color"] == 0:
                if self.check_liberties(node[0], node[1], color)[0] or \
                   self.check_captures(node[0], node[1], color, check_only = True) > 0:
                    
                    legal_moves.append(node)
        legal_moves.append("pass")
        legal_moves.append("resign")
        return legal_moves.map(self._pos_to_coord)
    
    def print_board(self):
        print("  ", end="")
        for i in range(self.size):
            print(chr(ord("A") + i), end=" ")
        print()
        for i in range(self.size):
            print(i + 1, end=" ")
            for j in range(self.size):
                if self.matrix_board[i, j] == 0:
                    print("_", end=" ")
                elif self.matrix_board[i, j] == 1:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print(i + 1)
        print("  ", end="")
        for i in range(self.size):
            print(chr(ord("A") + i), end=" ")
        print()

    def get_score_chinese(self):
        """
        Chinese scoring: area occupied or surrounding, captures don't count. Komi is added to white's score.
        """
        black_area = 0
        white_area = 0
        unvisited = set(self.graph_board.nodes)
        while unvisited:
            node = unvisited.pop()
            if self.graph_board.nodes[node]["color"] == 1:
                black_area += 1
            elif self.graph_board.nodes[node]["color"] == -1:
                white_area += 1
            else:
                ownership, area = self.check_territory(node[0], node[1])
                if ownership == 1:
                    black_area += len(area)
                elif ownership == -1:
                    white_area += len(area)
                unvisited -= area
        return black_area, white_area + self.game.komi
    
    def check_territory(self, i, j):
        black_flag = False
        white_flag = False
        queue = [(i, j)]
        visited = {(i, j)}
        visited_empty = {(i, j)}
        while queue:
            x, y = queue.pop()
            for neighbor in self.graph_board.neighbors((x, y)):
                if neighbor not in visited:
                    if self.graph_board.nodes[neighbor]["color"] == 1:
                        black_flag = True
                    elif self.graph_board.nodes[neighbor]["color"] == -1:
                        white_flag = True
                    else:
                        queue.append(neighbor)
                        visited_empty.add(neighbor)
                    visited.add(neighbor)
        if black_flag and white_flag:
            return 0, visited_empty
        else:
            # only time when both are false is starting position
            return (1, visited_empty) if black_flag else (-1, visited_empty) 
        
    def check_life(self, group):
        # TODO: check life of groups
        pass

if __name__ == "__main__":
    game = GoGame(BOARD_SIZE)
    game.user_play()

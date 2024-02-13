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

# constants
BOARD_SIZE = 9
BOT_PLAY = True

import networkx as nx
import numpy as np
import re

class GoGame:
    """
    Class to represent a game of Go.

    Constructor Inputs:
        - board_size: int, size of board (5-19)
        - komi: float, number of points white gets for going second (default: -1, which
          means use website's default)
    
    Attributes:
        - board_size: int, size of board (5-19)
        - komi: float, number of points white gets for going second
        - board: GoBoard object, represents the board
        - history: list of tuples of ints, represents the board at each turn
        - turn: int, 1 for black, -1 for white
        - passes: int, number of passes in a row (2 passes -> game over)
        - captures: dict, number of captures for each color
        - game_over: bool, True if game is over
        - black_score: int, score for black (using Chinese scoring)
        - white_score: int, score for white (using Chinese scoring)

    """
    def __init__(self, board_size: int, komi: float = -1):
        if board_size < 5 or board_size > 19:
            raise ValueError("board_size must be between 5 and 19")
        self.board_size = board_size

        # komi is the number of points white gets for going second
        if komi < 0:
            komi = 0 if self.board_size < 9 else 6.5
        self.komi = komi

        self.board = GoBoard(self.board_size, self)

        # example state: ([1, 0, 0, 0, 0], [0, 0, 0, 0, 16]) -> black has a stone at A1, white has a stone at E5
        self.history = [([0]*self.board_size, [0]*self.board_size)]
        self.turn = 1 # 1: black, -1: white
        self.passes = 0
        self.captures = {1: 0, -1: 0}
        self.game_over = False
        self.black_score = None
        self.white_score = None

    def play_move(self, coord) -> bool:
        """
        Attempts to play a move at coord. Returns True if successful, False otherwise.

            - coord: string, e.g. "A1", "S1", "pass", "resign"
        """
        coord = coord.upper()
        if coord == "PASS":
            self.passes += 1
            self.turn = -self.turn
            self.history.append((self.board.black_ints, self.board.white_ints))
            if self.passes == 2: # 2 passes -> game over
                self.black_score, self.white_score = self.board.get_score_chinese()
                # probably comment this out for bot self-play
                print(f"Game over! Score is: Black {self.black_score} - White {self.white_score}")
                self.game_over = True
            return True
        
        elif coord == "RESIGN": # we may want to remove this option for bot self-play
            self.history.append((self.board.black_ints, self.board.white_ints))
            print("Game over!")
            self.black_score = -1 * self.turn
            self.white_score = 0
            self.game_over = True
            return True
        
        else:
            # check if move is valid, and make move if it is
            if self.board.add_stone(coord, self.turn):

                # reset passes
                self.passes = 0

                # switch turns
                self.turn = -self.turn

                # add to history
                self.history.append((self.board.black_ints, self.board.white_ints))
                return True
            else:
                return False
            
    def user_play(self):
        """
        Allows user to play against itself. Prints board after each move.
        """

        colors = {1: "Black", -1: "White"}
        # game loop
        while not self.game_over:
            this_turn = self.turn
            self.board.print_board()

            # turn loop (until valid move is played)
            while self.turn == this_turn and not self.game_over:
                coord = input(f"{colors[self.turn]}'s turn. Enter a valid move:")
                # use regex to check if coord is letter number, also check if it's on the board
                if re.match(r"[A-Za-z][0-9]+", coord) or coord.upper() == "PASS" or coord.upper() == "RESIGN":
                    # in case something like Z1000 is entered:
                    try:
                        _ = self.board._coord_to_pos(coord)
                    except:
                        print("That is not on the board. Please enter a valid move such as 'A1' or 'pass' or 'resign'.")
                    
                    # play the move
                    if not self.play_move(coord):
                            print("Invalid move!")
                else:
                    print("Invalid input. Please enter a valid move such as 'A1' or 'pass' or 'resign'.")

        # print result
        if self.black_score > self.white_score:
            print("Black wins!")
        elif self.white_score > self.black_score:
            print("White wins!")
        else:
            print("Tie!")
    
    def bot_play(self, black_policy, white_policy, print_board: bool = False):
        """
        Allows two agents to play against each other.
        """
        from agent import Agent

        black_agent = Agent(1, black_policy)
        white_agent = Agent(-1, white_policy)

        agents = {1: black_agent, -1: white_agent}
        my_turn = 1 # 1: black, -1: white
        while not self.game_over:
            if print_board:
                self.board.print_board()
                print("\n")
            move = agents[my_turn].act(self.board)
            if move == "resign" and print_board:
                print("Game over. " + ("Black" if my_turn == 1 else "White") + " resigns.")
            if move == "pass" and print_board:
                print("Pass.")
            if not self.play_move(move):
                raise ValueError("Invalid move: " + str(move))
            else:
                my_turn *= -1
        if self.black_score > self.white_score:
            return 1
        elif self.white_score > self.black_score:
            return -1
        else:
            return 0
                
class GoBoard:
    """
    Class to represent the board.
    
    Constructor Inputs:
        - size: int, size of board (5-19)
        - game: GoGame object, represents the game
        
    Attributes:
        - size: int, size of board (5-19)
        - graph_board: networkx graph, represents the board
        - matrix_board: numpy array, represents the board
        - white_ints: list of ints, represents the board
        - black_ints: list of ints, represents the board
        - game: GoGame object, represents the game
    """

    def __init__(self, size: int, game: GoGame):
        # size of board (5-19)
        self.size = size
        # networkx graph, nodes are tuples of ints, edges represent adjacency, color represents stone color
        self.graph_board = self._init_graph_board() 
        # GoGame object, represents the game
        self.game = game
        # numpy array, 1 for black, -1 for white, 0 for empty
        self.matrix_board = np.zeros((size, size), dtype=np.int8)
        # for each color: list of ints, each int represents a row 
        self.white_ints = [0] * size 
        self.black_ints = [0] * size 

    def __eq__(self, other):
        """
        Two GoBoards are equal if their white_ints and black_ints are equal. 
        Captured stones and turn are not considered.

        This is unused currently
        """
        return self.white_ints == other.white_ints and self.black_ints == other.black_ints
        # unused, since we only store the ints in history, but could be useful

    def _init_graph_board(self):
        """
        Initializes the graph_board attribute.
        
        Nodes are represented by tuples of ints, e.g. (0, 0) is A1, (18, 0) is S1, (0, 18) is A19, etc.
        Edges represent adjacency on the board.
        Node attributes:
            - color: int, 1 for black, -1 for white, 0 for empty
        """

        graph_board = nx.grid_2d_graph(self.size, self.size)
        for node in graph_board.nodes:
            graph_board.nodes[node]["color"] = 0
        return graph_board
    
    def get_color(self, coord):
        """
        Returns the color at coord.
        I don't think this is used anywhere, but it could improve readability.
        """
        x, y = self._coord_to_pos(coord)
        return self.graph_board.nodes[(x, y)]["color"]

    def _coord_to_pos(self, coord):
        # A1 -> (0, 0), S1 -> (18, 0)
        x = ord(coord[0].upper()) - ord("A") # typically "I" is skipped. We can worry about this later.
        y = int(coord[1:]) - 1
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            raise ValueError("Invalid coordinate")
        return x, y
    
    def _pos_to_coord(self, x, y):
        # (0, 0) -> A1, (18, 0) -> S1
        # typically "I" is skipped. We can worry about this later.
        return chr(x + ord("A")) + str(y + 1)

    def add_stone(self, coord, color):
        """
        Attempts to add a stone at coord. Returns True if successful, False otherwise.
        
        If successful, updates the board and game.
        
        Inputs:
            - coord: string, e.g. "A1", "S1"
            - color: int, 1 for black, -1 for white
        """

        # get numerical position, check for legal capture, capture if valid
        x, y = self._coord_to_pos(coord)
        capture_num = self.check_captures(x, y, color, check_only=False)
        self.game.captures[color] += capture_num

        # if stone has liberties or captures legally, add stone, update all representations, return True
        if self.check_liberties(x, y, color)[0] or capture_num > 0:
            self.graph_board.nodes[(x, y)]["color"] = color
            self.matrix_board[y, x] = color
            self.update_ints()
            return True
        else:
            return False 
        
    def check_liberties(self, x, y, color):
        """
        Checks liberties of a group of stones at (x, y) of color color.
        
        Inputs:
            - x: int, x-coordinate
            - y: int, y-coordinate
            - color: int, 1 for black, -1 for white
        
        Returns:
            - liberties: list of tuples of ints, each tuple represents a liberty
            - visited_this_color: list of tuples of ints, the group
        """

        # initialize BFS variables
        liberties = []
        visted = {(x, y)}
        queue = [(x, y)]

        # initialize group
        visited_this_color = {(x, y)}

        # BFS to find liberties
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
        """
        Checks for captures at (x, y) of color color.

        Inputs:
            - x: int, x-coordinate
            - y: int, y-coordinate
            - color: int, 1 for black, -1 for white
            - check_only: bool, if True, only checks for captures, doesn't actually capture
                                if False, captures if valid

        Returns:
            - capture_num: int, number of captures, or 1 if check_only and there is a capture
                (doesn't really matter in Chinese scoring, but could be useful)
        """

        neigbors = self.graph_board.neighbors((x, y))
        capture_num = 0

        # check each neighbor to see if it ran out of liberties + is of opposite color + is not a ko
        for neighbor in neigbors:
            if self.graph_board.nodes[neighbor]["color"] == -color:
                liberties, group = self.check_liberties(neighbor[0], neighbor[1], -color)
                # if it ran out of liberties, capture it / return 1 if check_only
                if len(set(liberties) - {(x,y)}) == 0:
                    capture_add = self.capture_stones(group, x, y, check_only=check_only)
                    capture_num += capture_add
        return capture_num
      
    def capture_stones(self, group, x, y, check_only=False):
        """
        Captures a group of stones, placing a stone on (x,y).

        Inputs:
            - group: list of tuples of ints, each tuple represents a stone being captured
            - x: int, x-coordinate of move
            - y: int, y-coordinate of move
        """
        # create possible game states for if capture is made
        possible_board = self.graph_board.copy()
        possible_matrix_board = self.matrix_board.copy()

        possible_board.nodes[(x, y)]["color"] = self.game.turn
        possible_matrix_board[y, x] = self.game.turn
        for stone in group:
            possible_board.nodes[stone]["color"] = 0
            possible_matrix_board[stone[1], stone[0]] = 0
            possible_ints = self.update_ints(board=possible_matrix_board, in_place=False)

        # check if ko rule is violated
        if possible_ints in self.game.history:
            return 0
        
        # if not, update board
        elif not check_only:
            self.graph_board = possible_board
            self.matrix_board = possible_matrix_board
        return len(group)

    def update_ints(self, board = None, in_place = True):
        """
        Finds the ints representation of the board. If in_place, updates the board's ints attributes.
        If not in_place, returns the ints representation. If no board is provided, uses the current board.
        
        Inputs:
            - board: numpy array, represents the board
            - in_place: bool, if True, updates the board's ints attributes
                            if False, returns the ints representation
        """

        white_ints = [0] * self.size
        black_ints = [0] * self.size
        if board is None:
            board = self.matrix_board
        for i in range(self.size):
            for j in range(self.size):
                # for each row i, the int is the sum of 2^j for each j such that board[i, j] == this color 
                if board[i, j] == 1:
                    black_ints[i] += 1 << j
                elif board[i, j] == -1:
                    white_ints[i] += 1 << j
        
        # update board's ints attributes if in_place, otherwise return ints
        if in_place:
            self.white_ints = white_ints
            self.black_ints = black_ints
        else:
            return black_ints, white_ints

    def get_legal_moves(self, color):
        """
        Returns a list of legal moves for color.
        
        Inputs:
            - color: int, 1 for black, -1 for white
                (This isn't really necessary, since we can just use self.game.turn, 
                    but it may be useful to see opponent's possible moves?)
            
        Returns:
            - legal_moves: list of strings, each string is a coordinate such as "A1" or "pass" or "resign"
        """

        legal_moves = []
        for node in self.graph_board.nodes:
            if self.graph_board.nodes[node]["color"] == 0:
                if self.check_liberties(node[0], node[1], color)[0] or \
                   self.check_captures(node[0], node[1], color, check_only = True) > 0:
                    
                    legal_moves.append(node)

        legal_moves = [self._pos_to_coord(*m) for m in legal_moves]
        legal_moves.append("pass")
        legal_moves.append("resign") # we may want to remove this option for bot self-play!!!
        return legal_moves
    
    def print_board(self):
        """
        Prints the board.
        X: black, O: white, _: empty
        """

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
        Chinese scoring: area occupied or surrounding, captures don't count. 
        Komi is added to white's score.

        Returns:
            - black_score: int, score for black
            - white_score: int, score for white (including komi)
        """
        black_area = 0
        white_area = 0
        unvisited = set(self.graph_board.nodes)
        # visit each node, check if it's black, white, or empty.
        # if empty, check if it's black territory, white territory, or neutral
        # There is no notion of life/death implemented yet, so we assume all stones are alive!
        while unvisited:
            node = unvisited.pop()
            if self.graph_board.nodes[node]["color"] == 1:
                black_area += 1
            elif self.graph_board.nodes[node]["color"] == -1:
                white_area += 1
            else:
                # call helper function to check territory and remove from unvisited
                ownership, area = self._check_territory(node[0], node[1])
                if ownership == 1:
                    black_area += len(area)
                elif ownership == -1:
                    white_area += len(area)
                unvisited -= area
        return black_area, white_area + self.game.komi
    
    def _check_territory(self, i, j):
        """
        Helper function for get_score_chinese.

        Inputs:
            - i: int, x-coordinate
            - j: int, y-coordinate

        Returns:
            - ownership: int, 1 for black, -1 for white, 0 for neutral
            - visited_empty: set of tuples of ints, empty nodes in territory
        """
        black_flag = False
        white_flag = False
        queue = [(i, j)]
        visited = {(i, j)}
        visited_empty = {(i, j)} # empty nodes in territory

        # BFS to check territory
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
            # only time when both are false is starting position,
            return (1, visited_empty) if black_flag else (-1, visited_empty) 
        
    def check_life(self, group):
        # TODO: check life of groups
        # THIS SEEMS SUPER IMPORTANT BUT QUITE HARD
        pass

if __name__ == "__main__":
    game = GoGame(BOARD_SIZE)
    if not BOT_PLAY:
        game.user_play()
    else:
        from agent import random_move, capture_first, random_diagonal
        
        # result = game.bot_play(random_diagonal, capture_first, print_board=True)

       #  play 100 games of bot vs bot
        results = []
        for i in range(100):
            result = game.bot_play(random_move, capture_first, print_board=False)
            results.append(result)
            print(f"Game {i+1}: {'Black' if result == 1 else 'White' if result == -1 else 'Draw'}")
        # print record of math (Black wins / Ties / White wins)
        print(f"Record: \nBlack Wins: {results.count(1)}\nDraws: {results.count(0)}\nWhite Wins: {results.count(-1)}")
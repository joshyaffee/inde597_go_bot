"""
Environment for a Go Board.
Methods will be modelled after OpenAI Gym's environment API when possible.
"""

import numpy as np
import networkx as nx

class GoBoard:

    graph_board = None
    matrix_board = None
    white_ints = None
    black_ints = None
    integer_representation = None

    def __init__(self, size: int = 9, komi: float = -1):
        self.size = size
        if komi < 0:
            self.komi = 0 if self.size < 9 else 6.5
        self.komi = komi
        self.history = []
        self.turn = 1 # 1: black, -1: white
        self.passes = 0
        self.captures = {1: 0, -1: 0}
        self.game_over = False
        self.black_score = None
        self.white_score = None
        self.matrix_board = np.zeros((size, size), dtype=int)
        self.init_graph_board()
        self.white_ints = [0] * size # each entry incodes a row of white stones
        self.black_ints = [0] * size # each entry incodes a row of black stones
        self.integer_representation = [self.white_ints, self.black_ints]
    
    def reset(self, board = None, turn = 1):
        """
        Resets the board to the initial state. If a board representation is provided
        """
        if board is None:
            self.__init__(self.size, self.komi)
        elif type(board) is np.ndarray:
            if board.shape != (self.size, self.size):
                raise ValueError("Board shape must match size of GoBoard environment")
            self.matrix_board = board

            # update graph representation
            self.init_graph_board(self.size)
            
            # update ints representation
            self.update_ints(board)

        elif type(board) is list:
            if len(board) != 2:
                raise ValueError("Board representation must be a list of two lists")
            if len(board[0]) != self.size or len(board[1]) != self.size:
                raise ValueError("Board representation must be a list of two lists of length equal to the size of the GoBoard environment")
            self.integer_representation = board
            self.update_matrix_from_ints(board)

        elif type(board) is nx.Graph:
            # check that the nodes are labeled correctly
            if set(board.nodes) != set([(i, j) for i in range(self.size) for j in range(self.size)]):
                raise ValueError("Graph nodes must be labeled with tuples of integers from 0 to size - 1")
            for node in board.nodes:
                if board.nodes[node]["color"] not in [-1, 0, 1]:
                    raise ValueError("Graph nodes must have a 'color' attribute with value -1, 0, or 1")
                else:
                    self.matrix_board[node[0], node[1]] = board.nodes[node]["color"]
            self.graph_board = board
            self.update_ints(board)

        else:
            raise ValueError("Invalid board representation")
        
        self.turn = turn
        self.history = []
        self.passes = 0
        self.captures = {1: 0, -1: 0}
        self.game_over = False
        self.black_score = None
        self.white_score = None

    def step(self, action, representation = 0):
        """
        Takes a step in the game.

        Inputs:
            - action: string, e.g. "A1", "S1", "pass", "resign"
            - representation: int, 0 for ints, 1 for matrix, 2 for graph

        Returns:
            - rep: list of ints or numpy array or networkx graph, depending on representation
                - ints representation: [black_ints, white_ints]; each list contains an int i
                  for each row such that the j-th bit is 1 if there is a black/white stone
                  in that position of the row.
                - matrix representation: numpy array of shape (size, size), 1 for black,
                  -1 for white, 0 for empty
                - graph representation: networkx graph, each node has a "color" attribute.
                  Node labels are tuples of ints (i, j) where i is the row and j is the column.
            - reward: int, 1 for black win, -1 for white win, 0 for draw (white wants to
              minimize value, black wants to maximize value)
            - game_over: bool, whether the game is over
        """
        coord = action.upper()

        if coord == "PASS":
            self.passes += 1
            self.turn *= -1
            self.history.append((self.black_ints, self.white_ints))
            if self.passes == 2:
                self.game_over = True
                self.black_score, self.white_score = self.get_score_chinese()

        elif coord == "RESIGN": # we may want to remove this option for bot self-play
            self.history.append((self.black_ints, self.white_ints))
            print("Game over!")
            self.black_score = -1 * self.turn
            self.white_score = 0
            self.game_over = True
        else:
            if self.add_stone(coord, self.turn):
                # reset passes
                self.passes = 0
                # switch turns
                self.turn = -self.turn
                # add to history
                self.history.append((self.black_ints, self.white_ints))
            else:
                raise ValueError("Invalid move")
                # or return very negative reward and same state

        if self.game_over:
            self.black_score, self.white_score = self.get_score_chinese()
        else:
            self.black_score, self.white_score = 0, 0

        if representation == 0:
            rep = self.integer_representation
        elif representation == 1:
            rep = self.matrix_board
        else:
            rep = self.graph_board

        reward = self.black_score - self.white_score
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1
        else:
            reward = 0

        return rep, reward, self.game_over

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
            self.integer_representation = [black_ints, white_ints]
        else:
            return [black_ints, white_ints]
        
    def update_matrix_from_ints(self, ints):
        """
        Updates the matrix representation of the board from the ints representation.
        """
        white_ints, black_ints = ints
        for i in range(self.size):
            for j in range(self.size):
                if white_ints[i] & (1 << j):
                    self.matrix_board[i, j] = -1
                elif black_ints[i] & (1 << j):
                    self.matrix_board[i, j] = 1
                else:
                    self.matrix_board[i, j] = 0
        self.integer_representation = ints

    def init_graph_board(self):
        """
        Initializes the graph representation of the board.
        """

        graph_board = nx.grid_2d_graph(self.size, self.size)
        for node in graph_board.nodes:
            graph_board.nodes[node]["color"] = 0
       
        
        for i in range(self.size):
            for j in range(self.size):
                graph_board.nodes[(i,j)]['color'] = self.matrix_board[i,j]

        self.graph_board = graph_board

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
        self.captures[color] += capture_num

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

        possible_board.nodes[(x, y)]["color"] = self.turn
        possible_matrix_board[y, x] = self.turn
        for stone in group:
            possible_board.nodes[stone]["color"] = 0
            possible_matrix_board[stone[1], stone[0]] = 0
            possible_ints = self.update_ints(board=possible_matrix_board, in_place=False)

        # check if ko rule is violated
        if possible_ints in self.history:
            return 0
        
        # if not, update board
        elif not check_only:
            self.graph_board = possible_board
            self.matrix_board = possible_matrix_board
        return len(group)

    def get_legal_moves(self, color):
        """
        Returns a list of legal moves for color.
        
        Inputs:
            - color: int, 1 for black, -1 for white
            
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
        # legal_moves.append("resign") # we may want to remove this option for bot self-play!!!
        return legal_moves
    
    def render(self):
        self.print_board()

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
        return black_area, white_area + self.komi
    
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
            # only time when both are false is starting position
            return (1, visited_empty) if black_flag else (-1, visited_empty) 
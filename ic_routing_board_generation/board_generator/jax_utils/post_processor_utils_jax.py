import random
from typing import List, Tuple
import numpy as np
from copy import deepcopy
from ic_routing_board_generation.board_generator.numpy_board_generation.bfs_board import BFSBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.board_generator_random_walk_rb import RandomWalkBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.board_generator_wfc_oj import WFCBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.lsystems_numpy import LSystemBoardGen
from ic_routing_board_generation.board_generator.numpy_data_model.board_generator_data_model import Position
from ic_routing_board_generation.board_generator.numpy_data_model.abstract_board import AbstractBoard


# EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3  # Ideally should be imported from Jumanji
from jumanji.environments.routing.connector.constants import EMPTY, PATH, POSITION, TARGET
STARTING_POSITION = POSITION  # Resolve ambiguity of POSITION constant


# Define exceptions for the board validity checks
class IncorrectBoardSizeError(Exception):
    """ Raised when a board size does not match the specified dimensions."""
    pass


class NumAgentsOutOfRangeError(Exception):
    """ Raised when self._wires_on_board is negative."""
    pass


class EncodingOutOfRangeError(Exception):
    """ Raised when one or more cells on the board have an invalid index."""
    pass


class DuplicateHeadsTailsError(Exception):
    """ Raised when one of the heads or tails of a wire is duplicated."""
    pass


class MissingHeadTailError(Exception):
    """ Raised when one of the heads or tails of a wire is missing."""
    pass


class InvalidWireStructureError(Exception):
    """ Raised when one or more of the wires has an invalid structure, e.g. looping or branching."""
    pass


class PathNotFoundError(Exception):
    """ Raised when a path cannot be found between a head and a target."""
    pass


def get_wires_on_board(board_layout) -> int:
    """ Returns the number of wires on the board by counting the number of unique wire encodings."""
    return len(np.unique(board_layout[board_layout % 3 == PATH]))


#def process_board(self) -> None:
#    """Processes the board by getting the heads, targets and paths."""
#
#    self.heads, self.targets = self.get_heads_and_targets()
#    # find the paths
#    self.paths = self.get_paths_from_heads_and_targets()
#
#def get_paths_from_heads_and_targets(self) -> List[List[Tuple[int, int]]]:
#    """Gets the paths from all heads to all targets via BFS using only valid moves and cells with wire encodings."""
#    paths = []
#    for i in range(len(self.heads)):
#        #paths.append(self.get_path_from_head_and_target(self.heads[i], self.targets[i]))
#    return paths
#
#def get_path_from_head_and_target(self, head, target) -> List[Tuple[int, int]]:
#    """Gets the path from a head to a target via BFS using only valid moves and cells with wire encodings.
#    Essentially remove_extraneous_path_cells"""
#    # path = [head]
#    # valid moves are up, down, left and right with the bounds of the board
#    valid_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#    # Shuffle valid moves to ensure that the search is not biased
#    random.shuffle(valid_moves)
#    # Get the head and target encodings
#    head_encoding = self.board_layout[head]
#    target_encoding = self.board_layout[target]
#    path_encoding = head_encoding - 1
#    # Only cells with value head_encoding, target_encoding or path_encoding are valid
#    valid_cells = [head_encoding, target_encoding, path_encoding, EMPTY]
#    # Initialize the queue
#    queue = [head]
#    # Initialize the explored array
#    explored = np.full(self.board_layout.shape, False)
#    # Initialize the parent array
#    parent = np.full(self.board_layout.shape, None)
#    # Initialize the path
#    path = []
#
#    while len(queue) > 0:
#        # Get the current cell
#        current_cell = queue.pop(0)
#        # Mark the current cell as explored
#        explored[current_cell] = True
#        # Check if the current cell is the target
#        if current_cell == target:
#            # Get the path from the target to the head
#            path = self.get_path_from_target_to_head(parent, target)
#            break
#        # Get the neighbours of the current cell
#        neighbours = self.get_neighbours(current_cell, valid_moves, valid_cells)
#        # Loop through the neighbours
#        for neighbour in neighbours:
#            # Check if the neighbour has been explored
#            if not explored[neighbour]:
#                # Add the neighbour to the queue
#                queue.append(neighbour)
#                # Mark the neighbour as explored
#                explored[neighbour] = True
#                # Set the parent of the neighbour
#                parent[neighbour] = current_cell
#
#    self.remove_extraneous_path_cells(path, path_encoding)
#
#    # Raise error if path not found
#    if len(path) == 0 or (head not in path) or (target not in path):
#        raise PathNotFoundError
#    return path


def remove_extraneous_path_cells(path: List[Tuple[int, int]], path_encoding: int, board_layout) -> None:
    """Removes extraneous path cells from the board layout."""
    # Change any cell with the same wire_encoding but not in the path to an empty cell
    path_set = set(path)
    for i in range(board_layout.shape[0]):
        for j in range(board_layout.shape[1]):
            if board_layout[i, j] == path_encoding and (i, j) not in path_set:
                board_layout[i, j] = EMPTY
            elif board_layout[i, j] == EMPTY and (i, j) in path_set:
                board_layout[i, j] = PATH
    return board_layout


@staticmethod
def get_path_from_target_to_head(parent, target) -> List[Tuple[int, int]]:
    """Gets the path from a target to a head."""
    # Initialize the path
    path = [target]
    # Get the parent of the target
    parent_cell = parent[target]
    # Loop until the parent cell is None
    while parent_cell is not None:
        # Add the parent cell to the path
        path.append(parent_cell)
        # Get the parent of the parent cell
        parent_cell = parent[parent_cell]
    # Reverse the path
    path.reverse()
    return path


# def get_neighbours(cell, valid_moves, valid_cells) -> List[Tuple[int, int]]:
#    """Gets the valid neighbours of a cell."""
#    # Initialize the list of neighbours
#    neighbours = []
#    # Loop through the valid moves
#    for move in valid_moves:
#        # Get the neighbour
#        neighbour = (cell[0] + move[0], cell[1] + move[1])
#        # Check if the neighbour is valid
#        if is_valid_cell(neighbour, valid_cells):
#            # Add the neighbour to the list of neighbours
#            neighbours.append(neighbour)
#    return neighbours
#
#
# def is_valid_cell(cell: Tuple[int, int], valid_cells: List[Tuple[int, int]]) -> bool:
#    """Checks if a cell is valid."""
#    # Check if the cell is within the bounds of the board
#    if cell[0] < 0 or cell[0] >= self.board_layout.shape[0] or cell[1] < 0 or cell[1] >= self.board_layout.shape[1]:
#        return False
#    # Check if the cell has a valid encoding
#    if self.board_layout[cell] not in valid_cells:
#        return False
#    return True


@staticmethod
def get_path_length(path: List[Tuple[int, int]]) -> int:
    """Gets the length of a path"""
    return len(path)

#def count_path_bends(self, path: List[Tuple[int, int]]) -> int:
#    """Counts the number of bends in a path"""
#    bends = 0
#    for i in range(1, len(path) - 1):
#        # Get the previous and next positions
#        prev_pos = path[i - 1]
#        next_pos = path[i + 1]
#        # Check if the current position is a bend
#        if self.is_bend(prev_pos, path[i], next_pos):
#            bends += 1
#    return bends


@staticmethod
def is_bend(prev_pos: Tuple[int, int], pos: Tuple[int, int], next_pos: Tuple[int, int]) -> bool:
    """Checks if a position is a bend"""
    prev_row, prev_col = prev_pos
    next_row, next_col = next_pos
    # Get the row and column of the current position
    row, col = pos
    # Check if the current position is a bend
    if (row == prev_row and row == next_row) or (col == prev_col and col == next_col):
        return False
    return True

#def proportion_filled(self) -> float:
#    """Returns the proportion of the board that is filled with wires"""
#    filled_positions = np.count_nonzero(self.board_layout)
#    # Get the total number of positions
#    total_positions = self.board_layout.shape[0] * self.board_layout.shape[1]
#    # Return the percentage of filled positions
#    return filled_positions / total_positions
#
#def distance_between_heads_and_targets(self) -> List[float]:
#    """Returns the L2 distance between the heads and targets of the wires"""
#    distances = []
#    for head, target in zip(self.heads, self.targets):
#        distances.append(self.get_distance_between_cells(head, target))
#    return distances


@staticmethod
def get_distance_between_cells(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> float:
    """Returns the L2 distance between two cells"""
    return ((cell1[0] - cell2[0]) ** 2 + (cell1[1] - cell2[1]) ** 2) ** 0.5

#def remove_wire(self, wire_index: int) -> None:
#    """Removes a wire from the board"""
#    if wire_index >= len(self.heads):
#        raise ValueError(f"Wire index out of range. Only {len(self.heads)} wires on the board.")
#    else:
#        # Get the head, target and path of the wire
#        head, target, path = self.heads[wire_index], self.targets[wire_index], self.paths[wire_index]
#        # Remove the wire from the board
#        for pos in path:
#            self.board_layout[pos[0]][pos[1]] = 0
#        # Remove the wire from the list of heads, targets and paths
#        self.heads.pop(wire_index)
#        self.targets.pop(wire_index)
#        self.paths.pop(wire_index)
#
#        assert len(self.heads) == len(self.targets) == len(
#            self.paths), "Heads, targets and paths not of equal length"
#        assert head not in self.heads, "Head not removed"
#        assert target not in self.targets, "target not removed"
#        assert path not in self.paths, "Path not removed"
#
#def get_board_layout(self) -> np.ndarray:
#    # Returns the board layout
#    return self.board_layout
#
#def get_board_statistics(self) -> Dict[str, Union[int, float]]:
#    """Returns a dictionary of statistics about the board"""
#    num_wires = len(self.heads)
#    wire_lengths = [self.get_path_length(path) for path in self.paths]
#    avg_wire_length = sum(wire_lengths) / num_wires
#    wire_bends = [self.count_path_bends(path) for path in self.paths]
#    avg_wire_bends = sum(wire_bends) / num_wires
#    avg_head_target_distance = sum(self.distance_between_heads_and_targets()) / num_wires
#    proportion_filled = self.proportion_filled()
#
#    # Return summary dict
#    summary_dict = dict(num_wires=num_wires, wire_lengths=wire_lengths, avg_wire_length=avg_wire_length,
#                        wire_bends=wire_bends, avg_wire_bends=avg_wire_bends,
#                        avg_head_target_distance=avg_head_target_distance, percent_filled=proportion_filled)
#
#    return summary_dict


def is_valid_board(board) -> bool:
    """ Return a boolean indicating if the board is valid.  Raise an exception if not. """
    is_valid = True
    if not verify_board_size(board):
        raise IncorrectBoardSizeError
    if board.wires_on_board < 0:
        raise NumAgentsOutOfRangeError
    if not verify_encodings_range(board):
        raise EncodingOutOfRangeError
    if not verify_number_heads_tails(board):
        pass
    if not verify_wire_validity(board):
        raise InvalidWireStructureError
    return is_valid


def verify_board_size(board) -> bool:
    # Verify that the size of a board layout matches the specified dimensions.
    return np.shape(board.board_layout) == (board.rows, board.cols)


def verify_encodings_range(board) -> bool:
    # Verify that all the encodings on the board within the range of 0 to 3 * board._wires_on_board.
    wires_only = np.setdiff1d(board.board_layout, np.ndarray([EMPTY]))
    print(f'Max: {np.max(wires_only)}, Min: {np.min(wires_only)}')
    print(f'Wires on board: {board.wires_on_board}')
    print(f'Wires only: {wires_only}')
    if board.wires_on_board == 0:
        # if no wires, we should have nothing left of the board
        return len(wires_only) == 0
    if np.min(board.board_layout) < 0:
        return False
    if np.max(board.board_layout) > 3 * board.wires_on_board:
        print(np.max(board.board_layout))
        print(3 * board.board.wires_on_board)
        print('Epic Fail')
        return False
    return True


def verify_number_heads_tails(board_layout) -> bool:
    # Verify that each wire has exactly one head and one target.
    wires_only = np.setdiff1d(board_layout, np.ndarray([EMPTY]))
    is_valid = True
    for num_wire in range(board_layout.wires_on_board):
        heads = np.count_nonzero(wires_only == (num_wire * 3 + POSITION))
        tails = np.count_nonzero(wires_only == (num_wire * 3 + TARGET))
        if heads < 1 or tails < 1:
            is_valid = False
            raise MissingHeadTailError
        if heads > 1 or tails > 1:
            is_valid = False
            raise DuplicateHeadsTailsError
    return is_valid


def verify_wire_validity(board_layout: np.ndarray) -> bool:
    """ Verify that each wire has a valid shape,
        ie, each head/target is connected to one wire cell, and each wire cell is connected to two.

        Args:
            board_layout (np.ndarray): 2D array with wires encoded

        Returns:
            (bool): True if the wires are continuous and don't double-back, False otherwise.
    """
    rows, cols = board_layout.shape[0], board_layout.shape[1]
    for row in range(rows):
        for col in range(cols):
            cell_label = board_layout[row, col]
            # Don't check empty cells
            if cell_label > 0:
                # Check whether the cell is a wiring path or a starting/target cell
                if position_to_cell_type(Position(row, col), board_layout) == PATH:
                    # Wiring path cells should have two neighbors of the same wire
                    if num_wire_neighbors(cell_label, row, col, board_layout) != 2:
                        print(
                            f"({row},{col}) == {cell_label}, {num_wire_neighbors(cell_label, row, col, board_layout)} neighbors")
                        return False
                else:
                    # Head and target cells should only have one neighbor of the same wire.
                    if num_wire_neighbors(cell_label, row, col, board_layout) != 1:
                        print(
                            f"HT({row},{col}) == {cell_label}, {num_wire_neighbors(cell_label, row, col, board_layout)} neighbors")
                        return False
    return True


def num_wire_neighbors(cell_label: int, row: int, col: int, board_layout: np.ndarray) -> int:
    """ Return the number of adjacent cells belonging to the same wire.

        Args:
            cell_label (int) : value of the cell to investigate
            row (int)
            col (int) : (row,col) = 2D position of the cell to investigate
            board_layout (np.ndarray): 2D array with wires encoded

        Returns:
            (int) : The number of adjacent cells belonging to the same wire.
    """
    rows, cols = board_layout.shape
    neighbors = 0
    wire_num = cell_label_to_wire_num(cell_label)
    min_val = 3 * wire_num + PATH  # PATH=1 is lowest val
    max_val = 3 * wire_num + TARGET  # TARGET=3 is highest val
    if row > 0:
        if min_val <= board_layout[row - 1, col] <= max_val:  # same wire above
            neighbors += 1
    if col > 0:
        if min_val <= board_layout[row, col - 1] <= max_val:  # same wire to the left
            neighbors += 1
    if row < rows - 1:
        if min_val <= board_layout[row + 1, col] <= max_val:  # same wire below
            neighbors += 1
    if col < cols - 1:
        if min_val <= board_layout[row, col + 1] <= max_val:  # same wire to the right
            neighbors += 1
    return neighbors


def swap_heads_targets(self):
    """ Randomly swap the head and target of each wire.  Self.board_layout in modified in-place
    """
    # Loop through all the paths on the board
    # Randomly swap the head and target of each wire (and reverse the direction of the wire)
    for path in self.paths:
        p = np.random.rand()
        if p < 0.5:
            # Swap the head and target of the wire
            head_encoding = self.board_layout[path[0][0], path[0][1]]
            target_encoding = self.board_layout[path[-1][0], path[-1][1]]
            self.board_layout[path[0][0], path[0][1]] = target_encoding
            self.board_layout[path[-1][0], path[-1][1]] = head_encoding
            # Reverse the direction of the wire
            path.reverse()


def shuffle_wire_encodings(self):
    """ Randomly shuffle the encodings of all wires.  Self.board_layout in modified in-place
    """
    # shuffle the indices of the wires and then assign them to the wires in order
    new_indices = list(range(len(self.paths)))
    random.shuffle(new_indices)
    heads = []
    targets = []
    paths = []
    for index, num in enumerate(new_indices):
        heads.append(self.heads[num])
        targets.append(self.targets[num])
        paths.append(self.paths[num])
        # update the encodings of the wires
        for i, pos in enumerate(paths[index]):
            if i == 0:
                self.board_layout[pos[0], pos[1]] = 3 * index + POSITION
            elif i == len(paths[index]) - 1:
                self.board_layout[pos[0], pos[1]] = 3 * index + TARGET
            else:
                self.board_layout[pos[0], pos[1]] = 3 * index + PATH


def position_to_wire_num(position: Position, board_layout: np.ndarray) -> int:
    """ Returns the wire number of the given cell position

        Args:
            position (Position): 2D tuple of the [row, col] position
            board_layout (np.ndarray): 2D layout of the board with wires encoded

        Returns:
            (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
    """
    row, col = position.x, position.y
    rows, cols = board_layout.shape[0], board_layout.shape[1]
    if (0 <= row < rows) and (0 <= col < cols):
        cell_label = board_layout[row, col]
        return cell_label_to_wire_num(cell_label)
    else:
        return -1


@staticmethod
def cell_label_to_wire_num(cell_label: int) -> int:
    """ Returns the wire number of the given cell value

        Args:
            cell_label (int) : the value of the cell in self.layout

        Returns:
            (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
    """
    if cell_label == 0:
        return -1
    else:
        return (cell_label - 1) // 3


def position_to_cell_type(position: Position, board_layout: np.ndarray) -> int:
    """
    Return the type of cell at position (row, col) in board_layout
        0 = empty
        1 = path
        2 = position (starting position)
        3 = target

        Args:
            position (Position) : The 2D position of the cell
            board_layout (np.ndarray): 2D layout of the encoded board

        Returns:
            (int) : The type of cell (0-3) as detailed above.
    """
    cell = board_layout[position.x, position.y]
    if cell == 0:
        return cell
    else:
        return ((cell - 1) % 3) + 1


# def is_adjacent(self, cell_a: Tuple, cell_b: Tuple) -> bool:
#     """ Return TRUE if the two cells are adjacent, FALSE otherwise
#
#        Args:
#            cell_a (Position) : X,Y position of a cell
#            cell_b (Position) : X,Y position of another cell
#
#        Returns:
#            bool : True if the cells are adjacent
#     """
#     manhattan_distance = abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])
#     return manhattan_distance == 1


def extend_wires(board_layout: np.ndarray, random_dir: bool = False) -> np.ndarray:
    """ Extend the heads and targets of each wire as far as they can go, preference given to current direction.
        The implementation is done in-place on self.board_layout

        Args:
            board_layout (np.ndarray): 2D layout of the encoded board (before wire extension)
            random_dir (bool): False (default) continues the previous direction as much as possible,
                                True would let the wires extend in a random walk fashion.

        Returns:
            (np.ndarray): 2D layout of the encoded board (after wire extension)
    """
    rows = len(board_layout)
    cols = len(board_layout[0])
    prev_layout = False  # Initially set prev_layout != board_layout
    # Continue as long as the algorithm is still changing the board
    while not np.all(prev_layout == board_layout):
        prev_layout = 1 * board_layout
        for row in range(rows):
            for col in range(cols):
                # If the cell is not a head or target, ignore it.
                cell_type = position_to_cell_type(Position(row, col), board_layout)
                if (cell_type != STARTING_POSITION) and (cell_type != TARGET):
                    continue

                # If we have found a head or target, try to extend it.
                #
                # Get the list of neighbors available to extend to.
                current_pos = Position(row, col)
                poss_extension_list = get_open_adjacent_cells(current_pos, board_layout)
                # Convert tuples to Position class
                poss_extension_list = [Position(cell[0], cell[1]) for cell in poss_extension_list]
                # For each possible cell, throw it out if it already touches part of the same wire.
                current_wire_num = position_to_wire_num(current_pos, board_layout)
                for cell in deepcopy(poss_extension_list):  # Use a copy so we can modify the list in the loop
                    if num_wire_adjacencies(cell, current_wire_num, board_layout) > 1:
                        poss_extension_list.remove(cell)
                # If there is no room to extend, move on.
                if len(poss_extension_list) == 0:
                    continue
                # First find the neighboring cell that is part of the same wire, prioritize extending away from it.
                neighbors_list = get_neighbors_same_wire(Position(row, col), board_layout)

                if len(neighbors_list) == 0:
                    print(board_layout)
                    print(Position(row,col))
                    print(current_wire_num)

                # There should only be one neighbour to choose from for a head or starting_position cell
                neighbor = neighbors_list[0]
                # Try to extend away from previous neighbor
                priority_neighbor = Position(row + (row - neighbor.x), col + (col - neighbor.y))

                # Prioritize extending away from the previous neighbor if possible.
                if (priority_neighbor in poss_extension_list) and not random_dir:
                    extension_pos = priority_neighbor
                else:
                    extension_pos = random.choice(poss_extension_list)
                # Extend head/target into new cell
                board_layout[extension_pos.x, extension_pos.y] = board_layout[current_pos.x, current_pos.y]
                cell_type = position_to_cell_type(current_pos, board_layout)
                # Convert old head/target cell to a wire
                board_layout[current_pos.x, current_pos.y] += PATH - cell_type
    return board_layout


# This method is used by the extend_wires method
def get_neighbors_same_wire(input_pos: Position, board_layout: np.ndarray) -> List:
    """ Returns a list of adjacent cells belonging to the same wire.

        Args:
            input_pos (Position): 2D position in self.layout
            board_layout (np.ndarray): 2D layout of board with wires encoded

        Returns:
            (List) : a list of cells (2D positions) adjacent to the queried cell which belong to the same wire
    """
    output_list = []
    wire_num = position_to_wire_num(input_pos, board_layout)
    pos_up = Position(input_pos.x - 1, input_pos.y)
    pos_down = Position(input_pos.x + 1, input_pos.y)
    pos_left = Position(input_pos.x, input_pos.y - 1)
    pos_right = Position(input_pos.x, input_pos.y + 1)
    if position_to_wire_num(pos_up, board_layout) == wire_num:
        output_list.append(pos_up)
    if position_to_wire_num(pos_down, board_layout) == wire_num:
        output_list.append(pos_down)
    if position_to_wire_num(pos_left, board_layout) == wire_num:
        output_list.append(pos_left)
    if position_to_wire_num(pos_right, board_layout) == wire_num:
        output_list.append(pos_right)
    return output_list


# This method is used by the extend_wires method
def num_wire_adjacencies(cell: Position, wire_num: int, board_layout: np.ndarray) -> int:
    """ Returns the number of cells adjacent to the input which belong to the wire specified by wire_num.

        Args:
            cell (tuple): 2D position in self.board_layout
            wire_num (int): Count adjacent contacts with this specified wire.
            board_layout (np.ndarray): 2D array of board with wires encoded

        Returns:
            (int) : The number of adjacent cells belonging to the specified wire
    """
    num_adjacencies = 0
    if position_to_wire_num(Position(cell.x - 1, cell.y), board_layout) == wire_num:
        num_adjacencies += 1
    if position_to_wire_num(Position(cell.x + 1, cell.y), board_layout) == wire_num:
        num_adjacencies += 1
    if position_to_wire_num(Position(cell.x, cell.y - 1), board_layout) == wire_num:
        num_adjacencies += 1
    if position_to_wire_num(Position(cell.x, cell.y + 1), board_layout) == wire_num:
        num_adjacencies += 1
    return num_adjacencies


# This method is used by the extend_wires method
def get_open_adjacent_cells(input_pos: Position, board_layout: np.ndarray) -> List:
    """ Returns a list of open cells adjacent to the input cell.

        Args:
            input_pos (Position): The input cell to search adjacent to.
            board_layout (np.ndarray): 2D layout of the board with wires encoded

        Returns:
            List: List of 2D integer tuples, up to four available cells adjacent to the input cell.
    """
    rows, cols = board_layout.shape[0], board_layout.shape[1]
    adjacent_list = []
    # Check above, below, to the left and the right and add those cells to the list if available.
    # Check left
    new_pos_x, new_pos_y = input_pos.x - 1, input_pos.y
    if (new_pos_x >= 0) and (board_layout[new_pos_x, new_pos_y] == EMPTY):
        adjacent_list.append((new_pos_x, new_pos_y))
    # Check up
    new_pos_x, new_pos_y = input_pos.x, input_pos.y - 1
    if (new_pos_y >= 0) and (board_layout[new_pos_x, new_pos_y] == EMPTY):
        adjacent_list.append((new_pos_x, new_pos_y))
    # Check right
    new_pos_x, new_pos_y = input_pos.x + 1, input_pos.y
    if (new_pos_x < rows) and (board_layout[new_pos_x, new_pos_y] == EMPTY):
        adjacent_list.append((new_pos_x, new_pos_y))
    # Check down
    new_pos_x, new_pos_y = input_pos.x, input_pos.y + 1
    if (new_pos_y < cols) and (board_layout[new_pos_x, new_pos_y] == EMPTY):
        adjacent_list.append((new_pos_x, new_pos_y))
    return adjacent_list


## This method is NO LONGER used by the get_open_adjacent_cells method
#    def is_valid_extension(input: Position, wire_list: List, board_layout: np.ndarray) -> bool:
#    """ Returns a boolean, true if the cell is valid to add to the wire.
#
#         Args:
#            input (Position): The input cell to investigate.
#            wire_list (List): List of cells already in the wire.
#            board_layout (np.ndarray): 2D layout of the board with wires encoded
#
#        Returns:
#            bool: False if the cell is already in use,
#                  False if the cell connects the wire in a loop.
#                  True, otherwise.
#    return (board_layout[input.x, input.y] == EMPTY) and (input.x, input.y) not in wire_list \
#        and (number_of_adjacent_wires(input, wire_list) < 2)
#
#
## This method is used by a method used by the extend_wires method
#def number_of_adjacent_wires(input: Position, wire_list: List) -> int:
#    """ Returns the number of cells adjacent to the input cell which are in the wire_list.
#
#    Args:
#        input (Position): The input cell to search adjacent to.
#        wire_list (List): List of cells already in the wire.
#
#    Returns:
#        int: Number of adjacent cells that are in the wire_list.
#    """
#    num_adjacent = 0
#    # Check above, below, to the left and the right and count the number in the wire_list.
#    if (input.x - 1, input.y) in wire_list:
#        num_adjacent += 1
#    if (input.x + 1, input.y) in wire_list:
#        num_adjacent += 1
#    if (input.x, input.y - 1) in wire_list:
#        num_adjacent += 1
#    if (input.x, input.y + 1) in wire_list:
#        num_adjacent += 1
#    return num_adjacent


def training_board_from_solved_board(board_layout: np.ndarray) -> np.ndarray:
    """ Zero out and PATH cells to convert from a solved board to a training board.

        Args:
            board_layout (np.ndarray): 2D layout of the encoded solved board

        Returns:
            (nd.nparray): 2D layout of the encoded training board
    """
    rows, cols = board_layout.shape
    for row in range(rows):
        for col in range(cols):
            old_encoding = board_layout[row, col]
            if (old_encoding % 3) == PATH:
                board_layout[row, col] = 0
    return board_layout


def count_detours(board_layout: np.ndarray, count_current_wire: bool = False) -> int:
    """ Returns the number of wires that have to detour around a head or target cell.

        Args:
            board_layout (np.ndarray): 2D layout of the board with wires encoded
            count_current_wire (bool): Should we count wires that wrap around their own heads/targets?
                                            (default = False)

        Returns:
            (int) : The number of wires that have to detour around a head or target cell.
    """
    rows, cols = board_layout.shape
    num_detours = 0
    for x in range(rows):
        for y in range(cols):
            cell_type = position_to_cell_type(Position(x, y), board_layout)
            if (cell_type != STARTING_POSITION) and (cell_type != TARGET):
                continue
            current_wire = position_to_wire_num(Position(x, y), board_layout)
            #
            above = board_layout[:x, y]
            above = [cell_label_to_wire_num(cell_label) for cell_label in above if cell_label != 0]
            if not count_current_wire:
                above = [wire_num for wire_num in above if wire_num != current_wire]
            below = board_layout[x + 1:, y]
            below = [cell_label_to_wire_num(cell_label) for cell_label in below if cell_label != 0]
            if not count_current_wire:
                below = [wire_num for wire_num in below if wire_num != current_wire]
            common = (set(above) & set(below))
            num_detours += len(common)
            #
            left = board_layout[x, :y].tolist()
            left = [cell_label_to_wire_num(cell_label) for cell_label in left if cell_label != 0]
            if not count_current_wire:
                left = [wire_num for wire_num in left if wire_num != current_wire]
            right = board_layout[x, y+1:].tolist()
            right = [cell_label_to_wire_num(cell) for cell in right if cell != 0]
            if not count_current_wire:
                right = [wire_num for wire_num in right if wire_num != current_wire]
            common = (set(right) & set(left))
            num_detours += len(common)
    return num_detours


####################################################################################################
# JAX UTILS ########################################################################################
###################################################################################################
from jax import Array
from jax.random import PRNGKey
import jax.numpy as jnp
import jax
INVALID = jnp.array((-999, -999))  # Encoding for an invalid 2D position for lists that must be of constant length
INVALID4 = jnp.array([(-999, -999), (-999, -999), (-999, -999), (-999, -999)]) # Encoding for list of four invalid positions


@jax.disable_jit()
#@jax.jit
def extend_wires_jax(board_layout: Array, key: PRNGKey, randomness: float = 0.0,
                     two_sided: bool = True, max_iterations: int = 1e23) -> Array:
    """ Extend the heads and targets of each wire as far as they can go, preference given to current direction.
        The implementation is done in-place on self.board_layout

        Args:
            board_layout (Array): 2D layout of the encoded board (before wire extension)
            key( PRNGKey): a random key to use for randomly selecting directions (as needed)
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.

        Returns:
            (Array): 2D layout of the encoded board (after wire extension)
    """
    rows, cols = board_layout.shape
    prev_layout = board_layout.at[0, 0].add(1)  # Make prev_layout != board_layout
    # Continue as long as the algorithm is still changing the board
    """ Implement this as jax code below:
    while ~jnp.equal(prev_layout, board_layout):
    """
    def while_cond(carry):
        prev_layout, board_layout, _ = carry
        return ~jnp.array_equal(prev_layout, board_layout)

    def while_body(carry):
        prev_layout, board_layout, key = carry
        #print(board_layout)

        # Randomly flip/flop the board between iterations to remove any directional bias.
        key, flipkey, flopkey = jax.random.split(key, 3)
        do_flip = jax.random.choice(flipkey, jnp.array([True, False]))
        do_flop = jax.random.choice(flopkey, jnp.array([True, False]))
        board_layout = jax.lax.select(do_flip, board_layout[::-1, :], board_layout)
        board_layout = jax.lax.select(do_flop, board_layout[:, ::-1], board_layout)

        # Make a copy of the board_layout to check if it changes
        prev_layout = board_layout.at[0, 0].add(0)  # Like a deepcopy
        for row in range(rows):
            for col in range(cols):
                # Get the list of neighbors available to extend to.
                current_pos = jnp.array((row, col))
                poss_extension_list = get_open_adjacent_cells_jax(current_pos, board_layout)

                # For each possible cell, mark it INVALID if it already touches part of the same wire.
                current_wire_num = position_to_wire_num_jax(current_pos, board_layout)
                for (i, cell_pos) in enumerate(poss_extension_list):
                    num_adjacencies = num_wire_adjacencies_jax(cell_pos, current_wire_num, board_layout)
                    cell_or_invalid = jax.lax.select(num_adjacencies > 1, INVALID, cell_pos)
                    poss_extension_list = poss_extension_list.at[i].set(cell_or_invalid)

                # Get the position of the wire that led into this cell and prioritize continuing the same direction
                prev_neighbor = get_previous_neighbor_jax(current_pos, board_layout)
                priority_neighbor = (row + (row-prev_neighbor[0]), col + (col-prev_neighbor[1]))
                priority_neighbor = jnp.array(priority_neighbor)

                # If the current cell is not a head or target, it's not extendable,
                # so change everything in the list to INVALID
                cell_type = position_to_cell_type_jax(current_pos, board_layout)
                is_extendable = (two_sided & ((cell_type == STARTING_POSITION) | (cell_type == TARGET))) | \
                                (~two_sided & (cell_type == TARGET))
                poss_extension_list = jax.lax.select(is_extendable, poss_extension_list, INVALID4)

                # If there are no valid options to extend to, stop extending
                stop_extension = jnp.array_equal(poss_extension_list, INVALID4)
                poss_extension_list = jax.lax.select(stop_extension,
                                                     jnp.array([current_pos, current_pos, current_pos, current_pos]),
                                                     poss_extension_list)

                # Prioritize extending away from the previous neighbor if possible, or random choice if not.
                #key, choice_key = jax.random.split(key)

                # Pick a random selection from the possible extensions until we pick a valid one.
                def choice_body_func(carry):
                    poss_extension_list, extension_pos, key = carry
                    key, choice_key = jax.random.split(key)
                    extension_pos = jax.random.choice(choice_key, poss_extension_list)
                    #print("extension_pos = ", extension_pos)
                    carry = (poss_extension_list, extension_pos, key)
                    return carry

                def choice_cond_func(carry):
                    _, extension_pos, _ = carry
                    return jnp.array_equal(extension_pos, INVALID)

                # Is the prioity neighbor available?
                extension_pos = INVALID
                _, extension_pos, _ = jax.lax.while_loop(choice_cond_func, choice_body_func,
                                                         (poss_extension_list, extension_pos, key))
                priority_neighbor_available = False
                for item in poss_extension_list:
                    item_match = jnp.array_equal(item, priority_neighbor)
                    priority_neighbor_available = jax.lax.select(item_match, True, priority_neighbor_available)
                # Use random choice if we are in random mode.
                # Otherwise rioritize extending away from the previous neighbor if possible, random if not available.
                key, random_key = jax.random.split(key)
                use_random = (randomness > jax.random.uniform(random_key))
                extension_pos = jax.lax.select(priority_neighbor_available & ~use_random,
                                               priority_neighbor, extension_pos)

                # Extend head/target into new cell
                # print(f"Set {extension_pos[0]},{extension_pos[1]} to {board_layout[current_pos[0], current_pos[1]]}")
                board_layout = board_layout.at[extension_pos[0], extension_pos[1]]\
                                            .set(board_layout[current_pos[0], current_pos[1]])
                cell_type = position_to_cell_type_jax(current_pos, board_layout)
                # Convert old head/target cell to a wire (unless stop_extension is True)
                cell_type_offset = jax.lax.select(stop_extension, 0, PATH - cell_type)
                # print(f"Add {cell_type_offset} to {current_pos[0]}, {current_pos[1]}")
                board_layout = board_layout.at[current_pos[0], current_pos[1]].add(cell_type_offset)
                # print(board_layout)

        # Undo random flip/flopping to compare to prev_layout
        board_layout = jax.lax.select(do_flip, board_layout[::-1, :], board_layout)
        board_layout = jax.lax.select(do_flop, board_layout[:, ::-1], board_layout)
        # For jax.lax.while_loop
        carry = (prev_layout, board_layout, key)
        return carry

    # For jax.lax.while_loop
    _, board_layout, _ = jax.lax.while_loop(while_cond, while_body, (prev_layout, board_layout, key))
    return board_layout


#@jax.jit
# This method is used by the extend_wires_jax method
def get_previous_neighbor_jax(input_pos: jnp.array, board_layout: Array) -> jnp.array:
    """ Returns the position of an adjacent cell of the same wire as the current cell.

        Note that if the input cell is a head or target, then there will only be one adjacent cell in the wire.
        If the current cell is a PATH, it will return one of the two adjacent cells in the wire.
        If the current cell is EMPTY, then it will return something INVALID

        Args:
            input_pos (jnp.array): 2D position of cell in board_layout
            board_layout (Array): 2D layout of board with wires encoded

        Returns:
            (jnp.array) : 2D position of a cell of the same wire.
    """
    row, col = input_pos[0], input_pos[1]
    input_wire = position_to_wire_num_jax(input_pos, board_layout)
    output_pos = INVALID
    neighbors_positions = jnp.array([(row-1, col), (row+1, col), (row, col-1), (row, col+1)])
    for neighbor_position in neighbors_positions:
        neighbor_position = jnp.array(neighbor_position)
        neighbor_wire = position_to_wire_num_jax(neighbor_position, board_layout)
        is_prev_neighbor = (neighbor_wire == input_wire) & \
                           (neighbor_position[0] >= 0) & (neighbor_position[0] < board_layout.shape[0]) & \
                           (neighbor_position[1] >= 0) & (neighbor_position[1] < board_layout.shape[1])
        output_pos = jax.lax.select(is_prev_neighbor, neighbor_position, output_pos)
    return output_pos


#@jax.jit
# This method is used by the extend_wires_jax method
def get_neighbors_same_wire_jax(input_pos: jnp.array, board_layout: Array) -> jnp.array:
    """ Returns a list of adjacent cells belonging to the same wire.

        Args:
            input_pos (jnp.array): 2D position of cell in board_layout
            board_layout (Array): 2D layout of board with wires encoded

        Returns:
            (jnp.array) : a list of cells (2D positions) adjacent to the queried cell which belong to the same wire
                            The output will always have four entries but some will be INVALID.
    """
    output_list = []
    wire_num = position_to_wire_num_jax(input_pos, board_layout)
    row, col = input_pos[0], input_pos[1]
    pos_up = jnp.array((row - 1, col))
    pos_down = jnp.array((row + 1, col))
    pos_left = jnp.array((row, col - 1))
    pos_right = jnp.array((row, col + 1))
    # Check each of the four directions
    adjacent_cell = jax.lax.select(position_to_wire_num_jax(pos_up, board_layout) == wire_num,
                                   pos_up, INVALID)
    output_list.append(adjacent_cell)
    adjacent_cell = jax.lax.select(position_to_wire_num_jax(pos_down, board_layout) == wire_num,
                                   pos_down, INVALID)
    output_list.append(adjacent_cell)
    adjacent_cell = jax.lax.select(position_to_wire_num_jax(pos_left, board_layout) == wire_num,
                                   pos_left, INVALID)
    output_list.append(adjacent_cell)
    adjacent_cell = jax.lax.select(position_to_wire_num_jax(pos_right, board_layout) == wire_num,
                                   pos_right, INVALID)
    output_list.append(adjacent_cell)
    #  There will always be four items in the list but some of them might be 'INVALID'
    output_list = jnp.array(output_list)
    return output_list


#@jax.jit
# This method is used by the extend_wires_jax method
def num_wire_adjacencies_jax(input_pos: jnp.array, wire_num: Array, board_layout: Array) -> Array:
    """ Returns the number of cells adjacent to cell which belong to the wire specified by wire_num.

        Args:
            input_pos (tuple): 2D position in board_layout
            wire_num (int): Count adjacent contacts with this specified wire.
            board_layout (Array): 2D array of board with wires encoded

        Returns:
            (Array) : The number of adjacent cells belonging to the specified wire
    """
    num_adjacencies = 0
    # Check above
    wire_adjacent = position_to_wire_num_jax(jnp.array((input_pos[0] - 1, input_pos[1])), board_layout)
    num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
    # Check below
    wire_adjacent = position_to_wire_num_jax(jnp.array((input_pos[0] + 1, input_pos[1])), board_layout)
    num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
    # Check left
    wire_adjacent = position_to_wire_num_jax(jnp.array((input_pos[0], input_pos[1] - 1)), board_layout)
    num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
    # Check right
    wire_adjacent = position_to_wire_num_jax(jnp.array((input_pos[0], input_pos[1] + 1)), board_layout)
    num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
    return num_adjacencies


#@jax.jit
# This method is used by the extend_wires_jax method
def get_open_adjacent_cells_jax(input_pos: jnp.array, board_layout: Array) -> jnp.array:
    """ Returns a list of open cells adjacent to the input cell.

        Args:
            input_pos (Array, Tuple[int, int]): The 2D position of the input cell to search adjacent to.
            board_layout (Array): 2D layout of the board with wires encoded

        Returns:
            (jnp.array): List of 2D integer tuples, up to four available cells adjacent to the input cell.
    """
    rows, cols = board_layout.shape
    adjacent_list = []
    # Check above, below, to the left and the right and add those cells to the list if available.
    # Check above
    adjacent_x, adjacent_y = input_pos[0] - 1, input_pos[1]
    adjacent_cell = jax.lax.select((adjacent_x >= 0) & (board_layout[adjacent_x, adjacent_y] == EMPTY),
                                   jnp.array((adjacent_x, adjacent_y)), INVALID)
    adjacent_list.append(adjacent_cell)
    # Check left
    adjacent_x, adjacent_y = input_pos[0], input_pos[1] - 1
    adjacent_cell = jax.lax.select((adjacent_y >= 0) & (board_layout[adjacent_x, adjacent_y] == EMPTY),
                                   jnp.array((adjacent_x, adjacent_y)), INVALID)
    adjacent_list.append(adjacent_cell)
    # Check below
    adjacent_x, adjacent_y = input_pos[0] + 1, input_pos[1]
    adjacent_cell = jax.lax.select((adjacent_x < rows) & (board_layout[adjacent_x, adjacent_y] == EMPTY),
                                   jnp.array((adjacent_x, adjacent_y)), INVALID)
    adjacent_list.append(adjacent_cell)
    # Check right
    adjacent_x, adjacent_y = input_pos[0], input_pos[1] + 1
    adjacent_cell = jax.lax.select((adjacent_y < cols) & (board_layout[adjacent_x, adjacent_y] == EMPTY),
                                   jnp.array((adjacent_x, adjacent_y)), INVALID)
    adjacent_list.append(adjacent_cell)
    #  There will always be four items in the list but some of them might be 'INVALID'
    adjacent_list = jnp.array(adjacent_list)
    return adjacent_list


#@jax.jit
def position_to_wire_num_jax(position: jnp.array, board_layout: Array) -> Array:
    """ Returns the wire number of the given cell position

        Args:
            position (jnp.array[int, int]): 2D tuple of the [row, col] position
            board_layout (Array): 2D layout of the board with wires encoded

        Returns:
            (int) : The wire number that the cell belongs to.
                    Returns -1 if not part of a wire or out-of-bounds.
    """
    row, col = position[0], position[1]
    rows, cols = board_layout.shape
    cell_encoding = board_layout[row, col]
    wire_num = jax.lax.select((0 <= row) & (row < rows) & (0 <= col) & (col < cols),
                              cell_encoding_to_wire_num_jax(cell_encoding), -1)
    return wire_num


#@jax.jit
def cell_encoding_to_wire_num_jax(cell_encoding: Array) -> Array:
    """ Returns the wire number of the given cell value

        Args:
            cell_encoding (int) : the value of the cell in self.layout

        Returns:
            (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
    """
    output = jax.lax.select(cell_encoding == 0, -1, int((cell_encoding - 1) // 3))
    return output




#@jax.jit
def position_to_cell_type_jax(position: jnp.array, board_layout: Array) -> Array:
    """ Return the type of cell at position (row, col) in board_layout
        0 = empty
        1 = path
        2 = position (starting position)
        3 = target

        Args:
            position (jnp.array) : The 2D position of the cell
            board_layout (Array): 2D layout of the encoded board

        Returns:
            (int) : The type of cell (0-3) as detailed above.
    """
    cell_encoding = board_layout[position[0], position[1]]
    cell_type = jax.lax.select(cell_encoding == 0, cell_encoding, ((cell_encoding - 1) % 3) + 1)
    return cell_type


#@jax.jit
def training_board_from_solved_board_jax(board_layout: Array) -> Array:
    """ Zero out and PATH cells to convert from a solved board to a training board.

        Args:
            board_layout (Array): 2D layout of the encoded solved board

        Returns:
            (Array): 2D layout of the encoded training board
    """
    rows, cols = board_layout.shape
    for row in range(rows):
        for col in range(cols):
            old_encoding = board_layout[row, col]
            new_encoding = jax.lax.select((old_encoding % 3) == PATH, 0, old_encoding)
            board_layout = board_layout.at[row, col].set(new_encoding)
    return board_layout


#TODO JAXIFY
def count_detours_jax(board_layout: Array, count_current_wire: bool = False) -> jnp.array:
    """ Returns the number of wires that have to detour around a head or target cell.

        Args:
            board_layout (Array): 2D layout of the board with wires encoded
            count_current_wire (bool): Should we count wires that wrap around their own heads/targets?
                                            (default = False)

        Returns:
            (jnp.array(int)) : The number of wires that have to detour around a head or target cell.
    """
    rows, cols = board_layout.shape
    num_detours = 0
    for x in range(rows):
        for y in range(cols):
            cell_type = position_to_cell_type(Position(x, y), board_layout)
            if (cell_type != STARTING_POSITION) and (cell_type != TARGET):
                continue
            current_wire = position_to_wire_num(Position(x, y), board_layout)
            #
            above = board_layout[:x, y]
            above = [cell_label_to_wire_num(cell_label) for cell_label in above if cell_label != 0]
            if not count_current_wire:
                above = [wire_num for wire_num in above if wire_num != current_wire]
            below = board_layout[x + 1:, y]
            below = [cell_label_to_wire_num(cell_label) for cell_label in below if cell_label != 0]
            if not count_current_wire:
                below = [wire_num for wire_num in below if wire_num != current_wire]
            common = (set(above) & set(below))
            num_detours += len(common)
            #
            left = board_layout[x, :y].tolist()
            left = [cell_label_to_wire_num(cell_label) for cell_label in left if cell_label != 0]
            if not count_current_wire:
                left = [wire_num for wire_num in left if wire_num != current_wire]
            right = board_layout[x, y+1:].tolist()
            right = [cell_label_to_wire_num(cell) for cell in right if cell != 0]
            if not count_current_wire:
                right = [wire_num for wire_num in right if wire_num != current_wire]
            common = (set(right) & set(left))
            num_detours += len(common)
    return num_detours


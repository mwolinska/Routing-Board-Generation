import random
from typing import List
import numpy as np
from copy import deepcopy
from routing_board_generation.board_generation_methods.numpy_implementation.data_model.board_generator_data_model import (
    Position,
)

# EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3  # Ideally should be imported from Jumanji
from jumanji.environments.routing.connector.constants import (
    EMPTY,
    PATH,
    POSITION,
    TARGET,
)

from routing_board_generation.board_generation_methods.numpy_implementation.utils.exceptions import (
    IncorrectBoardSizeError,
    NumAgentsOutOfRangeError,
    EncodingOutOfRangeError,
    InvalidWireStructureError,
    MissingHeadTailError,
    DuplicateHeadsTailsError,
)

STARTING_POSITION = POSITION  # Resolve ambiguity of POSITION constant


def get_wires_on_board(board_layout) -> int:
    """Returns the number of wires on the board by counting the number of unique wire encodings."""
    return len(np.unique(board_layout[board_layout % 3 == PATH]))


def is_valid_board(board) -> bool:
    """Return a boolean indicating if the board is valid.  Raise an exception if not."""
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
    print(f"Max: {np.max(wires_only)}, Min: {np.min(wires_only)}")
    print(f"Wires on board: {board.wires_on_board}")
    print(f"Wires only: {wires_only}")
    if board.wires_on_board == 0:
        # if no wires, we should have nothing left of the board
        return len(wires_only) == 0
    if np.min(board.board_layout) < 0:
        return False
    if np.max(board.board_layout) > 3 * board.wires_on_board:
        print(np.max(board.board_layout))
        print(3 * board.board.wires_on_board)
        print("Epic Fail")
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
    """Verify that each wire has a valid shape,
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
                            f"({row},{col}) == {cell_label}, {num_wire_neighbors(cell_label, row, col, board_layout)} neighbors"
                        )
                        return False
                else:
                    # Head and target cells should only have one neighbor of the same wire.
                    if num_wire_neighbors(cell_label, row, col, board_layout) != 1:
                        print(
                            f"HT({row},{col}) == {cell_label}, {num_wire_neighbors(cell_label, row, col, board_layout)} neighbors"
                        )
                        return False
    return True


def num_wire_neighbors(
    cell_label: int, row: int, col: int, board_layout: np.ndarray
) -> int:
    """Return the number of adjacent cells belonging to the same wire.

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
    """Randomly swap the head and target of each wire.  Self.board_layout in modified in-place"""
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
    """Randomly shuffle the encodings of all wires.  Self.board_layout in modified in-place."""
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
    """Returns the wire number of the given cell position

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


def cell_label_to_wire_num(cell_label: int) -> int:
    """Returns the wire number of the given cell value

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
    """Extend the heads and targets of each wire as far as they can go, preference given to current direction.
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
                poss_extension_list = [
                    Position(cell[0], cell[1]) for cell in poss_extension_list
                ]
                # For each possible cell, throw it out if it already touches part of the same wire.
                current_wire_num = position_to_wire_num(current_pos, board_layout)
                for cell in deepcopy(
                    poss_extension_list
                ):  # Use a copy so we can modify the list in the loop
                    if num_wire_adjacencies(cell, current_wire_num, board_layout) > 1:
                        poss_extension_list.remove(cell)
                # If there is no room to extend, move on.
                if len(poss_extension_list) == 0:
                    continue
                # First find the neighboring cell that is part of the same wire, prioritize extending away from it.
                neighbors_list = get_neighbors_same_wire(
                    Position(row, col), board_layout
                )

                if len(neighbors_list) == 0:
                    print(board_layout)
                    print(Position(row, col))
                    print(current_wire_num)

                # There should only be one neighbour to choose from for a head or starting_position cell
                neighbor = neighbors_list[0]
                # Try to extend away from previous neighbor
                priority_neighbor = Position(
                    row + (row - neighbor.x), col + (col - neighbor.y)
                )

                # Prioritize extending away from the previous neighbor if possible.
                if (priority_neighbor in poss_extension_list) and not random_dir:
                    extension_pos = priority_neighbor
                else:
                    extension_pos = random.choice(poss_extension_list)
                # Extend head/target into new cell
                board_layout[extension_pos.x, extension_pos.y] = board_layout[
                    current_pos.x, current_pos.y
                ]
                cell_type = position_to_cell_type(current_pos, board_layout)
                # Convert old head/target cell to a wire
                board_layout[current_pos.x, current_pos.y] += PATH - cell_type
    return board_layout


# This method is used by the extend_wires method
def get_neighbors_same_wire(input_pos: Position, board_layout: np.ndarray) -> List:
    """Returns a list of adjacent cells belonging to the same wire.

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
def num_wire_adjacencies(
    cell: Position, wire_num: int, board_layout: np.ndarray
) -> int:
    """Returns the number of cells adjacent to the input which belong to the wire specified by wire_num.

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
    """Returns a list of open cells adjacent to the input cell.

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


def training_board_from_solved_board(board_layout: np.ndarray) -> np.ndarray:
    """Zero out and PATH cells to convert from a solved board to a training board.

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
    """Returns the number of wires that have to detour around a head or target cell.

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
            above = [
                cell_label_to_wire_num(cell_label)
                for cell_label in above
                if cell_label != 0
            ]
            if not count_current_wire:
                above = [wire_num for wire_num in above if wire_num != current_wire]
            below = board_layout[x + 1 :, y]
            below = [
                cell_label_to_wire_num(cell_label)
                for cell_label in below
                if cell_label != 0
            ]
            if not count_current_wire:
                below = [wire_num for wire_num in below if wire_num != current_wire]
            common = set(above) & set(below)
            num_detours += len(common)
            #
            left = board_layout[x, :y].tolist()
            left = [
                cell_label_to_wire_num(cell_label)
                for cell_label in left
                if cell_label != 0
            ]
            if not count_current_wire:
                left = [wire_num for wire_num in left if wire_num != current_wire]
            right = board_layout[x, y + 1 :].tolist()
            right = [cell_label_to_wire_num(cell) for cell in right if cell != 0]
            if not count_current_wire:
                right = [wire_num for wire_num in right if wire_num != current_wire]
            common = set(right) & set(left)
            num_detours += len(common)
    return num_detours

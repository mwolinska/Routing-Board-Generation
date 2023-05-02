import random
import time
from typing import List, Union, Dict, Tuple, Optional

import numpy as np
from jumanji.environments.routing.connector.constants import (
    PATH,
    EMPTY,
    POSITION,
    TARGET,
)

from routing_board_generation.board_generation_methods.jax_implementation.board_generation.lsystems import (
    JAXLSystemBoard,
)
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.bfs_board import (
    BFSBoard,
)
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.numberlink import (
    NumberLinkBoard,
)
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.random_walk import (
    RandomWalkBoard,
)
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.wave_function_collapse import (
    WFCBoard,
)
from routing_board_generation.board_generation_methods.numpy_implementation.data_model.abstract_board import (
    AbstractBoard,
)
from routing_board_generation.board_generation_methods.numpy_implementation.utils.exceptions import (
    IncorrectBoardSizeError,
    NumAgentsOutOfRangeError,
    EncodingOutOfRangeError,
    DuplicateHeadsTailsError,
    MissingHeadTailError,
    InvalidWireStructureError,
    PathNotFoundError,
)


class BoardProcessor:
    def __init__(self, board: Union[np.ndarray, AbstractBoard]) -> None:
        """Constructor for the BoardProcessor class."""
        if isinstance(board, np.ndarray):
            self.board_layout = board
        else:
            self.board_layout = board.return_solved_board()

        self.board = board
        self.heads, self.targets, self.paths = None, None, None
        self.rows = self.board_layout.shape[0]
        self.cols = self.board_layout.shape[1]
        # if isinstance(board, AbstractBoard):
        #     # Check that board is valid!
        #     self.is_valid_board()
        self.process_board()

        # Given a board, we want to extract the positions of the heads, targets and paths
        # We also want to check that the board is valid, i.e. that it has the correct number of wires, heads and tails

    def get_heads_and_targets(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Returns the heads and targets of the board layout
        heads are encoded as 2,5,8,11,...
        targets are encoded as 3,6,9,12,...
        both are greater than 0
        """
        # Get the heads and targets in the right order
        heads = []
        targets = []
        board_layout = self.board_layout
        # Get the maximum value in the board layout
        max_val = np.max(board_layout)
        # Get the heads and targets
        for i in range(1, max_val + 1):
            # Get the head and target
            if i % 3 == POSITION:
                try:
                    head = np.argwhere(board_layout == i)[0]
                    heads.append(tuple(head))
                    target = np.argwhere(board_layout == i + 1)[0]
                    targets.append(tuple(target))
                except IndexError:
                    print(
                        f"IndexError: i = {i}, max_val = {max_val}, board_layout = {board_layout}"
                    )
        return heads, targets

    def get_wires_on_board(self) -> int:
        """Returns the number of wires on the board by counting the number of unique wire encodings."""
        return len(np.unique(self.board_layout[self.board_layout % 3 == PATH]))

    def process_board(self) -> None:
        """Processes the board by getting the heads, targets and paths."""

        self.heads, self.targets = self.get_heads_and_targets()
        # find the paths
        self.paths = self.get_paths_from_heads_and_targets()

    def get_paths_from_heads_and_targets(self) -> List[List[Tuple[int, int]]]:
        """Gets the paths from all heads to all targets via BFS using only valid moves and cells with wire encodings."""
        paths = []
        for i in range(len(self.heads)):
            paths.append(
                self.get_path_from_head_and_target(self.heads[i], self.targets[i])
            )
        return paths

    def get_path_from_head_and_target(self, head, target) -> List[Tuple[int, int]]:
        """Gets the path from a head to a target via BFS using only valid moves and cells with wire encodings.
        Essentially remove_extraneous_path_cells"""
        # path = [head]
        # valid moves are up, down, left and right with the bounds of the board
        valid_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # Shuffle valid moves to ensure that the search is not biased
        random.shuffle(valid_moves)
        # Get the head and target encodings
        head_encoding = self.board_layout[head]
        target_encoding = self.board_layout[target]
        path_encoding = head_encoding - 1
        # Only cells with value head_encoding, target_encoding or path_encoding are valid
        valid_cells = [head_encoding, target_encoding, path_encoding, EMPTY]
        # Initialize the queue
        queue = [head]
        # Initialize the explored array
        explored = np.full(self.board_layout.shape, False)
        # Initialize the parent array
        parent = np.full(self.board_layout.shape, None)
        # Initialize the path
        path = []

        while len(queue) > 0:
            # Get the current cell
            current_cell = queue.pop(0)
            # Mark the current cell as explored
            explored[current_cell] = True
            # Check if the current cell is the target
            if current_cell == target:
                # Get the path from the target to the head
                path = self.get_path_from_target_to_head(parent, target)
                break
            # Get the neighbours of the current cell
            neighbours = self.get_neighbours(current_cell, valid_moves, valid_cells)
            # Loop through the neighbours
            for neighbour in neighbours:
                # Check if the neighbour has been explored
                if not explored[neighbour]:
                    # Add the neighbour to the queue
                    queue.append(neighbour)
                    # Mark the neighbour as explored
                    explored[neighbour] = True
                    # Set the parent of the neighbour
                    parent[neighbour] = current_cell

        self.remove_extraneous_path_cells(path, path_encoding)

        # Raise error if path not found
        if len(path) == 0 or (head not in path) or (target not in path):
            raise PathNotFoundError
        return path

    def remove_extraneous_path_cells(
        self, path: List[Tuple[int, int]], path_encoding: int
    ) -> None:
        """Removes extraneous path cells from the board layout."""
        # Change any cell with the same wire_encoding but not in the path to an empty cell
        path_set = set(path)
        for i in range(self.board_layout.shape[0]):
            for j in range(self.board_layout.shape[1]):
                if self.board_layout[i, j] == path_encoding and (i, j) not in path_set:
                    self.board_layout[i, j] = EMPTY
                elif self.board_layout[i, j] == EMPTY and (i, j) in path_set:
                    self.board_layout[i, j] = int(path_encoding)

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

    def get_neighbours(self, cell, valid_moves, valid_cells) -> List[Tuple[int, int]]:
        """Gets the valid neighbours of a cell."""
        # Initialize the list of neighbours
        neighbours = []
        # Loop through the valid moves
        for move in valid_moves:
            # Get the neighbour
            neighbour = (cell[0] + move[0], cell[1] + move[1])
            # Check if the neighbour is valid
            if self.is_valid_cell(neighbour, valid_cells):
                # Add the neighbour to the list of neighbours
                neighbours.append(neighbour)
        return neighbours

    def is_valid_cell(
        self, cell: Tuple[int, int], valid_cells: List[Tuple[int, int]]
    ) -> bool:
        """Checks if a cell is valid."""
        # Check if the cell is within the bounds of the board
        if (
            cell[0] < 0
            or cell[0] >= self.board_layout.shape[0]
            or cell[1] < 0
            or cell[1] >= self.board_layout.shape[1]
        ):
            return False
        # Check if the cell has a valid encoding
        if self.board_layout[cell] not in valid_cells:
            return False
        return True

    @staticmethod
    def get_path_length(path: List[Tuple[int, int]]) -> int:
        """Gets the length of a path"""
        return len(path)

    def count_path_bends(self, path: List[Tuple[int, int]]) -> int:
        """Counts the number of bends in a path"""
        bends = 0
        for i in range(1, len(path) - 1):
            # Get the previous and next positions
            prev_pos = path[i - 1]
            next_pos = path[i + 1]
            # Check if the current position is a bend
            if self.is_bend(prev_pos, path[i], next_pos):
                bends += 1
        return bends

    @staticmethod
    def is_bend(
        prev_pos: Tuple[int, int], pos: Tuple[int, int], next_pos: Tuple[int, int]
    ) -> bool:
        """Checks if a position is a bend"""
        prev_row, prev_col = prev_pos
        next_row, next_col = next_pos
        # Get the row and column of the current position
        row, col = pos
        # Check if the current position is a bend
        if (row == prev_row and row == next_row) or (
            col == prev_col and col == next_col
        ):
            return False
        return True

    def proportion_filled(self) -> float:
        """Returns the proportion of the board that is filled with wires"""
        filled_positions = np.count_nonzero(self.board_layout)
        # Get the total number of positions
        total_positions = self.board_layout.shape[0] * self.board_layout.shape[1]
        # Return the percentage of filled positions
        return filled_positions / total_positions

    def distance_between_heads_and_targets(self) -> List[float]:
        """Returns the L1 distance between the heads and targets of the wires"""
        distances = []
        for head, target in zip(self.heads, self.targets):
            distances.append(self.get_distance_between_cells(head, target))
        return distances

    @staticmethod
    def get_distance_between_cells(
        cell1: Tuple[int, int], cell2: Tuple[int, int]
    ) -> float:
        """Returns the L1 distance between two cells"""
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

    def remove_wire(self, wire_index: int) -> None:
        """Removes a wire from the board"""
        if wire_index >= len(self.heads):
            raise ValueError(
                f"Wire index out of range. Only {len(self.heads)} wires on the board."
            )
        else:
            # Get the head, target and path of the wire
            head, target, path = (
                self.heads[wire_index],
                self.targets[wire_index],
                self.paths[wire_index],
            )
            # Remove the wire from the board
            for pos in path:
                self.board_layout[pos[0]][pos[1]] = 0
            # Remove the wire from the list of heads, targets and paths
            self.heads.pop(wire_index)
            self.targets.pop(wire_index)
            self.paths.pop(wire_index)

            assert (
                len(self.heads) == len(self.targets) == len(self.paths)
            ), "Heads, targets and paths not of equal length"
            assert head not in self.heads, "Head not removed"
            assert target not in self.targets, "target not removed"
            assert path not in self.paths, "Path not removed"

    def get_board_layout(self) -> np.ndarray:
        # Returns the board layout
        return self.board_layout

    def get_board_statistics(self) -> Dict[str, Union[int, float]]:
        """Returns a dictionary of statistics about the board"""

        num_wires = len(self.heads)
        if num_wires == 0:
            num_wires = 1
        wire_lengths = [self.get_path_length(path) for path in self.paths]
        avg_wire_length = sum(wire_lengths) / num_wires
        wire_bends = [self.count_path_bends(path) for path in self.paths]
        avg_wire_bends = sum(wire_bends) / num_wires
        avg_head_target_distance = (
            sum(self.distance_between_heads_and_targets()) / num_wires
        )
        proportion_filled = self.proportion_filled()

        # Return summary dict
        summary = dict(
            num_wires=num_wires,
            wire_lengths=wire_lengths,
            avg_wire_length=avg_wire_length,
            wire_bends=wire_bends,
            avg_wire_bends=avg_wire_bends,
            avg_head_target_distance=avg_head_target_distance,
            percent_filled=proportion_filled,
        )

        return summary

    def is_valid_board(self) -> bool:
        """Return a boolean indicating if the board is valid.  Raise an exception if not."""
        is_valid = True
        if not self.verify_board_size():
            raise IncorrectBoardSizeError
        if self.board.wires_on_board < 0:
            raise NumAgentsOutOfRangeError
        if not self.verify_encodings_range():
            raise EncodingOutOfRangeError
        if not self.verify_number_heads_tails():
            pass
        if not self.verify_wire_validity():
            raise InvalidWireStructureError
        return is_valid

    def verify_board_size(self) -> bool:
        """Verify that the board size is correct."""
        return np.shape(self.board_layout) == (self.board.rows, self.board.cols)

    def verify_encodings_range(self) -> bool:
        """Verify that the encodings are within the correct range."""
        wires_only = np.setdiff1d(self.board_layout, np.array([EMPTY]))
        if self.board.wires_on_board == 0:
            # if no wires, we should have nothing left of the board
            return len(wires_only) == 0
        if np.min(self.board_layout) < 0:
            return False
        if np.max(self.board_layout) > 3 * self.board.wires_on_board:
            return False
        return True

    #
    def verify_number_heads_tails(self) -> bool:
        """Verify that each wire has exactly one head and one tail."""
        wires_only = np.setdiff1d(self.board_layout, np.array([EMPTY]))
        is_valid = True
        for num_wire in range(self.board.wires_on_board):
            heads = np.count_nonzero(wires_only == (num_wire * 3 + POSITION))
            tails = np.count_nonzero(wires_only == (num_wire * 3 + TARGET))
            if heads < 1 or tails < 1:
                raise MissingHeadTailError
            if heads > 1 or tails > 1:
                raise DuplicateHeadsTailsError
        return is_valid

    def verify_wire_validity(self) -> bool:
        """Verify that each wire has a valid structure."""
        for row in range(self.rows):
            for col in range(self.cols):
                cell_label = self.board_layout[row, col]
                # Don't check empty cells
                if cell_label > 0:
                    # Check whether the cell is a wiring path or a starting/target cell
                    if self.position_to_cell_type(row, col) == PATH:
                        # Wiring path cells should have two neighbours of the same wire
                        if self.num_wire_neighbours(cell_label, row, col) != 2:
                            print(
                                f"({row},{col}) == {cell_label}, {self.num_wire_neighbours(cell_label, row, col)} neighbours"
                            )
                            return False
                    else:
                        # Head and target cells should only have one neighbour of the same wire.
                        if self.num_wire_neighbours(cell_label, row, col) != 1:
                            print(
                                f"HT({row},{col}) == {cell_label}, {self.num_wire_neighbours(cell_label, row, col)} neighbours"
                            )
                            return False
        return True

    def num_wire_neighbours(self, cell_label: int, row: int, col: int) -> int:
        """Return the number of adjacent cells belonging to the same wire.

        Args:
            cell_label (int) : value of the cell to investigate
            row (int)
            col (int) : (row,col) = 2D position of the cell to investigate

            Returns:
            (int) : The number of adjacent cells belonging to the same wire.
        """
        neighbours = 0
        wire_num = self.cell_label_to_wire_num(cell_label)
        min_val = 3 * wire_num + PATH  # PATH=1 is lowest val
        max_val = 3 * wire_num + TARGET  # TARGET=3 is highest val
        if row > 0:
            if min_val <= self.board_layout[row - 1, col] <= max_val:  # same wire above
                neighbours += 1
        if col > 0:
            if (
                min_val <= self.board_layout[row, col - 1] <= max_val
            ):  # same wire to the left
                neighbours += 1
        if row < self.rows - 1:
            if min_val <= self.board_layout[row + 1, col] <= max_val:  # same wire below
                neighbours += 1
        if col < self.cols - 1:
            if (
                min_val <= self.board_layout[row, col + 1] <= max_val
            ):  # same wire to the right
                neighbours += 1
        return neighbours

    def swap_heads_targets(self) -> None:
        """Randomly swap the head and target of each wire.  Self.board_layout in modified in-place."""
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
                    self.board_layout[pos[0], pos[1]] = int(3 * index + POSITION)
                elif i == len(paths[index]) - 1:
                    self.board_layout[pos[0], pos[1]] = int(3 * index + TARGET)
                else:
                    self.board_layout[pos[0], pos[1]] = int(3 * index + PATH)

    def position_to_wire_num(self, row: int, col: int) -> int:
        """Returns the wire number of the given cell position

        Args:
            row (int): row of the cell
            col (int): column of the cell

        Returns:
            (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return -1
        else:
            cell_label = self.board_layout[row, col]
            return self.cell_label_to_wire_num(cell_label)

    @staticmethod
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

    def position_to_cell_type(self, row: int, col: int) -> int:
        """
        Return the type of cell at position (row, col) in self.layout
            0 = empty
            1 = path
            2 = position (starting position)
            3 = target

        Args:
            row (int) : The row of the cell
            col (int) : The column of the cell

        Returns:
            (int) : The type of cell (0-3) as detailed above.
        """
        cell = self.board_layout[row, col]
        if cell == 0:
            return cell
        else:
            return ((cell - 1) % 3) + 1


def board_processor_tests(n: int, p: Optional[float] = 0) -> None:
    """Runs a series of tests on the board processors."""
    generator_list = [
        RandomWalkBoard,
        BFSBoard,
        JAXLSystemBoard,
        NumberLinkBoard,
        WFCBoard,
    ]
    fill_methods = [None, BFS_fill, LSystem_fill, None, None]
    for index, generator in enumerate(generator_list):
        start_time = time.time()
        summary = {}
        print(generator.__name__)
        print("Summary of results:")
        for i in range(n):
            board = generator(10, 10, 5)
            if fill_methods[index]:
                fill_methods[index](board)
            if p > random.random():
                print(board.return_solved_board())
            boardprocessor = BoardProcessor(board)
            summary[i] = boardprocessor.get_board_statistics()

        print(f"Time taken: {time.time() - start_time} for {n} boards")
        for key, value in summary[0].items():
            if type(value) != list:
                print(f"{key}: {np.mean([summary[i][key] for i in range(n)])}")

        print("-----------------------")


def BFS_fill(board: BFSBoard) -> None:
    """Fills the board with a BFS algorithm."""
    test_threshold_dict = {"min_bends": 2, "min_length": 3}
    clip_nums = [2, 2] * 10
    clip_methods = ["shortest", "min_bends"] * 10
    board.fill_clip_with_thresholds(
        clip_nums, clip_methods, verbose=False, threshold_dict=test_threshold_dict
    )


def LSystem_fill(board: JAXLSystemBoard) -> None:
    """Fills the board with an LSystem algorithm."""
    board.fill(n_steps=100, pushpullnone_ratios=[2, 0.5, 1])


if __name__ == "__main__":
    # Fill and process 1000 boards
    board_processor_tests(1000)

    # Sample Usage

    # Create a board from a numpy array
    board = np.array(
        [
            [11, 10, 7, 7, 7, 7, 0, 0, 0, 0],
            [10, 10, 7, 7, 8, 7, 0, 0, 9, 0],
            [10, 10, 12, 7, 7, 7, 7, 7, 7, 14],
            [13, 13, 13, 13, 0, 13, 13, 7, 7, 13],
            [13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
            [13, 15, 4, 6, 4, 4, 4, 13, 13, 0],
            [0, 0, 4, 4, 4, 0, 4, 0, 0, 0],
            [1, 1, 3, 1, 1, 1, 4, 5, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ]
    )

    boardprocessor = BoardProcessor(board)

    # Shuffle wire encodings
    boardprocessor.shuffle_wire_encodings()
    print(boardprocessor.get_board_layout())

    # Remove a wire
    boardprocessor.remove_wire(0)
    print(boardprocessor.get_board_layout())

    # Get Board Statistics
    summary_dict = boardprocessor.get_board_statistics()

    # Create a RandomWalkBoard
    # board_ = RandomWalkBoard(10, 10, 5)
    board_ = NumberLinkBoard(10, 10, 5)
    print(f"{board_.return_solved_board()}")
    boardprocessor_ = BoardProcessor(board_)

    # Shuffle wire encodings
    boardprocessor_.shuffle_wire_encodings()
    print("Shuffled Wire Encodings")
    print(f"{boardprocessor_.get_board_layout()}")

    # Remove a wire
    boardprocessor_.remove_wire(0)
    print("Removed Wire")
    print(f"{boardprocessor_.get_board_layout()}")

    # Get Board Statistics
    summary_dict_ = boardprocessor.get_board_statistics()
    print(summary_dict_)

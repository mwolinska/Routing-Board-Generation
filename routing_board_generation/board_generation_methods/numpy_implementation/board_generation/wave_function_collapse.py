# ------------------------------------------------------------------------------
# Portions of this code are based on, or adapted from, the work of Isaac Karth:

# MIT License

# Copyright (c) 2020 Isaac Karth

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------
import numpy as np
from copy import deepcopy

from routing_board_generation.board_generation_methods.numpy_implementation.data_model.abstract_board import (
    AbstractBoard,
)


class AbstractTile:
    """
    Used to represent any possible 1x1 tile.

    We store information about:
    - The connections to the other tiles
    - The neighbours that can connect to this tile
    - The exclusions that cannot connect to this tile
    """

    @property
    def get_all_tiles(self):
        """
        Return all possible tiles, labelled by their connections
        """
        all_tiles = [
            frozenset(),
            frozenset({"top", "bottom"}),
            frozenset({"left", "right"}),
            frozenset({"top", "left"}),
            frozenset({"top", "right"}),
            frozenset({"bottom", "right"}),
            frozenset({"bottom", "left"}),
            frozenset({"top"}),
            frozenset({"right"}),
            frozenset({"bottom"}),
            frozenset({"left"}),
        ]
        return all_tiles

    @property
    def get_all_directions(self):
        """
        Return all possible directions to connect to the tile
        """
        return ["top", "bottom", "left", "right"]

    def get_reverse_direction(self, direction):
        """
        Return the opposite direction
        """
        reverse_directions = {
            "top": "bottom",
            "bottom": "top",
            "left": "right",
            "right": "left",
        }
        return reverse_directions[direction]

    def add_neighbours_exclusions(self):
        """
        For a given tile, add the neighbours and exclusions.
        These are stored by looking at every possible direction, and checking if
        the tile can connect to the other tile in that direction.
        """
        # Add the neighbours
        for candidate in self.get_all_tiles:
            # Loop through all possible directions to connect to the tile
            for direction in self.get_all_directions:
                # Reverse the directions for the other tile
                reverse_direction = self.get_reverse_direction(direction)
                # Check if the tile can connect to the other tile
                if direction in self.connections and reverse_direction in candidate:
                    # Add the other tile to the neighbours
                    self.neighbours[direction].add(candidate)
                # Also ok if neither tile trying to connect to the other
                elif (
                    direction not in self.connections
                    and reverse_direction not in candidate
                ):
                    self.neighbours[direction].add(candidate)
                # Otherwise, add the other tile to the exclusions
                else:
                    self.exclusions[direction].add(candidate)


class Tile(AbstractTile):
    def __init__(self, connections):
        """
        Specify a tile by its connections
        """
        self.connections = connections
        self.idx = self.get_all_tiles.index(connections)
        # Specify the pieces that can connect to this tile
        self.neighbours = {"top": set(), "bottom": set(), "left": set(), "right": set()}
        # Specify the pieces that cannot connect to this tile
        self.exclusions = {"top": set(), "bottom": set(), "left": set(), "right": set()}
        # Add the neighbours and exclusions
        self.add_neighbours_exclusions()


class WFCUtils:
    """
    Utility functions for the WFC algorithm
    """

    def __init__(self):
        self.abstract_tile = AbstractTile()

    def check_side(self, side1, side2):
        """
        (Unsure!)
        """
        ratio = 1.0
        num_pixels = np.prod(side1.shape)
        threshold = ratio * num_pixels
        if np.sum(side1 == side2) >= threshold:
            return True
        elif np.sum(side1[:-1] == side2[1:]) >= threshold:
            return True
        elif np.sum(side1[1:] == side2[:-1]) >= threshold:
            return True

    def all_valid_choices(self, i, j, rows, cols, num_tiles):
        """
        Used to initialise the choices dictionary.
        Also used in reduce_prob to remove invalid choices.
        """
        choices = np.arange(num_tiles).tolist()
        # TODO: Remove some boundary tiles from the choices
        if i == 0:
            choices = [x for x in choices if x not in [1, 3, 4, 7]]
        if i == rows - 1:
            choices = [x for x in choices if x not in [1, 5, 6, 9]]
        if j == 0:
            choices = [x for x in choices if x not in [2, 3, 6, 10]]
        if j == cols - 1:
            choices = [x for x in choices if x not in [2, 4, 5, 8]]
        return choices

    def reduce_prob(self, choices, tiles, row, col, rows, cols, TILE_IDX_LIST):
        """
        Reduce the probability of the remaining choices by removing invalid choices.

        Args:
            choices: The current choices for each tile
            tiles: The current tiles
            row: The current row
            col: The current column
            rows: The number of rows
            cols: The number of columns
            TILE_IDX_LIST: The list of possible tiles
        """
        neighbor_choices = []
        # Changed this to be a function of the tile
        valid_choices = self.all_valid_choices(row, col, rows, cols, len(TILE_IDX_LIST))
        # Check the top, bottom, left, right neighbors
        for i, j, direction in [
            [row - 1, col, "bottom"],
            [row + 1, col, "top"],
            [row, col - 1, "right"],
            [row, col + 1, "left"],
        ]:
            exclusion_idx_list = []
            if 0 <= i < rows and 0 <= j < cols:
                # Look at every choice for the neighbor
                for tile_idx in choices[(i, j)]:
                    tile = Tile(tiles[tile_idx])
                    exclusion_idx_list.append(tile.exclusions[direction])
            total_num = len(exclusion_idx_list)
            if len(exclusion_idx_list) > 0:
                for idx in TILE_IDX_LIST:
                    tile_connections = tiles[idx]
                    vote = 0
                    for exclusion in exclusion_idx_list:
                        # Need to convert to indexes
                        if tile_connections in exclusion:
                            vote += 1
                    # If every neighbor has this tile as an exclusion, remove it
                    if (vote == total_num) and (idx in valid_choices):
                        valid_choices.remove(idx)
        if len(valid_choices) == 0:
            return None
        else:
            choices[(row, col)] = valid_choices
            return choices

    def get_min_entropy_coord(self, entropy_board, observed):
        """
        Return the coordinates of the tile with the minimum entropy.

        If there are multiple tiles with the same minimum entropy, return one
        of them at random.

        If there are no tiles with entropy > 0, return -1, -1

        Args:
            entropy_board (np.array): The entropy board
            observed (np.array): The observed board

        Returns:
            (int, int): The coordinates of the tile with the minimum entropy
        """
        rows, cols = entropy_board.shape
        min_row, min_col = -1, -1
        min_entropy = 1000
        coord_list = []
        for row in range(rows):
            for col in range(cols):
                if not observed[row, col]:
                    if 1 <= entropy_board[row, col] < min_entropy:
                        min_entropy = entropy_board[row, col]
                        coord_list = []
                        coord_list.append((row, col))
                    elif 1 <= entropy_board[row, col] == min_entropy:
                        coord_list.append((row, col))
        if len(coord_list) > 0:
            coord_idx = np.random.choice(np.arange(len(coord_list)))
            min_row, min_col = coord_list[coord_idx]
            return min_row, min_col
        else:
            return -1, -1

    def update_entropy(self, choices, rows, cols):
        """
        Update the entropy board
        """
        entropy_board = np.zeros(shape=(rows, cols))
        for row in range(rows):
            for col in range(cols):
                entropy_board[row, col] = len(choices[(row, col)])
        return entropy_board

    def step(self, info, row_col=None):
        """
        Perform one step of the WFC algorithm
        """
        entropy_board = info["entropy_board"]
        tile_idx_list = info["tile_idx_list"]
        observed = info["observed"]
        choices = info["choices"]
        history = info["history"]
        canvas = info["canvas"]
        tiles = info["tiles"]
        rows = info["rows"]
        cols = info["cols"]
        weights = info["weights"]
        if row_col:
            row, col = row_col
        else:
            row, col = self.get_min_entropy_coord(entropy_board, observed)
        # TODO: change here to weighted random choice, include
        # custom weights for each tile
        relevant_weights = [weights[tile_idx] for tile_idx in choices[(row, col)]]
        relevant_weights = np.array(relevant_weights) / np.sum(relevant_weights)
        state = np.random.choice(choices[(row, col)], p=relevant_weights)
        history.append((row, col, state, choices[(row, col)]))
        choices_temp = deepcopy(choices)
        choices_temp[(row, col)] = [state]
        retract = False

        # compute new probability for 4 immediate neighbors
        for i, j in [[row - 1, col], [row + 1, col], [row, col - 1], [row, col + 1]]:
            if 0 <= i < rows and 0 <= j < cols:
                if not observed[i, j]:
                    attempt = self.reduce_prob(
                        choices_temp, tiles, i, j, rows, cols, tile_idx_list
                    )
                    if attempt:
                        choices_temp = attempt
                    else:
                        retract = True
                        break

        canvas[row, col] = state
        observed[row, col] = True
        info["entropy_board"] = entropy_board
        info["observed"] = observed
        info["choices"] = choices_temp
        info["history"] = history
        info["canvas"] = canvas
        info["tiles"] = tiles

        return info, retract


class WFCBoard(AbstractBoard):
    def __init__(self, rows: int, cols: int, num_agents: int):
        """

        Args:
            x: width of the board
            y: height of the board
            num_agents: number of agents
        """
        self.x = rows
        self.y = cols
        self.grid = [[None for i in range(self.x)] for j in range(self.y)]
        # Generate the tile set. This includes how tiles can connect to each other
        self.abstract_tile = AbstractTile()
        self.weights = self.generate_weights(self.x, self.y, num_agents)
        # self
        self.utils = WFCUtils()
        self.num_agents = num_agents

        # Generate the boards
        _, self.wired_board, self.unwired_board = self.wfc()

    def generate_weights(self, x, y, num_agents):
        """
        Currently just hard-coding a set of weights for the tiles.

        TODO: make this a function of the board size and number of agents

        Returns:
            weights (list): A list of weights for each tile
        """
        weights = [
            6,  # empty
            7,
            7,  # wire
            1,
            1,
            1,
            1,  # turn
            0.5,
            0.5,
            0.5,
            0.5,  # start / end
        ]
        return weights

    def wire_separator(self, final_canvas):
        """
        Given a solved board, separate the wires into individual wires.

        Pseudo code:
        1. Whilst there are still wires on the board:
            1.1. Find the first wire
            1.2. Follow the wire until it ends
            1.3. Add the wire to the output board
            1.4. Remove the wire from the input board

        Args:
            final_canvas (np.array): The solved board with wires not separated

        Returns:
            output_array (np.array): The solved board with wires separated

        """
        canvas = deepcopy(final_canvas)
        # Initialise the output board
        output_board = np.zeros(shape=(self.y, self.x), dtype=int)
        # Initialise the wire counter
        wire_counter = 0
        # Loop through the board, looking for wires
        while np.any(canvas > 6):
            # Find the first start of a wire
            # This corresponds to values 7, 8, 9, 10
            start = tuple(np.argwhere(canvas > 6)[0])
            # Follow the wire until it ends
            wire = self.follow_wire(start, canvas)
            # Add the wire to the output board
            # Change this to be proper values, not just the wire counter
            output_board[start] = 2 + 3 * wire_counter
            canvas[start] = 0
            output_board[wire[-1]] = 3 + 3 * wire_counter
            canvas[wire[-1]] = 0
            wire = wire[1:-1]
            for part in wire:
                output_board[part] = 1 + 3 * wire_counter
                # Remove the wire from the input board
                canvas[part] = 0
            # Increment the wire counter
            wire_counter += 1

        return output_board, wire_counter

    def follow_wire(self, start, canvas):
        """
        From a given start, follow the wire until it ends.

        Args:
            start:  Coordinates of the start of the wire
            canvas: The board to follow the wire on
        Returns:
            wire: List of coordinates of the wire
        """
        # Initialise the wire
        wire = [start]
        # Initialise the current position
        current_position = start
        # Initialise the current direction
        current_direction = tuple(
            self.abstract_tile.get_all_tiles[canvas[tuple(start)]]
        )[0]
        # Loop until the wire ends
        while True:
            directions = {
                "top": (-1, 0),
                "bottom": (1, 0),
                "left": (0, -1),
                "right": (0, 1),
            }
            # Find the next position
            next_position = tuple(
                [
                    current_position[i] + directions[current_direction][i]
                    for i in range(2)
                ]
            )
            # Check if the next position is an end point
            if 7 <= canvas[next_position] <= 10:
                # Add the end point to the wire
                wire.append(next_position)
                # Break the loop
                break
            # Otherwise, add the next position to the wire
            wire.append(next_position)
            # Update the current position
            current_position = next_position
            # Update the current direction
            possible_directions = set(
                deepcopy(self.abstract_tile.get_all_tiles[canvas[next_position]])
            )
            if current_direction == "top":
                possible_directions.remove("bottom")
            elif current_direction == "bottom":
                possible_directions.remove("top")
            elif current_direction == "left":
                possible_directions.remove("right")
            elif current_direction == "right":
                possible_directions.remove("left")
            current_direction = list(possible_directions)[0]

        return wire

    def remove_wires(self, wired_output, wire_counter):
        """
        Given a solved board with wires, remove excess wires.

        Args:
            wired_output: solved board with wires
            wire_counter: number of wires on the board

        TODO: Incorporate Ugo and Randy's fancy removal methods.
        """
        output = deepcopy(wired_output)
        # Loop through the wires
        upper_limit = 3 * self.num_agents
        for i in range(self.x):
            for j in range(self.y):
                if output[i, j] > upper_limit:
                    output[i, j] = 0

        return output

    def update_weights(self):
        """
        Change the weights to make it more likely to generate a board with
        more wires.

        Works by decreasing the weights corresponding to turns, straight lines, and empty space;
        increase the weights corresponding to start and end points.

        Edit weights in place.
        """
        for i in range(len(self.weights)):
            if i < 7:
                self.weights[i] *= 0.8
            else:
                self.weights[i] *= 1.2
        return

    def wfc(self):
        """
        Main function that implements the WFC algorithm to create boards.

        Returns:
            info: dictionary containing information about the board
            wired_output: final solved board as a numpy array
            unwired_output: final solved board, with wires removed, as a numpy array
        """
        cols = self.x
        rows = self.y
        tiles = AbstractTile().get_all_tiles
        tile_idx_list = list(range(len(tiles)))
        utils = WFCUtils()
        history = []
        retract = False
        num_tiles = len(tiles)
        observed = np.zeros(shape=(rows, cols))
        canvas = np.zeros(shape=(rows, cols), dtype=int) - 1
        entropy_board = np.zeros(shape=(rows, cols)) + num_tiles
        weights = self.weights
        choices = {}
        for i in range(rows):
            for j in range(cols):
                choices[(i, j)] = utils.all_valid_choices(i, j, rows, cols, num_tiles)

        info = dict(
            entropy_board=entropy_board,
            observed=observed,
            choices=choices,
            history=history,
            canvas=canvas,
            tiles=tiles,
            rows=rows,
            cols=cols,
            tile_idx_list=tile_idx_list,
            weights=weights,
        )

        info_history = []
        info_history_full = []

        while not np.all(info["observed"] == True):
            info_history.append(deepcopy(info))
            info, retract = utils.step(info)
            info_history_full.append(deepcopy(info))

            while retract:
                # undo one step
                last_step = info["history"].pop()
                last_row, last_col, last_choice, valid_choices = last_step
                valid_choices.remove(last_choice)
                if len(valid_choices) > 0:
                    info["choices"][(last_row, last_col)] = valid_choices
                else:
                    info = info_history.pop()
                info, retract = utils.step(info, (last_row, last_col))
                info_history_full.append(deepcopy(info))

            entropy_board = utils.update_entropy(choices, rows, cols)
        info_history.append(deepcopy(info))
        canvas = info["canvas"]
        # Need to separate the individual wires
        wired_output, wire_counter = self.wire_separator(canvas)
        unwired_output = np.zeros(shape=(rows, cols))
        # Remove wires to get the number of wires,
        # change weights and repeat if we have too few wires
        if wire_counter < self.num_agents:
            # Change the weights
            self.update_weights()
            # Repeat
            return self.wfc()
        else:
            # Remove the wires
            wired_output = self.remove_wires(wired_output, wire_counter)
        # Create the unwired output
        for i in range(rows):
            for j in range(cols):
                if wired_output[i, j] % 3 == 1:
                    unwired_output[i, j] = 0
                else:
                    unwired_output[i, j] = wired_output[i, j]
        return info, wired_output, unwired_output

    def return_training_board(self) -> np.ndarray:
        """
        Returns the board as a numpy array, with wires removed.
        """
        # Change data type to int
        self.unwired_board = self.unwired_board.astype(int)
        return self.unwired_board

    def return_solved_board(self) -> np.ndarray:
        """
        Returns the board as a numpy array, with wires.
        """
        # Change data type to int
        self.wired_board = self.wired_board.astype(int)
        return self.wired_board


if __name__ == "__main__":
    # These correspond to the weights we will use to pick tiles
    # Organised by index
    board = WFCBoard(8, 8, 5)
    unwired_output = board.return_training_board()
    wired_output = board.return_solved_board()
    print(wired_output)
    print(unwired_output)

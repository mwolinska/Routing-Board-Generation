from ic_routing_board_generation.board_generator.numpy_data_model.abstract_board import AbstractBoard
from ic_routing_board_generation.board_generator.jax_utils.post_processor_utils_jax import extend_wires_jax
from dataclasses import dataclass
from ic_routing_board_generation.board_generator.numpy_data_model.abstract_board import AbstractBoard
from ic_routing_board_generation.board_generator.jax_utils.post_processor_utils_jax import extend_wires_jax, training_board_from_solved_board_jax
#import numpy as np

from jax import Array
from jax.random import PRNGKey
import jax.numpy as jnp
import jax

from jumanji.environments.routing.connector.constants import EMPTY, PATH, POSITION, TARGET
STARTING_POSITION = POSITION  # Resolve ambiguity of POSITION constant

@jax.disable_jit()
#@jax.jit
class RandomSeedBoard(AbstractBoard):
    """ The boards are 2D np.ndarrays of wiring routes on a printed circuit board.

    The coding of the boards is as follows:
    Empty cells are coded 0.
    Positions (starting positions) are encoded starting from 2 in multiples of 3: 2, 5, 8, ...
    Targets are encoded starting from 3 in multiples of 3: 3, 6, 9, ...
    Wiring paths connecting the position/target pairs are encoded starting
            at 1 in multiples of 3: 1, 4, 7, ...

    Args:
        rows, cols (int, int) : Dimensions of the board.
        num_agents (int) : Number of wires to attempt to add to the board

    """
    def __init__(self, rows: int, cols: int, num_agents: int = 0):
        super().__init__(rows, cols, num_agents)

        self._rows = rows
        self._cols = cols
        self.grid_size = jax.lax.select(rows > cols, rows, cols)
        self._num_agents = num_agents
        # Limit the number of wires to number_cells/3 to ensure we can fit them all
        max_num = jax.lax.select(num_agents > rows*cols/3, jnp.array(rows*cols/3, int), num_agents)
        self._wires_on_board = max_num


    #@jax.disable_jit()
    #@jax.jit
    def return_seeded_board(self, key: PRNGKey) -> Array:
        """ Generate and return an array of the board with the connecting wires encoded.

        Args:
            key (PRNGKey) : Random number generator key

        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """
        board_layout = jnp.zeros((self._rows, self._cols), int)
        for wire_num in range(self._num_agents):
            # Pick a random starting position that is EMPTY
            key, subkey = jax.random.split(key)

            # Pick a random STARTING_POSITION cell
            def position_cond_func(position_carry):
                row, col, board_layout, _ = position_carry
                empty_above = ((row - 1) >= 0) & (board_layout[row - 1, col] == EMPTY)
                empty_below = ((row + 1) < self._rows) & (board_layout[row + 1, col] == EMPTY)
                empty_left = ((col - 1) >= 0) & (board_layout[row, col - 1] == EMPTY)
                empty_right = ((col + 1) < self._cols) & (board_layout[row, col + 1] == EMPTY)
                valid_choice = (board_layout[row, col] == EMPTY) & (row >= 0) & \
                           (empty_above | empty_below | empty_left | empty_right)
                return ~valid_choice

            def position_body_func(position_carry):
                row, col, board_layout, key = position_carry
                key, subkey = jax.random.split(key)
                position = jax.random.randint(subkey, (1,), 0, self._rows * self._cols)
                row, col = jnp.divmod(position, self._cols)
                return row[0], col[0], board_layout, key

            # Choose a random position for STARTING_POSITION that is empty and has at least one empty neighbor
            row, col = -999, -999  # Invalid position
            row, col, board_layout, _ = jax.lax.while_loop(position_cond_func, position_body_func, (row, col, board_layout, key))
            board_layout = board_layout.at[row, col].set(3 * wire_num + STARTING_POSITION)

            # Pick an adjacent cell at random to be the TARGET
            def neighbor_cond_func(neighbor_carry):
                neighbor_pos, _ = neighbor_carry
                neighbor_row, neighbor_col = neighbor_pos[0], neighbor_pos[1]
                out_of_bounds = (neighbor_row < 0) | (neighbor_row >= self._rows) |\
                                   (neighbor_col < 0) | (neighbor_col >= self._cols)
                return (board_layout[neighbor_row, neighbor_col] != EMPTY) | out_of_bounds

            def neighbor_body_func(neighbor_carry):
                neighbor_pos, key = neighbor_carry
                key, subkey = jax.random.split(key)
                neighbor_pos = jax.random.choice(subkey, neighbors_positions)
                return neighbor_pos.reshape((2,)), key

            # Choose an empty neighbor to be the TARGET
            neighbors_positions = jnp.array([jnp.array((row-1, col)), jnp.array((row+1, col)),
                                             jnp.array((row, col-1)), jnp.array((row, col+1))])
            neighbor_pos = jnp.array((-999, -999))  # Invalid position is out of bounds
            neighbor_pos, _ = jax.lax.while_loop(neighbor_cond_func, neighbor_body_func, (neighbor_pos, key))
            output_encoding = 3 * wire_num + TARGET
            board_layout = board_layout.at[neighbor_pos[0], neighbor_pos[1]].set(output_encoding)
        return board_layout

#    def return_extended_board(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True) -> Array:
#        """ Generate and return an array of the board with the connecting wires zeroed out.
#
#        Args:
#            key (PRNGKey) : Random number generator key
#            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
#            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
#
#        Returns:
#            (Array) : 2D layout of the board with all wirings encoded
#        """
#        board_layout = self.return_seeded_board(key)
#        board_layout = extend_wires_jax(board_layout, key, randomness, two_sided)
#        return board_layout

    def return_solved_board(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True) -> Array:
        """ Generate and return an array of the board with the connecting wires zeroed out.

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.

        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """
        board_layout = self.return_seeded_board(key)
        board_layout = extend_wires_jax(board_layout, key, randomness, two_sided)
        return board_layout

    #@jax.jit
    def return_training_board(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True) -> Array:
        """ Generate and return an array of the board with the connecting wires zeroed out.

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.

        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """
        board_layout = self.return_solved_board(key, randomness, two_sided)
        training_board = training_board_from_solved_board_jax(board_layout)
        """
        rows, cols = board_layout.shape
        for row in range(rows):
            for col in range(cols):
                cell_encoding = board_layout[row, col]
                cell_encoding = jax.lax.select((cell_encoding % 3) == PATH, 0, cell_encoding)
                board_layout.at[row,col].set(cell_encoding)
        """
        return training_board

    #@jax.jit
    def generate_starts_ends(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True):
        """  Call generate, take the first and last cells of each wire

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.

        Returns:
            (Array) : List of starting points, dimension 2 x num_agents
            (Array) : List of target points, dimension 2 x num_agents
        """
        board = self.return_solved_board(key, randomness, two_sided)

        def find_positions(wire_id):
            wire_positions = board == 3 * wire_id + POSITION
            wire_targets = board == 3 * wire_id + TARGET

            # Compute indices where wire_positions and wire_targets are True
            start_indices = jnp.argwhere(wire_positions, size=2)
            end_indices = jnp.argwhere(wire_targets, size=2)

            # Take the first valid index (row)
            start = start_indices[0]
            end = end_indices[0]
            return start, end

        wire_ids = jnp.arange(self._num_agents)
        starts_ends = jax.vmap(find_positions)(wire_ids)
        #jax.debug.print("{x}", x=starts_ends)
        #jax.debug.print("{x}", x=type(starts_ends))
        starts, ends = starts_ends[0], starts_ends[1]

        # Want starts to be a tuple of arrays, first array is x coords, second is y coords
        starts = (starts[:, 0], starts[:, 1])
        ends = (ends[:, 0], ends[:, 1])

        return starts, ends

if __name__ == '__main__':
    board_object = RandomSeedBoard(3, 3, 3)

from ic_routing_board_generation.board_generator.jax_utils.grid_utils import (
    optimise_wire,
)
from ic_routing_board_generation.board_generator.jax_utils.post_processor_utils_jax import (
    extend_wires_jax,
    training_board_from_solved_board_jax,
)

import jax.numpy as jnp
import jax
import chex
from typing import Tuple

from jumanji.environments.routing.connector.constants import (
    EMPTY,
    PATH,
    POSITION,
    TARGET,
)

STARTING_POSITION = POSITION  # Resolve ambiguity of POSITION constant


class RandomSeedBoard:
    """The boards are 2D arrays of wiring routes on a printed circuit board.

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
        self._rows = rows
        self._cols = cols
        self.grid_size = jax.lax.select(rows > cols, rows, cols)
        self._num_agents = num_agents  # Number of wires requested
        # Limit the number of wires to number_cells/3 to ensure we can fit them all
        max_num = jax.lax.select(
            num_agents > rows * cols // 3, jnp.array(rows * cols // 3, jnp.int32), num_agents
        )
        self._wires_on_board = max_num  # Actual number of wires on the board

    def return_seeded_board(self, key: chex.PRNGKey) -> chex.Array:
        """Generate and return an array of the board with the connecting wires encoded.

        Args:
            key (chex.PRNGKey) : Random number generator key

        Returns:
            (chex.Array) : 2D layout of the board with all wirings encoded
        """
        board_layout = jnp.zeros((self._rows, self._cols), jnp.int32)
        for wire_num in range(self._num_agents):
            # Pick a random starting position that is EMPTY
            key, subkey = jax.random.split(key)

            def position_cond_func(
                position_carry: Tuple[int, int, chex.Array, int]
            ) -> bool:
                """Check if the position is valid."""
                row, col, board_layout, _ = position_carry
                empty_above = ((row - 1) >= 0) & (board_layout[row - 1, col] == EMPTY)
                empty_below = ((row + 1) < self._rows) & (
                    board_layout[row + 1, col] == EMPTY
                )
                empty_left = ((col - 1) >= 0) & (board_layout[row, col - 1] == EMPTY)
                empty_right = ((col + 1) < self._cols) & (
                    board_layout[row, col + 1] == EMPTY
                )
                valid_choice = (
                    (board_layout[row, col] == EMPTY)
                    & (row >= 0)
                    & (empty_above | empty_below | empty_left | empty_right)
                )
                return ~valid_choice

            def position_body_func(
                position_carry: Tuple[int, int, chex.Array, int]
            ) -> Tuple[int, int, chex.Array, int]:
                """Choose a random position on the board"""
                row, col, board_layout, key = position_carry
                key, subkey = jax.random.split(key)
                position = jax.random.randint(subkey, (1,), 0, self._rows * self._cols)
                row, col = jnp.divmod(position, self._cols)
                return row[0], col[0], board_layout, key

            # Choose a random position for STARTING_POSITION that is empty and has at least one empty neighbor
            row, col = -999, -999
            row, col, board_layout, _ = jax.lax.while_loop(
                position_cond_func, position_body_func, (row, col, board_layout, key)
            )
            board_layout = board_layout.at[row, col].set(
                3 * wire_num + STARTING_POSITION
            )

            def is_neighbor_valid(
                neighbor_carry: Tuple[chex.Array, chex.PRNGKey]
            ) -> bool:
                """Check if the neighbor is valid."""
                neighbor_pos, _ = neighbor_carry
                neighbor_row, neighbor_col = neighbor_pos[0], neighbor_pos[1]
                out_of_bounds = (
                    (neighbor_row < 0)
                    | (neighbor_row >= self._rows)
                    | (neighbor_col < 0)
                    | (neighbor_col >= self._cols)
                )
                return (
                    board_layout[neighbor_row, neighbor_col] != EMPTY
                ) | out_of_bounds

            def choose_neighbor(neighbor_carry: Tuple[chex.Array, chex.PRNGKey]) -> Tuple[chex.Array, chex.PRNGKey]:
                """Choose a random neighbor"""
                neighbor_pos, key = neighbor_carry
                key, subkey = jax.random.split(key)
                neighbor_pos = jax.random.choice(subkey, neighbors_positions)
                return neighbor_pos.reshape((2,)), key

            # Choose an empty neighbor to be the TARGET
            neighbors_positions = jnp.array(
                [
                    jnp.array((row - 1, col)),
                    jnp.array((row + 1, col)),
                    jnp.array((row, col - 1)),
                    jnp.array((row, col + 1)),
                ]
            )
            neighbor_pos = jnp.array((-999, -999))
            neighbor_pos, _ = jax.lax.while_loop(
                is_neighbor_valid, choose_neighbor, (neighbor_pos, key)
            )
            output_encoding = 3 * wire_num + TARGET
            board_layout = board_layout.at[neighbor_pos[0], neighbor_pos[1]].set(
                output_encoding
            )
        return board_layout

    def return_solved_board(
        self,
        key: chex.PRNGKey,
        randomness: float = 0.0,
        two_sided: bool = True,
        extension_iterations: int = 1,
        extension_steps: int = 1e23,
    ) -> chex.Array:
        """Generate and return an array of the board with the connecting wires zeroed out.

        Args:
            key (chex.PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization. (Default = 1)
            extension_steps (int): Max number of extension loops to perform during each iteration.  (Default = no limit)
                                Note that sometimes wires might extend one cell or multiple cells per steps.

        Returns:
            (chex.Array) : 2D layout of the board with all wirings encoded
        """
        from copy import deepcopy

        key, seedkey = jax.random.split(key)

        board_layout = self.return_seeded_board(seedkey)

        # While Loop: Loop through specified num of iterations as long as convergence hasn't occurred
        def ext_iterations_cond(carry):
            _, _, iteration_num, converged = carry
            return ~converged & (iteration_num < extension_iterations)

        def ext_iteration_body(carry):
            board_layout, key, iteration_num, converged = carry
            iteration_num = iteration_num + 1

            key, extkey, optkey = jax.random.split(key, 3)
            board_layout = extend_wires_jax(
                board_layout, extkey, randomness, two_sided, extension_steps
            )
            optkeys = jax.random.split(optkey, self._wires_on_board)
            board_layout_save = deepcopy(board_layout)

            # Optimise each wire individually
            def optimise_loop_func(wire_num, carry):
                board_layout, keys = carry
                board_layout = optimise_wire(keys[wire_num], board_layout, wire_num)
                carry = (board_layout, keys)
                return carry

            carry = (board_layout_save, optkeys)
            board_layout, _ = jax.lax.fori_loop(
                0, self._wires_on_board, optimise_loop_func, carry
            )

            carry = (board_layout, key, iteration_num, converged)
            return carry

        converged = False
        iteration_num = 0
        carry = (board_layout, key, iteration_num, converged)
        board_layout, key, iteration_num, converged = jax.lax.while_loop(
            ext_iterations_cond, ext_iteration_body, carry
        )
        return board_layout

    def return_training_board(
        self,
        key: chex.PRNGKey,
        randomness: float = 0.0,
        two_sided: bool = True,
        extension_iterations: int = 1,
        extension_steps: int = 1e23,
    ) -> chex.Array:
        """Generate and return an array of the board with the connecting wires zeroed out.

        Args:
            key (chex.PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization.
            extension_steps (int): Max number of extension loops to perform during each iteration.
                                Note that sometimes wires may extend multiple cells per extension loop.
        Returns:
            (chex.Array) : 2D layout of the board with all wirings encoded
        """
        board_layout = self.return_solved_board(
            key, randomness, two_sided, extension_iterations, extension_steps
        )
        training_board = training_board_from_solved_board_jax(board_layout)
        return training_board

    def generate_starts_ends(
        self,
        key: chex.PRNGKey,
        randomness: float = 0.0,
        two_sided: bool = True,
        extension_iterations: int = 1,
        extension_steps: int = 1e23,
    ) -> chex.Array:
        """Call generate, take the first and last cells of each wire

        Args:
            key (chex.PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization.
            extension_steps (int): Max number of extension loops to perform during each iteration.
                                Note that sometimes wires may extend multiple cells per extension loop.
        Returns:
            (chex.Array) : List of starting points, dimension 2 x num_agents
            (chex.Array) : List of target points, dimension 2 x num_agents
        """
        board_layout = self.return_solved_board(
            key, randomness, two_sided, extension_iterations, extension_steps
        )

        def find_positions(wire_id):
            wire_positions = board_layout == 3 * wire_id + POSITION
            wire_targets = board_layout   == 3 * wire_id + TARGET

            # Compute indices where wire_positions and wire_targets are True
            start_indices = jnp.argwhere(wire_positions, size=2)
            end_indices   = jnp.argwhere(wire_targets,   size=2)

            # Take the first valid index (row)
            start = start_indices[0]
            end = end_indices[0]
            return start, end

        wire_ids = jnp.arange(self._num_agents)
        starts_ends = jax.vmap(find_positions)(wire_ids)
        starts, ends = starts_ends[0], starts_ends[1]

        # Want starts to be a tuple of arrays, first array is x coords, second is y coords
        starts = (starts[:, 0], starts[:, 1])
        ends = (ends[:, 0], ends[:, 1])
        return starts, ends


if __name__ == "__main__":
    board_object = RandomSeedBoard(3, 3, 3)

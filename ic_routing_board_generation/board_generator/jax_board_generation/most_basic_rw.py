from dataclasses import dataclass
from functools import partial


from typing import List, Tuple
import jax.numpy as jnp
import jax
import chex

from ic_routing_board_generation.board_generator.jax_board_generation.grid_jax import Grid
from ic_routing_board_generation.board_generator.jax_data_model.wire import Wire, create_wire, \
    stack_push

EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3
@dataclass
class Position:
    """  Class of 2D tuple of ints, indicating the size of an array or a 2D position or vector."""
    x: int
    y: int

class SequentialRandomWalk:
    """Class for generating a board using a sequential random walk algorithm."""

    def __init__(self, rows: int, cols: int, num_agents: int = 3):
        self._rows = rows
        self._cols = cols
        self._num_agents = num_agents

    def return_blank_board(self) -> chex.Array:
        return jnp.zeros((self._rows, self._cols), dtype=int)

    def pick_start(
            self, key: chex.PRNGKey, grid: chex.Array, wire_id: int, max_length: int
        ) -> Tuple[Tuple[chex.PRNGKey, chex.Array, Wire, int, int], bool]:
        """Create a wire and populate start and end points randomly.
        
        Args:
            key: jax random key.
            grid: the current grid.
            wire_id: the id of the wire to be created.
            max_length: the maximum length of the wire to be created.
            
        Returns:
            A tuple of the new key, the new grid, the new wire, the wire id, and the max length.
            can_start: a boolean indicating whether we can place a wire on the board or not.

        """
        key, subkey = jax.random.split(key)
        grid_size = len(grid) * len(grid[0])
        # See if there is any empty space in the grid
        can_start = jnp.any(grid == 0)
        def inner_create_wire(key, grid=grid, wire_id=wire_id):
            # Pick a random coordinate which is empty (i.e. 0)
            coordinate_flat = jax.random.choice(
                key=subkey,
                a=jnp.arange(grid_size),
                shape=(),
                replace=False,
                p=jnp.where(grid.flatten() == 0, 1, 0),
            )

            # Create a 2D coordinate from the flat array.
            coordinate = jnp.divmod(coordinate_flat, self._rows)
            new_wire = create_wire(grid_size, coordinate, (-1, -1), wire_id)
            new_wire = stack_push(new_wire, coordinate)
            grid = grid.at[coordinate[0], coordinate[1]].set(3 * wire_id + POSITION)
        
            return new_wire, grid
        # Use jax.lax.cond to only run the inner function if there is empty space
        dummy_wire = create_wire(grid_size, (-1, -1), (-1, -1), wire_id)
        new_wire, grid = jax.lax.cond(can_start, inner_create_wire, lambda _: (dummy_wire, grid), key)
        return (key, grid, new_wire, wire_id, max_length), can_start

    def adjacent_cells(self, cell: int) -> chex.Array:
        """Given a cell, return a jnp.chex.Array of size 4 with the flat indices of
        adjacent cells. Padded with -1's if less than 4 adjacent cells (if on the edge of the grid).

        Args:
            cell: the flat index of the cell to find adjacent cells of.

        Returns:
            A jnp.chex.Array of size 4 with the flat indices of adjacent cells
            (padded with -1's if less than 4 adjacent cells).
        """
        available_moves = jnp.full(4, cell)
        direction_operations = jnp.array([-1 * self._rows, self._rows, -1, 1])
        # Create a mask to check 0 <= index < total size
        cells_to_check = available_moves + direction_operations
        is_id_in_grid = cells_to_check < self._rows * self._cols
        is_id_positive = 0 <= cells_to_check
        mask = is_id_positive & is_id_in_grid

        # Ensure adjacent cells doesn't involve going off the grid
        unflatten_available = jnp.divmod(cells_to_check, self._rows)
        unflatten_current = jnp.divmod(cell, self._rows)
        is_same_row = unflatten_available[0] == unflatten_current[0]
        is_same_col = unflatten_available[1] == unflatten_current[1]
        row_col_mask = is_same_row | is_same_col
        # Combine the two masks
        mask = mask & row_col_mask
        return jnp.where(mask == 0, -1, cells_to_check)

    def available_cells(self, grid: chex.Array, cell: int) -> chex.Array:
        """ Given a cell and the grid of the board, see which adjacent cells are available to move to
        (i.e. are currently unoccupied).
        TODO: Expand this to also check that cells do not touch the current wire more than once,
        to improve quality of generated boards.

        Args:
            grid: the current grid of the board.
            cell: the flat index of the cell to find adjacent cells of.

        Returns:
            A jnp.chex.Array of size 4 with the flat indices of adjacent cells.
        """
        adjacent_cells = self.adjacent_cells(cell)
        # Get the wire id of the current cell
        value = grid[jnp.divmod(cell, self._rows)]
        wire_id = (value - 1) // 3

        _, available_cells_mask = jax.lax.scan(self.is_cell_free, grid, adjacent_cells)
        # Also want to check if the cell is touching itself more than once
        _, touching_cells_mask = jax.lax.scan(self.is_cell_touching_self, (grid, wire_id), adjacent_cells)
        available_cells_mask = available_cells_mask & touching_cells_mask
        available_cells = jnp.where(available_cells_mask == 0, -1, adjacent_cells)
        return jnp.hstack((available_cells,
                           jnp.full(self._rows - len(available_cells) + 1,
                                    -1)))

    def is_cell_free(
            self, grid: chex.Array, cell: int
        ) -> Tuple[chex.Array, bool]:
        """Check if a given cell is free, i.e. has a value of 0.

        Args:
            grid: the current grid of the board.
            cell: the flat index of the cell to check.

        Returns:
            A tuple of the new grid and a boolean indicating whether the cell is free or not.
        """
        coordinate = jnp.divmod(cell, self._rows)
        return grid, jax.lax.select(cell == -1, False, grid[coordinate[0], coordinate[1]] == 0)
    
    def is_cell_touching_self(
            self, grid_wire_id: Tuple[chex.Array, int], cell: int,
        ) -> Tuple[Tuple[chex.Array, int], bool]: 
        """Check if the cell is touching any of the wire's own cells more than once.
        This means looking for surrounding cells of value 3 * wire_id + POSITION or
        3 * wire_id + PATH.
        """
        grid, wire_id = grid_wire_id
        # Get the adjacent cells of the current cell
        adjacent_cells = self.adjacent_cells(cell)
        def is_cell_touching_self_inner(grid, cell):
            coordinate = jnp.divmod(cell, self._rows)
            cell_value = grid[coordinate[0], coordinate[1]]
            touching_self = jnp.logical_or(jnp.logical_or(cell_value == 3 * wire_id + POSITION, cell_value == 3 * wire_id + PATH), cell_value == 3 * wire_id + TARGET)
            return grid, jnp.where(cell == -1, False, touching_self)

        # Count the number of adjacent cells with the same wire id
        _, touching_self_mask = jax.lax.scan(is_cell_touching_self_inner, grid, adjacent_cells)
        # If the cell is touching itself more than once, return False
        return (grid, wire_id), jnp.where(jnp.sum(touching_self_mask) > 1, False, True)


    def one_step(
            self, random_walk_tuple: Tuple[chex.PRNGKey, chex.Array, Wire]
        ) -> Tuple[chex.PRNGKey, chex.Array, Wire]:
        """Have a single agent take a single random step on the board.

        Args:
            random_walk_tuple: a tuple of the key, the grid, the current wire, and the wire id.

        Returns:
            A tuple of the key, the new grid, the new wire, and the wire id.
        """
        key, grid, wire, wire_id = random_walk_tuple
        key, subkey = jax.random.split(key)
        cell = wire.path[wire.insertion_index - 1][0] * self._cols + wire.path[wire.insertion_index - 1][1]
        available_cells = self.available_cells(grid=grid, cell=cell)
        step_coordinate_flat = jax.random.choice(
            key=subkey,
            a=available_cells,
            shape=(),
            replace=False,
            p=available_cells != -1,
        )
        coordinate = jnp.divmod(step_coordinate_flat, self._rows)
        new_wire = stack_push(wire, coordinate)
        # Add new coordinate
        grid = grid.at[coordinate[0], coordinate[1]].set(3 * wire_id + TARGET)
        # Change old coordinate to be part of wire
        grid = grid.at[wire.path[wire.insertion_index - 1][0], wire.path[wire.insertion_index - 1][1]].set(3 * wire_id + PATH)
        return key, grid, new_wire, wire_id
    
    def can_step(
            self, random_walk_tuple: Tuple[chex.PRNGKey, chex.Array, Wire]
        ) -> bool:
        """Check that a given wire can take a step on the board (i.e. it has an available cell to step to).

        Args:
            random_walk_tuple: a tuple of the key, the grid, the current wire, and the wire id.

        Returns:
            A boolean indicating whether the wire can take a step or not.
        """
        key, grid, wire, _ = random_walk_tuple
        cell = wire.path[wire.insertion_index - 1][0] * self._cols + wire.path[wire.insertion_index - 1][1]
        available_cells = self.available_cells(grid=grid, cell=cell)
        return jnp.any(available_cells != -1)

    def walk_randomly(
            self, random_walk_tuple: Tuple[chex.PRNGKey, chex.Array, Wire, int, int]
        ) -> Tuple[Tuple[chex.PRNGKey, chex.Array, Wire, int], bool]:
        """Perform the entire random walk for a single agent on the board.

        Args:
            random_walk_tuple: a tuple of the key, the grid, the current wire, the wire id, and the max length.

        Returns:
            A tuple of the key, the new grid, the new wire, and the wire id.
            A boolean indicating whether the wire was able to move or not.
        """
        max_length = random_walk_tuple[4]
        random_walk_tuple = random_walk_tuple[:4]
        moved = False

        def single_step(
                _, carry: Tuple[Tuple[chex.PRNGKey, chex.Array, Wire, int], bool]
            ) -> Tuple[Tuple[chex.PRNGKey, chex.Array, Wire, int], bool]:
            """The body of the loop. Works by checking the agent is able to step, and then stepping if so.
            Also tracks whether the agent was able to move or not.

            Args:
                carry: the current state of the loop.

            Returns:
                The new state of the board tuple, and a boolean indicating whether the agent was able to move or not.
            """
            random_walk_tuple, moved = carry
            can_step = self.can_step(random_walk_tuple)
            moved = moved | can_step
            random_walk_tuple = jax.lax.cond(can_step, self.one_step, lambda x: x, random_walk_tuple)
            return (random_walk_tuple, moved)

        random_walk_tuple, moved = jax.lax.fori_loop(0, max_length, single_step, (random_walk_tuple, moved))

        # Afterwards, change the first cell to be a head
        key, grid, wire, wire_id = random_walk_tuple
        grid = grid.at[wire.path[0][0], wire.path[0][1]].set(3 * wire_id + POSITION)
        random_walk_tuple = (key, grid, wire, wire_id, max_length)
        return random_walk_tuple, moved

    def add_agents(
            self, key: chex.PRNGKey, board: chex.Array, max_length: int
        ) -> Tuple[chex.Array, bool]:
        """Try to add all required agents to the board. This works by 
        adding each agent sequentially.

        Args:
            key: the jax key to use for random number generation.
            board: the current board grid.
            max_length: the maximum length of the random walk of each agent.

        Returns:
            board: jnp.chex.Array with wires added.
            success (bool): tracks whether n_agents were added to the board.
        """
        random_walk_tuple = (key, board, 0, 0, max_length)
        success = True
        for wire_id in range(self._num_agents):
            key, board, _, _, _  = random_walk_tuple
            key, subkey = jax.random.split(key)
            # Start the wire at a random location (if possible)
            random_walk_tuple, can_start = self.pick_start(subkey, board, wire_id, max_length)
            # If possible, try to walk randomly 
            random_walk_tuple, moved = jax.lax.cond(can_start, self.walk_randomly, lambda x: (x, False), random_walk_tuple)
            # Track if we have successfully started and then moved at least once.
            success = can_start & moved & success
        return random_walk_tuple[1], success

    def generate(
            self, key: chex.PRNGKey
        ) -> chex.Array:
        """Generates a board using sequential random walk, with all wires still present.
        Works by first trying to generate a board with the longest wires possible, repeating with a smaller
        maximum possible length of wire until generation is succesful.

        Args:
            key: the jax key to use for random number generation.
        
        Returns:
            board: jnp.chex.Array with wires added (or a blank board if generation failed).
        """
        def try_to_generate(
                self, max_length_int: int, i: int, state: Tuple[chex.Array, bool]
            ) -> Tuple[Tuple[chex.Array, bool], bool]:
            """Main loop. Checks whether a board has been successfully generated, tries to generate a new board if it hasn't.
            
            Args:
                max_length_int: the maximum length of the random walk of each agent.
                i: the current iteration of the loop.
                state: the current state of the loop.

            Returns:
                A tuple of the new board, and a boolean indicating whether the board was successfully generated.
            """
            board, success = state

            def generate_new_board() -> Tuple[Tuple[chex.Array, bool], bool]:
                """Tries to generate an entire board, starting from an empty board."""
                empty_board = jnp.zeros((self._rows, self._cols))
                new_board, new_success = self.add_agents(key, empty_board, max_length_int - i)
                return (new_board, new_success)

            def keep_successful_board() -> Tuple[Tuple[chex.Array, bool], bool]:
                """Returns the current board and success boolean."""
                return (board, success)

            # Call self.add_agents only if success is False
            updated_board, updated_success = jax.lax.cond(
                jnp.logical_not(success),
                generate_new_board,
                keep_successful_board
            )
            return (updated_board, updated_success)
        
        # Set the initial maximum length of path to try.
        # TODO: experiment with different ways of setting this.
        start_max_length = self._rows + self._cols
        max_length_int = jax.lax.convert_element_type(start_max_length, jnp.int32).astype(int)

        board = jnp.zeros((self._rows, self._cols))
        init_state = (board, False)

        # Create a partial function with the first two arguments fixed
        body_fun_partial = partial(try_to_generate, self, max_length_int)

        final_board, success = jax.lax.fori_loop(1, max_length_int + 1, body_fun_partial, init_state)

        def return_board() -> chex.Array:
            """Returns the final board, assuming generation was successful."""
            return final_board
        
        # TODO: check how to handle what happens if generation is unsuccesful.        
        def return_empty_board() -> chex.Array:
            """Returns an empty board, assuming generation was unsuccessful."""
            return jnp.zeros((self._rows, self._cols))

        return jax.lax.cond(success, return_board, return_empty_board)
    
    def generate_starts_ends(
            self, key: chex.PRNGKey
        ) -> Tuple[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array]]:
        """Call generate, take the first and last cells of each wire.
        Returns these cells formatted as required by the training process, i.e. as
        two tuples of dimension num_agents.

        Args:
            key: the jax key to use for random number generation.

        Returns:
            starts: tuple of arrays, first array is x coords, second is y coords.
            ends: tuple of arrays, first array is x coords, second is y coords.
        """
        board = self.generate(key)

        def find_positions(wire_id):
            wire_positions = board == 3 * wire_id + POSITION
            wire_targets = board == 3 * wire_id + TARGET

            # Compute indices where wire_positions and wire_targets are True
            start_indices = jnp.argwhere(wire_positions, size=2)
            end_indices = jnp.argwhere(wire_targets, size=2)
            start = start_indices[0]
            end = end_indices[0]
            return start, end

        wire_ids = jnp.arange(self._num_agents)
        starts_ends = jax.vmap(find_positions)(wire_ids)
        starts, ends = starts_ends[0], starts_ends[1]

        # For generation purposes, we want starts to be a tuple of arrays, first array is x coords, second is y coords
        starts = (starts[:, 0], starts[:, 1])
        # Likewise for ends
        ends = (ends[:, 0], ends[:, 1])

        return starts, ends


if __name__ == "__main__":
    board_generator = SequentialRandomWalk(100, 100, 10)
    key = jax.random.PRNGKey(42)
    # jit generate
    board_generator_jit = jax.jit(board_generator.generate)
    print(board_generator_jit(key))
    #print(board_generator.generate(key))

    # jit generate_starts_ends
    board_generator_starts_ends_jit = jax.jit(board_generator.generate_starts_ends)
    print(board_generator_starts_ends_jit(key))


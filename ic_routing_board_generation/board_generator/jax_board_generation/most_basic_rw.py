from dataclasses import dataclass
from functools import partial

from chex import PRNGKey, Array

#from ic_routing_board_generation.board_generator.abstract_board import AbstractBoard
import numpy as np
from copy import deepcopy
# import random
from typing import List, Tuple
import jax.numpy as jnp
import jax

from ic_routing_board_generation.board_generator.jax_board_generation.grid_jax import Grid
from ic_routing_board_generation.board_generator.jax_data_model.wire import Wire, create_wire, \
    stack_push

"""
#from jax.numpy import asarray # Currently jaxlib is not supported on windows.  This will have to be sorted.
#from env_viewer import RoutingViewer # Currently jaxlib is not supported on windows.  This will have to be sorted.
# import jumanji.environments.combinatorial.routing.constants
#from jumanji.environments.combinatorial.routing.constants import TARGET, HEAD, EMPTY
#from jumanji.environments.combinatorial.routing.constants import SOURCE as WIRE
"""
EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3
#  HEAD, TARGET, WIRE, EMPTY = 4,3,2,0
STARTING_POSITION = POSITION  # My internal variable to disambiguate the word "position"
# Also available to import from constants NOOP, LEFT, LEFT, UP, RIGHT, DOWN


@dataclass
class Position:
    """  Class of 2D tuple of ints, indicating the size of an array or a 2D position or vector."""
    x: int
    y: int

class JaxRandomWalk:
    def __init__(self, rows: int, cols: int, num_agents: int = 3):
        self._rows = rows
        self._cols = cols
        self._num_agents = num_agents

    def return_blank_board(self) -> Array:
        return jnp.zeros((self._rows, self._cols), dtype=int)

    def pick_start(self, key: PRNGKey, layout: Array, wire_id: int, max_length: int) -> Wire:
        """Create a wire and populate start and end points randomly.
        
        Args:
            key: jax random key
            layout: the current layout of the board
            wire_id: the id of the wire to be created
            max_length: the maximum length of the wire to be created
        Returns:
            A tuple of the new key, the new layout, the new wire, the wire id, and the max length
            can_start: a boolean indicating whether we can place a wire on the board or not

        """
        key, subkey = jax.random.split(key)
        grid_size = len(layout) * len(layout[0])
        # See if there is any empty space in the grid
        can_start = jnp.any(layout == 0)
        def inner_create_wire(key, layout=layout, wire_id=wire_id):
            # Pick a random coordinate which is empty (i.e. 0)
            coordinate_flat = jax.random.choice(
                key=key,
                a=jnp.arange(grid_size),
                shape=(),
                replace=False,
                p=jnp.where(layout.flatten() == 0, 1, 0),
            )

            # Create a 2D coordinate from the flat array.
            coordinate = jnp.divmod(coordinate_flat, self._rows)
            new_wire = create_wire(grid_size, coordinate, (-1, -1), wire_id)
            new_wire = stack_push(new_wire, coordinate)
            layout = layout.at[coordinate[0], coordinate[1]].set(3 * wire_id + POSITION)
        
            return new_wire, layout
        # Use jax.lax.cond to only run the inner function if there is empty space
        dummy_wire = create_wire(grid_size, (-1, -1), (-1, -1), wire_id)
        new_wire, layout = jax.lax.cond(can_start, inner_create_wire, lambda x: (dummy_wire, layout), key)
        return (key, layout, new_wire, wire_id, max_length), can_start

    def adjacent_cells(self, cell: int) -> Array:
        """
        Given a cell, return a jnp.Array of size 4 with the flat indices of
        adjacent cells. Padded with -1's if less than 4 adjacent cells (if on the edge of the grid)

        Args:
            cell: the flat index of the cell to find adjacent cells for
        Returns:
            A jnp.Array of size 4 with the flat indices of adjacent cells
            (padded with -1's if less than 4 adjacent cells)
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

    def available_cells(self, layout: Array, cell: int):
        """
        Given a cell and the layout of the board, see which adjacent cells are available to move to
        (i.e. are currently unoccupied).
        TODO: expand this to also check that cells do not touch the current wire more than once,
        to improve quality of generated boards.

        Args:
            layout: the current layout of the board
            cell: the flat index of the cell to find adjacent cells for
        Returns:
            A jnp.Array of size 4 with the flat indices of adjacent cells
        """
        adjacent_cells = self.adjacent_cells(cell)
        _, available_cells_mask = jax.lax.scan(self.is_cell_free, layout, adjacent_cells)
        available_cells = jnp.where(available_cells_mask == 0, -1, adjacent_cells)
        return jnp.hstack((available_cells,
                           jnp.full(self._rows - len(available_cells) + 1,
                                    -1)))

    def is_cell_free(self, layout: Array, cell: int):
        """
        Check if a given cell is free, i.e. has a value of 0.

        Args:
            layout: the current layout of the board
            cell: the flat index of the cell to check
        Returns:
            A tuple of the new layout and a boolean indicating whether the cell is free or not
        """
        coordinate = jnp.divmod(cell, self._rows)
        return layout, jax.lax.select(cell == -1, False, layout[coordinate[0], coordinate[1]] == 0)

    def one_step(self, random_walk_tuple: Tuple[PRNGKey, Array, Wire]):
        """
        Have a single agent take a single random step on the board.

        Args:
            random_walk_tuple: a tuple of the key, the layout, the current wire, and the wire id
        Returns:
            A tuple of the key, the new layout, the new wire, and the wire id
        """
        key, layout, wire, wire_id = random_walk_tuple
        key, subkey = jax.random.split(key)
        cell = wire.path[wire.insertion_index - 1][0] * self._cols + wire.path[wire.insertion_index - 1][1]
        available_cells = self.available_cells(layout=layout, cell=cell)
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
        layout = layout.at[coordinate[0], coordinate[1]].set(3 * wire_id + TARGET)
        # Change old coordinate to be part of wire, TODO: if not a head
        layout = layout.at[wire.path[wire.insertion_index - 1][0], wire.path[wire.insertion_index - 1][1]].set(3 * wire_id + PATH)
        return key, layout, new_wire, wire_id
    
    def can_step(self, random_walk_tuple: Tuple[PRNGKey, Array, Wire]):
        """
        Check that a given wire can take a step on the board (i.e. it has an available cell to step to)

        Args:
            random_walk_tuple: a tuple of the key, the layout, the current wire, and the wire id
        Returns:
            A boolean indicating whether the wire can take a step or not
        """
        key, layout, wire, _ = random_walk_tuple
        cell = wire.path[wire.insertion_index - 1][0] * self._cols + wire.path[wire.insertion_index - 1][1]
        available_cells = self.available_cells(layout=layout, cell=cell)
        return jnp.any(available_cells != -1)

    def walk_randomly(self, random_walk_tuple: Tuple[PRNGKey, Array, Wire, int, int]):
        """
        Perform the entire random walk for a single agent on the board.

        Args:
            random_walk_tuple: a tuple of the key, the layout, the current wire, the wire id, and the max length

        Returns:
            A tuple of the key, the new layout, the new wire, and the wire id
            A boolean indicating whether the wire was able to move or not
        """
        max_length = random_walk_tuple[4]
        random_walk_tuple = random_walk_tuple[:4]
        moved = False

        def body_fun(i, carry):
            """
            The body of the loop. Works by checking the agent is able to step, and then stepping if so.
            Also tracks whether the agent was able to move or not.

            Args:
                i: the current iteration of the loop
                carry: the current state of the loop
            Returns:
                The new state of the board tuple, and a boolean indicating whether the agent was able to move or not
            """
            random_walk_tuple, moved = carry
            can_step = self.can_step(random_walk_tuple)
            moved = moved | can_step
            random_walk_tuple = jax.lax.cond(can_step, self.one_step, lambda x: x, random_walk_tuple)
            return (random_walk_tuple, moved)

        random_walk_tuple, moved = jax.lax.fori_loop(0, max_length, body_fun, (random_walk_tuple, moved))

        # Afterwards, change the first cell to be a head
        key, layout, wire, wire_id = random_walk_tuple
        layout = layout.at[wire.path[0][0], wire.path[0][1]].set(3 * wire_id + POSITION)
        random_walk_tuple = (key, layout, wire, wire_id, max_length)
        return random_walk_tuple, moved

    def add_agents(self, key: PRNGKey, board: Array, max_length: int):
        """
        Try to add all required agents to the board. This works by 
        adding each agent sequentially.

        Args:
            key: the jax key to use for random number generation
            board: the current board layout
            max_length: the maximum length of the random walk of each agent

        Returns:
            board: jnp.Array with wires added.
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

    def generate(self, key):
        """
        Generates a board using sequential random walk, with all wires still present.
        Works by first trying to generate a board with the longest wires possible, repeating with a smaller
        maximum possible length of wire until generation is succesful.

        Args:
            key: the jax key to use for random number generation
        
        Returns:
            board: jnp.Array with wires added (or a blank board if generation failed)
        """
        def body_fun(self, max_length_int, i, state):
            """
            Main loop. Checks whether a board has been successfully generated, generates a new board if it hasn't.
            """
            board, success = state

            def update_board_and_success(_):
                empty_board = jnp.zeros((self._rows, self._cols))
                new_board, new_success = self.add_agents(key, empty_board, max_length_int - i)
                return (new_board, new_success)

            def keep_current_state(_):
                return (board, success)

            # Call self.add_agents only if success is False
            updated_board, updated_success = jax.lax.cond(
                jnp.logical_not(success),
                None,
                update_board_and_success,
                None,
                keep_current_state
            )

            def true_fun(_):
                return (updated_board, updated_success)

            def false_fun(_):
                return (updated_board, False)

            return jax.lax.cond(updated_success, None, true_fun, None, false_fun)
        
        # Set the initial maximum length of path to try. 
        start_max_length = self._rows * self._cols // self._num_agents
        max_length_int = jax.lax.convert_element_type(start_max_length, jnp.int32).astype(int)

        board = jnp.zeros((self._rows, self._cols))
        init_state = (board, False)

        # Create a partial function with the first two arguments fixed
        body_fun_partial = partial(body_fun, self, max_length_int)

        final_board, success = jax.lax.fori_loop(1, max_length_int + 1, body_fun_partial, init_state)

        # Use jax.lax.cond to return the successfully generated board if possible,
        # or an empty board if the generation was unsuccessful.
        # TODO: handle what happens if generation is unsuccesful.
        def true_fun(_):
            return final_board

        def false_fun(_):
            return jnp.zeros((self._rows, self._cols))

        return jax.lax.cond(success, None, true_fun, None, false_fun)
    
    def generate_starts_ends(self, key):
        """
        Call generate, take the first and last cells of each wire.

        Return these cells formatted as required by the training process, i.e. as
        two tuples of dimension num_agents

        Args:
            key: the jax key to use for random number generation

        Returns:
            starts: tuple of arrays, first array is x coords, second is y coords
            ends: tuple of arrays, first array is x coords, second is y coords
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
    board_generator = JaxRandomWalk(10, 10, 3)
    key = jax.random.PRNGKey(101)
    # jit generate
    board_generator_jit = jax.jit(board_generator.generate)
    print(board_generator_jit(key))
    #print(board_generator.generate(key))

    # jit generate_starts_ends
    board_generator_starts_ends_jit = jax.jit(board_generator.generate_starts_ends)
    print(board_generator_starts_ends_jit(key))


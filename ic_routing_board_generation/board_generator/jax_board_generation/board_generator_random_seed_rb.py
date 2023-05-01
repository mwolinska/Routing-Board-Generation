from ic_routing_board_generation.board_generator.numpy_data_model.abstract_board import AbstractBoard
from ic_routing_board_generation.board_generator.jax_utils.grid_utils import optimise_wire
from ic_routing_board_generation.board_generator.jax_utils.post_processor_utils_jax import extend_wires_jax, \
    training_board_from_solved_board_jax
#import numpy as np

from jax import Array
from jax.random import PRNGKey
import jax.numpy as jnp
import jax
from copy import deepcopy

from jumanji.environments.routing.connector.constants import EMPTY, PATH, POSITION, TARGET
STARTING_POSITION = POSITION  # Resolve ambiguity of POSITION constant
INVALID = jnp.array((-999, -999))  # Encoding for an invalid 2D position for lists that must be of constant length
INVALID4 = jnp.array([(-999, -999), (-999, -999), (-999, -999), (-999, -999)])  # Encoding for list of four invalid positions

#@jax.disable_jit()
#@jax.jit
class RandomSeedBoard(AbstractBoard):
    """ The boards are 2D arrays of wiring routed on a printed circuit board.

    The encoding of the board cells is as follows:
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
        self._num_agents = num_agents  # Number of wires requested
        # Limit the number of wires to number_cells/3 to ensure we can fit them all
        max_num = jax.lax.select(num_agents > rows*cols//3, jnp.array(rows*cols//3, int), num_agents)
        self._wires_on_board = max_num  # Actual number of wires on the board


    #@jax.disable_jit()
    #@jax.jit
    def return_seeded_board_danila(self, key: PRNGKey) -> Array:
        """ Generate and return an array of the board with the connecting wires encoded.

        Args:
            key (PRNGKey) : Random number generator key

        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """

        def generate_indices_one_side():
            """Create a list of agents with random initial starting and ending (as neighbours) locations."""
            row_indices, col_indices = jax.numpy.meshgrid(jax.numpy.arange(1, self._rows, 2),
                                                          jax.numpy.arange(1, self._cols, 2),
                                                          indexing='ij')
            index_choice = jax.numpy.stack((row_indices, col_indices), axis=-1).reshape(-1, 2)
            head_indices = jax.random.choice(key, index_choice, (self._num_agents,), replace=False)
            offset_array = jax.numpy.array([[-1, 0], [0, -1]])
            tail_offsets = jax.random.choice(key, offset_array, (self._num_agents,))
            tail_indices = head_indices + tail_offsets
            return head_indices, tail_indices

        def generate_indices_other_side():
            """Create a list of agents with random initial starting and ending (as neighbours) locations."""
            row_indices, col_indices = jax.numpy.meshgrid(jax.numpy.arange(0, self._rows - 1, 2),
                                                          jax.numpy.arange(0, self._cols - 1, 2),
                                                          indexing='ij')
            index_choice = jax.numpy.stack((row_indices, col_indices), axis=-1).reshape(-1, 2)
            head_indices = jax.random.choice(key, index_choice, (self._num_agents,), replace=False)
            offset_array = jax.numpy.array([[0, 1], [1, 0]])
            tail_offsets = jax.random.choice(key, offset_array, (self._num_agents,))
            tail_indices = head_indices + tail_offsets
            return head_indices, tail_indices

        xs, ys = jax.lax.cond(jax.random.randint(key, (), 0, 2), generate_indices_one_side,
                              generate_indices_other_side)
        board = jax.numpy.zeros((self.grid_size, self.grid_size), int)
        board = board.at[(xs[:, 0], xs[:, 1])].set(jnp.arange(len(xs)) * 3 + TARGET)
        board = board.at[(ys[:, 0], ys[:, 1])].set(jnp.arange(len(ys)) * 3 + POSITION)
        return board


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


    def return_solved_board(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True,
                            extension_iterations: int = 1, extension_steps: int = 1e23) -> Array:
        """ Generate and return an array of the board with the connecting wires zeroed out.

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization. (Default = 1)
            extension_steps (int): Max number of extension loops to perform during each iteration.  (Default = no limit)
                                Note that sometimes wires might extend one cell or multiple cells per steps.

        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """
        self.randomness = randomness
        self.two_sided = two_sided
        self.extension_iterations = extension_iterations
        self.extension_steps = extension_steps

        key, seedkey = jax.random.split(key)
        board_layout = self.return_seeded_board_danila(seedkey)
        # print("SEEDED BOARD")
        # print(board_layout)
        key, extkey = jax.random.split(key, 2)
        # Always do one iteration of extension, but it can be of zero length if desired
        if extension_iterations > 0:
            board_layout = self.extend_wires_jax(board_layout, extkey, randomness, two_sided, extension_steps)

        if extension_iterations > 1:
            def optim_extend_func(board_layout, key):
                key, extkey, optkey = jax.random.split(key, 3)
                optkeys = jax.random.split(optkey, self._wires_on_board)
                # Optimise each wire individually
                def optimise_wire_loop_func(wire_num, carry):
                    board_layout, keys = carry
                    board_layout = optimise_wire(keys[wire_num], board_layout, wire_num)
                    carry = (board_layout, keys)
                    return carry
                carry = (board_layout, optkeys)
                board_layout, _ = jax.lax.fori_loop(0, self._wires_on_board, optimise_wire_loop_func, carry)
                # Extend all the wires
                board_layout = self.extend_wires_jax(board_layout, extkey, randomness, two_sided, extension_steps)
                # print(board_layout)
                return board_layout, None

            keys = jax.random.split(key, extension_iterations-1)
            board_layout, _ = jax.lax.scan(optim_extend_func, board_layout, keys)
        return board_layout


    #@jax.jit
    def return_training_board(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True,
                              extension_iterations: int = 1, extension_steps: int = 1e23) -> Array:
        """ Generate and return an array of the board with the connecting wires zeroed out.

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization.
            extension_steps (int): Max number of extension loops to perform during each iteration.
                                Note that sometimes wires may extend multiple cells per extension loop.
        Returns:
            (Array) : 2D layout of the board with all wirings encoded
        """
        board_layout = self.return_solved_board(key, randomness, two_sided, extension_iterations, extension_steps)
        training_board = training_board_from_solved_board_jax(board_layout)
        return training_board

    #@jax.jit
    def generate_starts_ends(self, key: PRNGKey, randomness: float = 0.0, two_sided: bool = True,
                             extension_iterations: int = 1, extension_steps: int = 1e23) -> Array:
        """  Call generate, take the first and last cells of each wire

        Args:
            key (PRNGKey) : Random number generator key
            randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
            two_sided (bool): True => Wire extension extends both heads and targets.  False => Only targets.
            extension_iterations (int) : Number of iterations of wire-extension/BFS optimization.
            extension_steps (int): Max number of extension loops to perform during each iteration.
                                Note that sometimes wires may extend multiple cells per extension loop.
        Returns:
            (Array) : List of starting points, dimension 2 x num_agents
            (Array) : List of target points, dimension 2 x num_agents
        """
        board_layout = self.return_solved_board(key, randomness, two_sided, extension_iterations, extension_steps)

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
        #jax.debug.print("{x}", x=starts_ends)
        #jax.debug.print("{x}", x=type(starts_ends))
        starts, ends = starts_ends[0], starts_ends[1]

        # Want starts to be a tuple of arrays, first array is x coords, second is y coords
        starts = (starts[:, 0], starts[:, 1])
        ends = (ends[:, 0], ends[:, 1])
        return starts, ends




    #@jax.disable_jit()
    #@jax.jit
    def extend_wires_jax(self, board_layout: Array, key: PRNGKey, randomness: float = 0.0,
                         two_sided: bool = True, extension_steps: int = 1e23) -> Array:
        """ Extend the heads and targets of each wire as far as they can go, preference given to current direction.
            The implementation is done in-place on self.board_layout

            Args:
                board_layout (Array): 2D layout of the encoded board (before wire extension)
                key( PRNGKey): a random key to use for randomly selecting directions (as needed)
                randomness (float): How randomly to extend the wires, 0=>Keep same direction if possible, 1=>Random
                two_sided (bool): True => £xtend both heads and targets.  False => Only extend targets.

            Returns:
                (Array): 2D layout of the encoded board (after wire extension)
        """
        rows, cols = self._rows, self._cols
        # Assuming a valid board, we find the number of wires from the value of the TARGET cell of the last wire
        num_wires = self._wires_on_board
        num_extendables = jax.lax.select(two_sided, 2 * num_wires, num_wires)
        prev_layout = board_layout.at[0, 0].add(1)  # Make prev_layout != board_layout
        # Continue as long as the algorithm is still changing the board
        step_num = 0
        def while_cond(carry):
            prev_layout, board_layout, _, step_num = carry
            return ~jnp.array_equal(prev_layout, board_layout) & (step_num < extension_steps)

        def while_body(carry):
            prev_layout, board_layout, key, step_num = carry
            step_num = step_num + 1

            # Randomly flip/flop the board between iterations to remove any directional bias.
            key, flipkey, flopkey = jax.random.split(key, 3)
            do_flip = jax.random.choice(flipkey, jnp.array([True, False]))
            do_flop = jax.random.choice(flopkey, jnp.array([True, False]))
            board_layout = jax.lax.select(do_flip, board_layout[::-1, :], board_layout)
            board_layout = jax.lax.select(do_flop, board_layout[:, ::-1], board_layout)

            # Make a copy of the board_layout to check if it changes
            prev_layout = deepcopy(board_layout)





            """  THIS CHUNK OF CODE IS FOR FURTHER OPTIMISATION FOR CLEMENT THAT IS IN PROGRESS AS OF 1 MAY, 2023
            # JUST SAMPLE THE EXTENDABLES
            targets_bool = (((board_layout // 3) == TARGET) & (board_layout != 0))
            targets_pos = jnp.argwhere(targets_bool, size=num_extendables, fill_value=(-999, -999))
            print("targets=", targets_pos)
            starts_targets_bool = (((board_layout // 3) != PATH) & (board_layout != 0))
            starts_targets_pos = jnp.argwhere(starts_targets_bool, size=num_extendables, fill_value=(-999, -999))
            print("starts_targets=", starts_targets_pos)
            if two_sided:
                extendables_pos = starts_targets_pos
            else:
                extendables_pos = targets_pos
    
            #extendables_pos = jax.lax.select(two_sided, starts_targets_pos, targets_pos)
            print("extendables_pos=", extendables_pos)
            """






            def row_func(row, carry):
                board_layout, key = carry
                def col_func(col, carry):
                    board_layout, key = carry

                    # Get the list of neighbors available to extend to.
                    current_pos = jnp.array((row, col))
                    poss_extension_list = self.get_open_adjacent_cells_jax(current_pos, board_layout)

                    current_wire_num = self.position_to_wire_num_jax(current_pos, board_layout)
                    # For each possible cell, mark it INVALID if it already touches part of the same wire.
                    def invalidate_func(cell_pos, current_wire_num, board_layout):
                        num_adjacencies = self.num_wire_adjacencies_jax(cell_pos, current_wire_num, board_layout)
                        cell_or_invalid = jax.lax.select(num_adjacencies > 1, INVALID, cell_pos)
                        return cell_or_invalid
                    poss_extension_list = jax.vmap(invalidate_func, in_axes=(0, None, None))(poss_extension_list, current_wire_num, board_layout)

                    # Get the position of the wire that led into this cell and prioritize continuing the same direction
                    prev_neighbor = self.get_previous_neighbor_jax(current_pos, board_layout)
                    priority_neighbor = (row + (row-prev_neighbor[0]), col + (col-prev_neighbor[1]))
                    priority_neighbor = jnp.array(priority_neighbor)

                    # If the current cell is not a head or target, it's not extendable,
                    # so change everything in the list to INVALID
                    cell_type = self.position_to_cell_type_jax(current_pos, board_layout)
                    is_extendable = (two_sided & ((cell_type == STARTING_POSITION) | (cell_type == TARGET))) | \
                                    (~two_sided & (cell_type == TARGET))
                    poss_extension_list = jax.lax.select(is_extendable, poss_extension_list, INVALID4)

                    # If there are no valid options to extend to, stop extending
                    stop_extension = jnp.array_equal(poss_extension_list, INVALID4)
                    poss_extension_list = jax.lax.select(stop_extension,
                                                         jnp.array([current_pos, current_pos, current_pos, current_pos]),
                                                         poss_extension_list)

                    # Pick a random selection from the possible extensions until we pick a valid one.
                    def choice_body_func(carry):
                        poss_extension_list, extension_pos, key = carry
                        key, choice_key = jax.random.split(key)
                        extension_pos = jax.random.choice(choice_key, poss_extension_list)
                        carry = (poss_extension_list, extension_pos, key)
                        return carry

                    def choice_cond_func(carry):
                        _, extension_pos, _ = carry
                        return jnp.array_equal(extension_pos, INVALID)

                    # Is the prioity neighbor available?
                    extension_pos = INVALID
                    _, extension_pos, _ = jax.lax.while_loop(choice_cond_func, choice_body_func,
                                                             (poss_extension_list, extension_pos, key))

                    def priority_found_func(cell_pos):
                        item_match = jnp.array_equal(cell_pos, priority_neighbor)
                        return item_match
                    priority_neighbor_available = jnp.any(jax.vmap(priority_found_func)(poss_extension_list))

                    # Use random choice if we are in random mode.
                    # Otherwise rioritize extending away from the previous neighbor if possible, random if not available.
                    key, random_key = jax.random.split(key)
                    use_random = (randomness > jax.random.uniform(random_key))
                    extension_pos = jax.lax.select(priority_neighbor_available & ~use_random,
                                                   priority_neighbor, extension_pos)

                    # Extend head/target into new cell
                    board_layout = board_layout.at[extension_pos[0], extension_pos[1]]\
                                                .set(board_layout[current_pos[0], current_pos[1]])
                    cell_type = self.position_to_cell_type_jax(current_pos, board_layout)
                    # Convert old head/target cell to a wire (unless stop_extension is True)
                    cell_type_offset = jax.lax.select(stop_extension, 0, PATH - cell_type)
                    board_layout = board_layout.at[current_pos[0], current_pos[1]].add(cell_type_offset)


                    return board_layout, key
                # THESE FORI LOOPS DON'T IMPROVE COMPILATION TIME MUCH COMPARED TO NESTED FOR LOOPS
                carry = (board_layout, key)
                board_layout, key = jax.lax.fori_loop(0, cols, col_func, carry)
                return board_layout, key
            carry = (board_layout, key)
            board_layout, key = jax.lax.fori_loop(0, rows, row_func, carry)

            # Undo random flip/flopping to compare to prev_layout
            board_layout = jax.lax.select(do_flip, board_layout[::-1, :], board_layout)
            board_layout = jax.lax.select(do_flop, board_layout[:, ::-1], board_layout)
            # For jax.lax.while_loop
            carry = (prev_layout, board_layout, key, step_num)
            return carry

        # For jax.lax.while_loop
        carry = (prev_layout, board_layout, key, step_num)
        _, board_layout, _, _ = jax.lax.while_loop(while_cond, while_body, carry)
        return board_layout


    #@jax.jit
    # This method is used by the extend_wires_jax method
    def get_previous_neighbor_jax(self, input_pos: jnp.array, board_layout: Array) -> jnp.array:
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
        input_wire_num = self.position_to_wire_num_jax(input_pos, board_layout)
        neighbors_positions = jnp.array([(row-1, col), (row+1, col), (row, col-1), (row, col+1)])
        # Check each of four neighbors to see if it is part of the same wire
        def matching_wire_func(i, output_pos):
            neighbor_position = neighbors_positions[i]
            neighbor_wire_num = self.position_to_wire_num_jax(neighbor_position, board_layout)
            is_prev_neighbor = (neighbor_wire_num == input_wire_num) & \
                       (neighbor_position[0] >= 0) & (neighbor_position[0] < board_layout.shape[0]) & \
                       (neighbor_position[1] >= 0) & (neighbor_position[1] < board_layout.shape[1])
            output_pos = jax.lax.select(is_prev_neighbor, neighbor_position, output_pos)
            return output_pos
        matching_pos = jax.lax.fori_loop(0, 3, matching_wire_func, INVALID)
        return matching_pos


    #@jax.jit
    # This method is used by the extend_wires_jax method
    def get_neighbors_same_wire_jax(self, input_pos: jnp.array, board_layout: Array) -> jnp.array:
        """ Returns a list of adjacent cells belonging to the same wire.

            Args:
                input_pos (jnp.array): 2D position of cell in board_layout
                board_layout (Array): 2D layout of board with wires encoded

            Returns:
                (jnp.array) : a list of cells (2D positions) adjacent to the queried cell which belong to the same wire
                                The output will always have four entries but some will be INVALID.
        """
        output_list = []
        wire_num = self.position_to_wire_num_jax(input_pos, board_layout)
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
    def num_wire_adjacencies_jax(self, input_pos: jnp.array, wire_num: Array, board_layout: Array) -> Array:
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
        wire_adjacent = self.position_to_wire_num_jax(jnp.array((input_pos[0] - 1, input_pos[1])), board_layout)
        num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
        # Check below
        wire_adjacent = self.position_to_wire_num_jax(jnp.array((input_pos[0] + 1, input_pos[1])), board_layout)
        num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
        # Check left
        wire_adjacent = self.position_to_wire_num_jax(jnp.array((input_pos[0], input_pos[1] - 1)), board_layout)
        num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
        # Check right
        wire_adjacent = self.position_to_wire_num_jax(jnp.array((input_pos[0], input_pos[1] + 1)), board_layout)
        num_adjacencies = num_adjacencies + jax.lax.select(wire_adjacent == wire_num, 1, 0)
        return num_adjacencies


    #@jax.jit
    # This method is used by the extend_wires_jax method
    def get_open_adjacent_cells_jax(self, input_pos: jnp.array, board_layout: Array) -> jnp.array:
        """ Returns a list of open cells adjacent to the input cell.

            Args:
                input_pos (Array, Tuple[int, int]): The 2D position of the input cell to search adjacent to.
                board_layout (Array): 2D layout of the board with wires encoded

            Returns:
                (jnp.array): List of 2D integer tuples, up to four available cells adjacent to the input cell.
        """
        rows, cols = self._rows, self._cols
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
    def position_to_wire_num_jax(self, position: jnp.array, board_layout: Array) -> Array:
        """ Returns the wire number of the given cell position

            Args:
                position (jnp.array[int, int]): 2D tuple of the [row, col] position
                board_layout (Array): 2D layout of the board with wires encoded

            Returns:
                (int) : The wire number that the cell belongs to.
                        Returns -1 if not part of a wire or out-of-bounds.
        """
        rows, cols = self._rows, self._cols
        row, col = position[0], position[1]
        cell_encoding = board_layout[row, col]
        wire_num = jax.lax.select((0 <= row) & (row < rows) & (0 <= col) & (col < cols),
                                  self.cell_encoding_to_wire_num_jax(cell_encoding), -1)
        return wire_num


    #@jax.jit
    def cell_encoding_to_wire_num_jax(self, cell_encoding: Array) -> Array:
        """ Returns the wire number of the given cell value

            Args:
                cell_encoding (int) : the value of the cell in self.layout

            Returns:
                (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
        """
        output = jax.lax.select(cell_encoding == 0, -1, (cell_encoding - 1) // 3)
        return jax.lax.convert_element_type(output, jnp.int32)


    #@jax.jit
    def position_to_cell_type_jax(self, position: jnp.array, board_layout: Array) -> Array:
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


##############################################################################################
if __name__ == '__main__':
    board_object = RandomSeedBoard(3, 3, 3)

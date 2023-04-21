import random
from typing import Tuple, Optional
import jax.numpy as jnp
import jax.random
from chex import PRNGKey, Array
from copy import deepcopy

from ic_routing_board_generation.board_generator.jax_data_model.wire import Wire, create_wire, stack_reverse, stack_push

EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3  # Ideally should be imported from Jumanji


class Grid:
    def __init__(self, rows: int, cols: int, grid_layout: Optional[Array] = None) -> None:
        """
        Constructor for the Grid class.
        Args:
            rows: number of rows in the grid
            cols: number of columns in the grid
        Returns:
            None
        """
        self.rows = rows
        self.cols = cols
        self.grid_layout = None
        # # if layout is not provided, create a rows x cols grid with all cells empty, else use the provided layout in a jax way
        # # self.grid_layout = jax.lax.cond(
        # #     grid_layout is None,
        # #     lambda _: jnp.zeros((rows, cols), dtype=jnp.int32),
        # #     lambda _: grid_layout,
        # #     None
        # # )
        # self.grid_layout = jnp.zeros((rows, cols), dtype=jnp.int32) if grid_layout is None else grid_layout
        # assert self.grid_layout.shape == (rows, cols), "Grid layout shape does not match the rows and columns provided"

    def start_bfs(self, key: PRNGKey, board: Array) -> Tuple[PRNGKey, Wire, Array, Array, Array]:
        """This takes a key and a board and returns a bfs_stack

        Args:
            key (PRNGKey): Random key

        Returns:
            Tuple[PRNGKey, Wire, Array, Array, Array]: bfs_stack (key, wire, board, queue, visited)
            """
        # Make a deep copy of the board in case we need to revert
        self.grid_layout = deepcopy(board)

        # First pick a random start and end position
        key, subkey = jax.random.split(key)
        start_int, end_int, _ = self.pick_random_start_and_end(subkey, board)

        # Convert them to tuples
        start, end = self.convert_int_to_tuple(start_int), self.convert_int_to_tuple(end_int)

        # print(f'Start: {start}, End: {end}')

        # Get the wire id this is the maximum of the board//3
        wire_id = jnp.max(board) // 3

        # Create a Wire object
        wire = create_wire(max_size=self.rows * self.cols, start=start, end=end, wire_id=wire_id)

        # Create a queue and visited array
        queue = jnp.zeros((self.rows * self.cols), dtype=jnp.int32)
        visited = -1 * jnp.ones((self.rows * self.cols), dtype=jnp.int32)

        # Update queue to reflect the start position
        queue = queue.at[start_int].set(1)

        # Update visited to reflect the start position. This is done to avoid the start position being visited again.
        # We set it to itself.
        visited = visited.at[start_int].set(start_int)

        # Return the bfs_stack

        return key, wire, board, queue, visited

    def main_bfs_loop(self, key: PRNGKey, board: Array) -> Tuple[Wire, Array]:
        """This takes a key and a board and returns a bfs_stack

        Args:
            key (PRNGKey): Random key

        Returns:
            Tuple[PRNGKey, Wire, Array, Array, Array]: bfs_stack (key, wire, board, queue, visited)
            """
        # Make a deep copy of the board in case we need to revert
        self.grid_layout = deepcopy(board)

        # First pick a random start and end position
        key, subkey = jax.random.split(key)
        start_int, end_int, visited = self.pick_random_start_and_end(subkey, board)

        wire_id = jnp.max(board) // 3
        start, end = self.convert_int_to_tuple(start_int), self.convert_int_to_tuple(end_int)

        # Get path from start to end and visited
        wire = self.get_wire_from_start_end_visited(start_int, end_int, visited, wire_id)

        # populate board
        board = self.jax_fill_grid(wire, board)

        return wire, board

    def initialise_queue_and_visited(self, start_int: int) -> Tuple[Array, Array]:
        """Initialises the queue and visited arrays

        Args:
            start_int (int): The start position as an integer

        Returns:
            Tuple[Array, Array]: queue, visited
        """
        # Create a queue and visited array
        queue = jnp.zeros((self.rows * self.cols), dtype=jnp.int32)
        visited = -1 * jnp.ones((self.rows * self.cols), dtype=jnp.int32)

        # Update queue to reflect the start position
        queue = queue.at[start_int].set(1)

        # Update visited to reflect the start position. This is done to avoid the start position being visited again.
        # We set it to itself.
        # visited = visited.at[start_int].set(start_int)

        return queue, visited

    def is_queue_full(self,
                      bfs_stack: Tuple[PRNGKey, Wire, Array, Array, Array]
                      ):
        key, wire, board, queue, visited = bfs_stack
        return (queue != 0).any()

    # @jax.jit
    def bfs_loop(self, key, wire: Wire, board: Array, queue: Array, visited: Array):
        # @jax.disable_jit()
        def loop_body(full_stack):
            i, bfs_stack, end_reached = full_stack
            key, wire, board, queue, visited = bfs_stack

            # Update the queue and visited arrays
            queue, visited, board = grid.update_queue_and_visited(queue, visited, board)

            # Check if the end has been reached
            end_reached = grid.check_if_end_reached(wire, visited)

            # Return the loop condition and the updated bfs_stack
            return i + 1, (key, wire, board, queue, visited), end_reached

        i = 0

        loop_cond = lambda full_stack: jnp.logical_and(full_stack[0] < (self.cols * self.rows),
                                                       jnp.logical_not(full_stack[-1]))

        end_reached = False
        bfs_stack = (key, wire, board, queue, visited)
        full_stack = (i, bfs_stack, end_reached)

        final_i, final_bfs_stack, end_reached = jax.lax.while_loop(loop_cond, loop_body,
                                                                   full_stack)

        return final_bfs_stack

    def set_queue(self, wire: Wire, queue: Array, visited: Array) -> Array:
        """Sets queue to 1 for wire.start if haas not already been set or visited"""
        # Convert wire.start to int
        start_int = self.convert_tuple_to_int(wire.start)

        # Loop through queue and set wire start to 1 in a Jax way
        queue = queue.at[start_int].set(1)

        return queue

    def convert_tuple_to_int(self, position: Tuple[int, int]) -> Array:
        # return int(position[0] * self.cols + position[1])
        return jnp.array(position[0] * self.cols + position[1], dtype=jnp.int32)
        # return position[0] * self.cols + position[1]

    def convert_int_to_tuple(self, position: int) -> Array:
        # return int(position // self.cols), int(position % self.cols)
        return jnp.array([position // self.cols, position % self.cols], dtype=jnp.int32)
        # return position // self.cols, position % self.cols

    def update_queue_and_visited(self, queue: Array, visited: Array, board: Array) -> Tuple[Array, Array, Array]:
        """Updates the queue and visited arrays

        Args:
            queue (Array): Array indicating the next positions to visit
            visited (Array): Array indicating the previous state visited from each position
            board (Array): Array indicating the current state of the board

        Returns:
            Tuple[Array, Array, Array]: Updated queue, visited and board arrays

            """
        # Current position is the lowest value in the queue that is greater than 0
        curr_int = jnp.argmin(jnp.where((queue > 0), queue, jnp.inf))

        # Convert current position to tuple
        curr_pos = self.convert_int_to_tuple(curr_int)

        # Define possible movements
        row = [-1, 0, 1, 0]
        col = [0, 1, 0, -1]

        # Shuffle row and col in the same way
        perm = jax.random.permutation(jax.random.PRNGKey(0), jnp.arange(4))

        row = jnp.array(row, dtype=jnp.int32)[perm]
        col = jnp.array(col, dtype=jnp.int32)[perm]

        # Do a jax while loop of the update_queue_visited_loop

        # @jax.disable_jit()
        def qv_loop_cond(full_qv_stack):
            j, *_ = full_qv_stack
            return j < 4

        # @jax.disable_jit()
        def qv_loop_body(full_qv_stack):
            j, curr_pos, curr_int, row, col, visited, queue, board = full_qv_stack
            j, curr_pos, curr_int, row, col, visited, queue, board = self.update_queue_visited_loop(j, curr_pos,
                                                                                                    curr_int, row, col,
                                                                                                    visited, queue,
                                                                                                    board)

            return j, curr_pos, curr_int, row, col, visited, queue, board

        j = 0
        full_qv_stack = (j, curr_pos, curr_int, row, col, visited, queue, board)

        full_qv_stack = jax.lax.while_loop(qv_loop_cond, qv_loop_body, full_qv_stack)

        *_, visited, queue, board = full_qv_stack

        # remove current position from queue
        queue = queue.at[curr_int].set(0)

        return queue, visited, board

    def update_queue_visited_loop(self, j, curr_pos, curr_int, row, col, visited, queue, board):
        # Calculate new position
        new_row = jnp.array(curr_pos, dtype=jnp.int32)[0] + jnp.array(row, dtype=jnp.int32)[j]
        new_col = jnp.array(curr_pos, dtype=jnp.int32)[1] + jnp.array(col, dtype=jnp.int32)[j]
        pos_int = self.convert_tuple_to_int((new_row, new_col))

        # Check value of new position index in visited
        size_cond = jnp.logical_and(jnp.logical_and(0 <= new_row, new_row < self.rows),
                                    jnp.logical_and(0 <= new_col, new_col < self.cols))
        cond_1 = (visited[pos_int] == -1)
        cond_2 = (queue[pos_int] == 0)
        cond_3 = (board[new_row, new_col] == 0)
        # cond_4 = (j < 4)

        cond_a = size_cond  # & cond_4
        cond_b = cond_1 & cond_2 & cond_3

        condition = jax.lax.cond((cond_a & cond_b), lambda _: True, lambda _: False, None)

        curr_val = jnp.max(jnp.where((queue > 0), queue, -jnp.inf))
        queue = jax.lax.cond(
            condition, lambda _: queue.at[pos_int].set(curr_val + 1), lambda _: queue, None)
        visited = jax.lax.cond(condition, lambda _: visited.at[pos_int].set(curr_int), lambda _: visited, None)

        # print('queue', queue)
        # print('visited', visited)

        return j + 1, curr_pos, curr_int, row, col, visited, queue, board

    def check_if_end_reached(self, wire: Wire, visited: Array) -> bool:
        """Check if the end of the wire has been reached"""
        end_int = self.convert_tuple_to_int(wire.end)
        return visited[end_int] != -1

    def get_path(self, wire_visited_tuple: Tuple[Wire, Array, int]) -> Wire:
        """Populates wire path using wire operations and visited array"""
        wire, visited, curr_poss = wire_visited_tuple

        wire_start = self.convert_tuple_to_int(wire.start)

        # Using a jax while loop, update the wire path. The loop will stop when the wire start is reached
        def cond_fun(wire_visited_tuple):
            wire, _, _ = wire_visited_tuple
            index_to_check = jnp.max(jnp.array([0, wire.insertion_index - 1]))
            return (wire.path[index_to_check] != wire_start)

        def body_fun(wire_visited_tuple):
            wire, visited, curr_poss = wire_visited_tuple
            # Get the next position

            next_pos = visited[curr_poss]
            wire = stack_push(wire, curr_poss)
            wire_visited_tuple = (wire, visited, next_pos)
            return wire_visited_tuple

        wire_visited_tuple = jax.lax.while_loop(cond_fun, body_fun, wire_visited_tuple)
        wire, visited, _ = wire_visited_tuple
        wire = stack_reverse(wire)

        return wire

    def get_start_end(self, wire: Wire, board: Array) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Function to get the start and end positions of the path.
        Args:
            wire: wire to get the start and end positions for
            board: board to get the start and end positions for
        Returns:
            tuple containing the start and end positions
        """
        # Get the start position
        start = wire.start
        # Get the end position
        end = wire.end
        return start, end

    @staticmethod
    def pick_random_start_and_end_(key: PRNGKey, board: Array) -> Tuple[int, int]:
        """Function to pick a start and end position for the path. Only want positions where the board is 0"""
        # Get the flattened board
        board_flat = board.ravel()
        # Get the indices of the board where the value is 0
        indices = jnp.where(board_flat == 0)[0]
        # Sample two random indices from the indices array
        start, end = jax.random.choice(key, indices, shape=(2,), replace=False)

        return start, end

    def pick_random_start_and_end(self, key: PRNGKey, board: Array) -> Tuple[Array, Array, Array]:
        """Function to pick a start and end position for the path. Only want positions where the board is 0"""
        # Get the flattened board
        board_flat = board.ravel()
        # Get the indices of the board where the value is 0
        indices = jnp.where(board_flat == 0)[0]

        # Assert that there are at least 2 indices
        assert len(indices) >= 2, 'There are not enough indices to pick a start and end position'

        # Sample two random indices from the indices array
        # start, end = jax.random.choice(key, indices, shape=(2,), replace=False)

        # Split the key
        key, subkey = jax.random.split(key)

        start = jax.random.choice(subkey, indices, shape=(1,), replace=False)[0]

        # Split the key
        key, subkey = jax.random.split(key)

        end_indices, visited = self.get_contiguous_indices(start_int=start, board=board)
        # print('end_indices', end_indices)

        end = jax.random.choice(key, end_indices, shape=(1,), replace=False)[0]
        # print('start', start)
        # print('end', end)

        return start, end, visited

    def get_wire_from_start_end_visited(self, start_int: int, end_int: int, visited: Array, wire_id: int) -> Wire:
        """ Given a start and end, we can get the path from the visited array"""
        # Create a wire
        start, end = self.convert_int_to_tuple(start_int), self.convert_int_to_tuple(end_int)

        wire = create_wire(max_size=self.rows * self.cols, start=start, end=end, wire_id=wire_id)
        # Get the path
        print(f'end: {end_int}')
        filled_wire = self.get_path((wire, visited, end_int))
        return filled_wire

    def get_contiguous_indices(self, start_int: Array, board: Array) -> Array:
        """Function to get all the cells within a contiguous region starting from the start position on the board.
         A cell is in the contiguous region if it can be connected to the start cell by a path of adjacent cells.
         Valid cells have value 0.

            Args:
                start: start position
                board: board to get the contiguous region for
            Returns:
                array containing the indices of contiguous region
        """

        # Get the flattened board
        board_flat = board.ravel()
        # Get the indices of the board where the value is 0
        indices = jnp.where(board_flat == 0)[0]

        # initialise the queue and visited arrays
        queue, visited = self.initialise_queue_and_visited(start_int=start_int)

        # Get the row and column indices for the start position

        # Loop update the queue and visited arrays until the queue is empty
        def cond_fun(queue_visited_tuple):
            queue, visited, _ = queue_visited_tuple
            return jnp.any(queue > 0)

        def body_fun(queue_visited_tuple):
            # queue, visited, board = queue_visited_tuple
            queue_visited_tuple = self.update_queue_and_visited(*queue_visited_tuple)
            return queue_visited_tuple

        queue_visited_tuple = jax.lax.while_loop(cond_fun, body_fun, (queue, visited, board))

        _, visited, _ = queue_visited_tuple

        # Get the indices of the contiguous region
        contiguous_indices = jnp.where(visited > -1)[0]

        # Remove the start position from the contiguous indices
        contiguous_indices = contiguous_indices[contiguous_indices != start_int]
        # print('contiguous_indices', contiguous_indices)
        # print('start_int', start_int)
        return contiguous_indices, visited

    @staticmethod
    def jax_fill_grid(wire: Wire, board: Array) -> Array:
        """Places a wire path on the board in a Jax way
                paths are encoded as 3 * wire_num + 1
                starts are encoded as 3 * wire_num + 2
                ends are encoded as 3 * wire_num + 3 """
        path = wire.path[:wire.insertion_index]
        # Get the wire number
        wire_num = wire.wire_id

        # Start is at the first position in the path
        start = path[0]
        # End is at the last position in the path
        end = path[-1]

        # populate these positions in the flattened board
        board_flat = board.ravel()
        board_flat = board_flat.at[start].set(3 * wire_num + POSITION)
        board_flat = board_flat.at[end].set(3 * wire_num + TARGET)
        board_flat = board_flat.at[path[1:-1]].set(3 * wire_num + PATH)

        # reshape the board
        board = board_flat.reshape(board.shape)
        return board


if __name__ == '__main__':
    size = 10
    max_size = size ** 2
    num_agents = 5

    # Sample a start and end positions within the flattened board. Start and end positions are not the same.
    seed = random.randint(0, 1000)

    key = jax.random.PRNGKey(seed)
    board = jnp.zeros((size, size), dtype=jnp.int32)

    # Randomly generate board of int type with 90% 0s and 10% 1s
    # board = -10 * jax.random.bernoulli(key, 0.2, shape=(size, size)).astype(jnp.int32)
    print('Board: \n{}'.format(board))
    #
    # Initialize the grid
    grid = Grid(size, size)

    # First start BFS
    for k in range(num_agents):
        new_key, key = jax.random.split(key)
        print(f'Agent {k + 1}')
        # print('key: ', key)
        # print('new_key: ', new_key)

        # bfs_stack = grid.start_bfs(new_key, board)
        #
        # new_key = bfs_stack[0]
        # wire_1 = bfs_stack[1]
        # board = bfs_stack[2]
        # queue = bfs_stack[3]
        # visited = bfs_stack[4]
        #
        # final_bfs_stack = grid.bfs_loop(new_key, wire_1, board, queue, visited)
        #
        # new_key = final_bfs_stack[0]
        # wire_1 = final_bfs_stack[1]
        # board = final_bfs_stack[2]
        # queue = final_bfs_stack[3]
        # visited = final_bfs_stack[4]
        #
        # curr_pos = grid.convert_tuple_to_int(wire_1.end)
        # wire_1 = grid.get_path((wire_1, visited, curr_pos))
        # board = grid.jax_fill_grid(wire_1, board)
        wire, board = grid.main_bfs_loop(new_key, board)

        print(f'Board: \n {board}')

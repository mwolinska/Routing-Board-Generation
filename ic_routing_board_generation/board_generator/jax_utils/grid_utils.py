import random
from typing import Tuple, Optional
import jax.numpy as jnp
import jax.random
from chex import Array, PRNGKey

<<<<<<< refs/remotes/origin/dev-2:ic_routing_board_generation/board_generator/jax_utils/grid_utils.py
from ic_routing_board_generation.board_generator.jax_data_model.wire import Wire, create_wire, stack_reverse, stack_push
=======
from ic_routing_board_generation.board_generator.wire import Wire, create_wire, stack_reverse, stack_push

>>>>>>> Including optimise_wire function:ic_routing_board_generation/board_generator/grid_utils.py
EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3  # Ideally should be imported from Jumanji


# Usage Guide:
# 1. Given a partially filled board, wire_start, wire_end, and wire_id, we use the thin_wire function to
# thin out the wire so that is a valid wire on the board.

# 2. If unable to provide the start and the end positions,
# you can use the get_start_end_positions function to get the start and end positions of the wire on the board.

# Example below

#   wire_num = 0  # Wire number x, is x s.t. wire x is encoded as 3 * x + 1.
#   wire_start, wire_end = get_start_end_positions(board, wire_num)
#   seed = random.randint(0, 10000)
#   key = jax.random.PRNGKey(seed)
#   board = thin_wire(key, board, wire_start, wire_end, wire_num)


# Make a function that thins out a wire when given a board and wire

class Grid:
    def __init__(self, rows: int, cols: int, fill_num: Optional[int] = 0) -> None:
        """
        Constructor for the Grid class.
        Args:
            rows: number of rows in the grid
            cols: number of columns in the grid
            fill_num: number to fill the grid with


        Returns:
            None
        """
        self.rows = rows
        self.cols = cols
        self.fill_num = fill_num

    def convert_tuple_to_int(self, position: Tuple[int, int]) -> int:
        return jnp.array(position[0] * self.cols + position[1], dtype=jnp.int32)
        # return int(position[0] * self.cols + position[1])
        # return position[0] * self.cols + position[1]

    def convert_int_to_tuple(self, position: int) -> Tuple[int, int]:
        # return int(position // self.cols), int(position % self.cols)
        return jnp.array([position // self.cols, position % self.cols], dtype=jnp.int32)
        # return position // self.cols, position % self.cols

    # def update_queue_and_visited(self, queue: Array, visited: Array, board: Array, key: Optional[PRNGKey] = 100,
    #                              use_empty: Optional[bool] = False) -> Tuple[Array, Array, Array]:
    #     """Updates the queue and visited arrays
    #
    #     Args:
    #         queue (Array): Array indicating the next positions to visit
    #         visited (Array): Array indicating the previous state visited from each position
    #         board (Array): Array indicating the current state of the board
    #         key (Optional[PRNGKey], optional): Key for random number generation. Defaults to 100.
    #         use_empty (Optional[bool], optional): Whether to use empty positions or not. Defaults to False.
    #
    #     Returns:
    #         Tuple[Array, Array, Array]: Updated queue, visited and board arrays
    #
    #         """
    #     # Current position is the lowest value in the queue that is greater than 0
    #     curr_int = jnp.argmin(jnp.where((queue > 0), queue, jnp.inf))
    #
    #     # Convert current position to tuple
    #     curr_pos = self.convert_int_to_tuple(curr_int)
    #
    #     # Define possible movements
    #     row = [-1, 0, 1, 0]
    #     col = [0, 1, 0, -1]
    #
    #     # Loop through possible movements but shuffle the order
    #
    #     range_list = jnp.arange(4)
    #     range_list = jax.random.permutation(key, range_list, independent=True)
    #
    #     for i in range_list:
    #         # Calculate new position
    #
    #         new_row = curr_pos[0] + row[i]
    #         new_col = curr_pos[1] + col[i]
    #         pos_int = self.convert_tuple_to_int((new_row, new_col))
    #
    #         # Check value of new position index in visited
    #         use_cond = (board[new_row, new_col] == 3 * self.fill_num + PATH) if not use_empty else (
    #                     (board[new_row, new_col] == 3 * self.fill_num + PATH) or (board[new_row, new_col] == EMPTY))
    #
    #         condition = (0 <= new_col < self.cols) and (0 <= new_row < self.rows) and visited[
    #             pos_int] == -1 and queue[
    #                         pos_int] == 0 and use_cond  # (board[new_row, new_col] == 3 * self.fill_num + PATH)
    #
    #         # Update the queue if condition is met in Jax
    #         curr_val = jnp.max(jnp.where((queue > 0), queue, -jnp.inf))
    #
    #         queue = jax.lax.cond(condition, lambda x: x.at[pos_int].set(curr_val + 1), lambda x: x, queue)
    #         # queue = jax.lax.cond(condition, lambda x: x.at[pos_int].set(curr_val + 1), lambda x: x, queue)
    #         visited = jax.lax.cond(condition, lambda x: x.at[pos_int].set(curr_int), lambda x: x, visited)
    #
    #     # remove current position from queue
    #     queue = queue.at[curr_int].set(0)
    #
    #     return queue, visited, board
    #
    def update_queue_and_visited(self, queue: Array, visited: Array, board: Array, key: Optional[PRNGKey] = 100,
                                 use_empty: Optional[bool] = False) -> Tuple[Array, Array, Array]:
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
        perm = jax.random.permutation(key, jnp.arange(4), independent=True)

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
                                                                                                    board, use_empty)

            return j, curr_pos, curr_int, row, col, visited, queue, board

        j = 0
        full_qv_stack = (j, curr_pos, curr_int, row, col, visited, queue, board)

        full_qv_stack = jax.lax.while_loop(qv_loop_cond, qv_loop_body, full_qv_stack)

        *_, visited, queue, board = full_qv_stack

        # remove current position from queue
        queue = queue.at[curr_int].set(0)

        return queue, visited, board

    def update_queue_visited_loop(self, j, curr_pos, curr_int, row, col, visited, queue, board, use_empty=False):
        # Calculate new position
        new_row = jnp.array(curr_pos, dtype=jnp.int32)[0] + jnp.array(row, dtype=jnp.int32)[j]
        new_col = jnp.array(curr_pos, dtype=jnp.int32)[1] + jnp.array(col, dtype=jnp.int32)[j]
        pos_int = self.convert_tuple_to_int((new_row, new_col))

        # Check value of new position index in visited
        size_cond = jnp.logical_and(jnp.logical_and(0 <= new_row, new_row < self.rows),
                                    jnp.logical_and(0 <= new_col, new_col < self.cols))
        cond_1 = (visited[pos_int] == -1)
        cond_2 = (queue[pos_int] == 0)
        # cond_3 = (board[new_row, new_col] == 0)

        # cond_3 = (board[new_row, new_col] == 3 * self.fill_num + PATH) if not use_empty else ((board[new_row, new_col] == 3 * self.fill_num + PATH) or (board[new_row, new_col] == EMPTY))

        # cond_3 = jax.lax.cond(use_empty, lambda _: (board[new_row, new_col] == 3 * self.fill_num + PATH) or (board[new_row, new_col] == EMPTY), lambda _: (board[new_row, new_col] == 3 * self.fill_num + PATH), None)

        cond_3 = jnp.logical_or(jnp.logical_and(use_empty,
                                                jnp.logical_or((board[new_row, new_col] == 3 * self.fill_num + PATH),
                                                               (board[new_row, new_col] == EMPTY))),
                                jnp.logical_and(~use_empty, (board[new_row, new_col] == 3 * self.fill_num + PATH)))

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
        # Convert wire.end to int
        end_int = self.convert_tuple_to_int(wire.end)

        return visited[end_int] != -1

    def get_path(self, wire_visited_tuple: Tuple[Wire, Array, int]) -> Wire:
        """Populates wire path using wire operations and visited array"""
        # Convert wire.end to int
        # Push wire end at insertion index
        wire, visited, cur_poss = wire_visited_tuple

        wire_start = self.convert_tuple_to_int(wire.start)

        # Using a jax while loop update the wire path. The loop will stop when the wire start is reached
        def cond_fun(wire_visited_tuple):
            wire, visited, _ = wire_visited_tuple
            # Check that last position in wire path is not the wire start
            # Check the previous input which will be max of 0 and insertion index - 1
            index_to_check = jnp.max(jnp.array([0, wire.insertion_index - 1]))
            return (wire.path[index_to_check] != wire_start)

        def body_fun(wire_visited_tuple):
            wire, visited, cur_poss = wire_visited_tuple
            # Get the next position

            next_pos = visited[cur_poss]

            wire = stack_push(wire, cur_poss)

            wire_visited_tuple = (wire, visited, next_pos)

            return wire_visited_tuple

        wire_visited_tuple = jax.lax.while_loop(cond_fun, body_fun, wire_visited_tuple)

        wire, visited, _ = wire_visited_tuple
        wire = stack_reverse(wire)

        return wire

    def get_start_end(self, wire: Wire) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Function to get the start and end positions of the path.
        Args:
            wire: wire to get the start and end positions for
        Returns:
            tuple containing the start and end positions
        """
        # Get the start position
        start = wire.start
        # Get the end position
        end = wire.end
        return start, end

    def remove_path(self, board: Array, wire: Wire) -> Array:
        """Removes a wire path from the board before repopulating the new path"""
        # Get the wire number
        wire_fill_num = 3 * wire.wire_id + PATH

        # Populate the cells in the board that have the wire_fill_num to 0
        board = jnp.where(board == wire_fill_num, 0, board)

        return board

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


def get_start_end_positions(board: Array, wire_num: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Function to get the start and end positions of the path.
    Args:
        wire: wire to get the start and end positions for
    Returns:
        tuple containing the start and end positions
    """
    # Get the start position
    start = jnp.argwhere(board == 3 * wire_num + POSITION)
    # Get the end position
    end = jnp.argwhere(board == 3 * wire_num + TARGET)
    # convert numpy array to tuple
    start = tuple(start[0])
    end = tuple(end[0])
    return start, end


def thin_wire(key: PRNGKey, board: Array, wire_start: Tuple[int, int], wire_end: Tuple[int, int],
              wire_num: int) -> Array:
    """Thin a wire on the board by performing a bfs search and then populating the board
    Args:
        board: board to thin the wire on
        wire_start: start position of the wire
        wire_end: end position of the wire
        wire_num: wire number
        """
    # Initialize the grid
    grid = Grid(board.shape[0], board.shape[1], fill_num=wire_num)
    max_size = board.shape[0] * board.shape[1]

    # Initialize the wire
    wire = create_wire(max_size=max_size, wire_id=wire_num, start=wire_start, end=wire_end)

    # Set start and end in board to the same value as wire
    board = board.at[wire_start].set(grid.fill_num * 3 + 1)
    board = board.at[wire_end].set(grid.fill_num * 3 + 1)

    queue = jnp.zeros(max_size, dtype=jnp.int32)
    visited = -1 * jnp.ones(max_size, dtype=jnp.int32)

    # Update queue to reflect the start position
    queue = queue.at[grid.convert_tuple_to_int(wire.start)].set(1)

    # Update visited to reflect the start position. This is done to avoid the start position being visited again.
    # visited = visited.at[grid.convert_tuple_to_int(wire.start)].set(grid.convert_tuple_to_int(wire.start))

    count = 0
    for _ in range(max_size):
        # count += 1
        # print(f'count is {count}')
        # print(f'queue is {queue}')
        # print(f'visited is {visited}')

        bfs_stack = (key, wire, board, queue, visited)
        board = bfs_stack[2]
        queue = bfs_stack[3]
        visited = bfs_stack[4]

        queue, visited, board = grid.update_queue_and_visited(queue, visited, board, key)

        if grid.check_if_end_reached(wire, visited):
            break

    curr_pos = grid.convert_tuple_to_int(wire.end)
    wire = grid.get_path((wire, visited, curr_pos))
    board = grid.remove_path(board, wire)
    board = grid.jax_fill_grid(wire, board)

    return board


def optimise_wire(key: PRNGKey, board: Array, wire_num: int) -> Array:
    """Essentially the same as thin_wire but also uses empty spaces to optimise the wire
    Args:
        board: board to thin the wire on
        wire_start: start position of the wire
        wire_end: end position of the wire
        wire_num: wire number
        """
    # Initialize the grid
    grid = Grid(board.shape[0], board.shape[1], fill_num=wire_num)
    max_size = board.shape[0] * board.shape[1]

    start_num = 3 * wire_num + POSITION
    end_num = 3 * wire_num + TARGET

    # Find start and end positions
    wire_start = jnp.argwhere(board == start_num)[0]
    wire_end = jnp.argwhere(board == end_num)[0]

    # Initialize the wire
    wire = create_wire(max_size=max_size, wire_id=wire_num, start=wire_start, end=wire_end)

    # Set start and end in board to the same value as wire
    board = board.at[wire_start].set(grid.fill_num * 3 + 1)
    board = board.at[wire_end].set(grid.fill_num * 3 + 1)

    queue = jnp.zeros(max_size, dtype=jnp.int32)
    visited = -1 * jnp.ones(max_size, dtype=jnp.int32)

    # Update queue to reflect the start position
    queue = queue.at[grid.convert_tuple_to_int(wire.start)].set(1)

    # Update visited to reflect the start position. This is done to avoid the start position being visited again.
    # visited = visited.at[grid.convert_tuple_to_int(wire.start)].set(grid.convert_tuple_to_int(wire.start))

    count = 0
    for _ in range(max_size):
        bfs_stack = (key, wire, board, queue, visited)
        board = bfs_stack[2]
        queue = bfs_stack[3]
        visited = bfs_stack[4]

        queue, visited, board = grid.update_queue_and_visited(queue, visited, board, key, use_empty=True)

        if grid.check_if_end_reached(wire, visited):
            break

    curr_pos = grid.convert_tuple_to_int(wire.end)
    wire = grid.get_path((wire, visited, curr_pos))
    board = grid.remove_path(board, wire)
    board = grid.jax_fill_grid(wire, board)

    return board


if __name__ == '__main__':
    # Example Usage
    boards = [jnp.array([[0, 0, 0, 0, 0], [0, 3, 1, 1, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 2]],
                        dtype=jnp.int32),
              jnp.array([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 0.], [1., 1., 1., 1., 1.], [1., 1., 1., 2., 1.],
                         [1., 1., 3., 1., 1.]],
                        dtype=jnp.int32)]

    # wire_start = (4, 4)
    # wire_end = (1, 1)
    for i, board in enumerate(boards):
        print(f'Original board {i + 1}: \n {board} \n')

        wire_num = 0  # Wire number x, is x s.t. wire x is encoded as 3 * x + 1.

        wire_start, wire_end = get_start_end_positions(board, wire_num)

        seed = random.randint(0, 10000)

        key = jax.random.PRNGKey(seed)

        board1 = thin_wire(key, board, wire_start, wire_end, wire_num)
        board2 = optimise_wire(key, board, wire_num)

        print(f'Board after thinning wire {1}: \n {board1} \n')
        print(f'Board after optimising wire {1}: \n {board2} \n')

    board_2 = jnp.array([[4, 4, 4, 4, 4, 4],
                         [4, 3, 6, 4, 4, 4],
                         [5, 1, 1, 1, 1, 2]], dtype=jnp.int32)

    print(f'Original board 3: \n {board_2} \n')

    for i in list(range(2)):
        wire_num = i
        wire_start, wire_end = get_start_end_positions(board_2, wire_num)
        seed = random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)
        board_2 = thin_wire(key, board_2, wire_start, wire_end, wire_num)
<<<<<<< refs/remotes/origin/dev-2:ic_routing_board_generation/board_generator/jax_utils/grid_utils.py
        print(f'Board after thinning wire {i+1}: \n {board_2} \n')
=======
        print(f'Board after thinning wire {i + 1}: \n {board_2} \n')
>>>>>>> Including optimise_wire function:ic_routing_board_generation/board_generator/grid_utils.py

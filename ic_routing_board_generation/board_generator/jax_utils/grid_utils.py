import random
from typing import Tuple, Optional
import jax.numpy as jnp
import jax.random
from chex import Array, PRNGKey
from ic_routing_board_generation.board_generator.jax_data_model.wire import Wire, create_wire, stack_push

EMPTY, PATH, POSITION, TARGET = 0, 1, 2, 3  # Ideally should be imported from Jumanji


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

    def convert_tuple_to_int(self, position: Tuple[int, int]) -> Array:
        """Converts a tuple to an integer format but in a jax array"""
        return jnp.array(position[0] * self.cols + position[1], dtype=jnp.int32)

    def convert_int_to_tuple(self, position: int) -> Array:
        """Converts an integer to a tuple format but in a jax array"""
        return jnp.array([position // self.cols, position % self.cols], dtype=jnp.int32)

    def update_queue_and_visited(self, queue: Array, visited: Array, board: Array, key: Optional[PRNGKey] = 100,
                                 use_empty: Optional[bool] = False) -> Tuple[Array, Array, Array]:
        """Updates the queue and visited arrays

        Args:
            queue (Array): Array indicating the next positions to visit
            visited (Array): Array indicating the previous state visited from each position
            board (Array): Array indicating the current state of the board
            key (Optional[PRNGKey]): Key used for random number generation
            use_empty (Optional[bool]): Boolean indicating whether to use empty spaces as valid positions

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

        def qv_loop_cond(full_qv_stack):
            j, *_ = full_qv_stack
            return j < 4

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
        cond_3 = jax.lax.cond(use_empty, lambda _: jnp.logical_or((board[new_row, new_col] == 3 * self.fill_num + PATH),
                                                                  (board[new_row, new_col] == EMPTY)),
                              lambda _: (board[new_row, new_col] == 3 * self.fill_num + PATH), None)

        condition = jax.lax.cond((size_cond & cond_1 & cond_2 & cond_3), lambda _: True, lambda _: False, None)

        curr_val = jnp.max(jnp.where((queue > 0), queue, -jnp.inf))
        queue = jax.lax.cond(
            condition, lambda _: queue.at[pos_int].set(curr_val + 1), lambda _: queue, None)
        visited = jax.lax.cond(condition, lambda _: visited.at[pos_int].set(curr_int), lambda _: visited, None)

        return j + 1, curr_pos, curr_int, row, col, visited, queue, board

    def check_if_end_reached(self, wire: Wire, visited: Array) -> bool:
        """Check if the end of the wire has been reached"""
        # Convert wire.end to int
        end_int = self.convert_tuple_to_int(wire.end)

        return visited[end_int] != -1

    def get_path(self, wire_visited_tuple: Tuple[Wire, Array, int]) -> Wire:
        """Populates wire path using wire operations and visited array

        Args:
            wire_visited_tuple (Tuple[Wire, Array, int]): Tuple containing wire, visited array and current position

        Returns:
            Wire: Wire with path populated

            """
        wire, visited, cur_poss = wire_visited_tuple

        wire_start = self.convert_tuple_to_int(wire.start)

        # Using a jax while loop update the wire path. The loop will stop when the wire start is reached
        def cond_fun(wire_visited_tuple):
            wire, visited, _ = wire_visited_tuple
            # Check that last position in wire path is not the wire start
            # Check the previous input which will be max of 0 and insertion index - 1
            index_to_check = jnp.max(jnp.array([0, wire.insertion_index - 1]))
            return wire.path[index_to_check] != wire_start

        def body_fun(wire_visited_tuple):
            wire, visited, cur_poss = wire_visited_tuple

            # Get the next position
            next_pos = visited[cur_poss]
            wire = stack_push(wire, cur_poss)
            wire_visited_tuple = (wire, visited, next_pos)

            return wire_visited_tuple

        wire_visited_tuple = jax.lax.while_loop(cond_fun, body_fun, wire_visited_tuple)

        wire, visited, _ = wire_visited_tuple
        # wire = stack_reverse(wire)

        return wire

    @staticmethod
    def get_start_end(wire: Wire) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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

    @staticmethod
    def remove_path(board: Array, wire: Wire) -> Array:
        """Removes a wire path from the board before repopulating the new path
        paths are encoded as 3 * wire_num + PATH
        starts are encoded as 3 * wire_num + POSITION
        ends are encoded as 3 * wire_num + TARGET

        Args:
            board: board to remove the wire from
            wire: wire to remove from the board
        Returns:
            board with the wire removed from it
            """
        # Get the wire number
        wire_fill_num = 3 * wire.wire_id + PATH

        # Populate the cells in the board that have the wire_fill_num to 0
        board = jnp.where(board == wire_fill_num, 0, board)

        return board

    @staticmethod
    def jax_fill_grid2(wire: Wire, board: Array) -> Array:
        """Places a wire path on the board in a Jax way
                paths are encoded as 3 * wire_num + 1
                starts are encoded as 3 * wire_num + 2
                ends are encoded as 3 * wire_num + 3

            Args:
                wire: wire to place on the board
                board: board to place the wire on
            Returns:
                board with the wire placed on it
                """
        path = wire.path[:wire.insertion_index]
        # Get the wire number
        wire_num = wire.wire_id

        # Start is at the first position in the path
        start = path[0]
        # End is at the last position in the path
        end = path[-1]

        # populate these positions in the flattened board
        board_flat = board.ravel()
        board_flat = board_flat.at[start].set(3 * wire_num + TARGET) # Recall that path ordering is reversed
        board_flat = board_flat.at[end].set(3 * wire_num + POSITION) # Recall that path ordering is reversed
        board_flat = board_flat.at[path[1:-1]].set(3 * wire_num + PATH)

        # reshape the board
        board = board_flat.reshape(board.shape)
        return board

    @staticmethod
    def jax_fill_grid(wire: Wire, board: Array) -> Array:
        """Places a wire path on the board in a Jax way
                paths are encoded as 3 * wire_num + 1
                starts are encoded as 3 * wire_num + 2
                ends are encoded as 3 * wire_num + 3

            Args:
                wire: wire to place on the board
                board: board to place the wire on
            Returns:
                board with the wire placed on it
                """

        board_flat = board.ravel()
        # Need to populate the start and end positions and the path positions using a jax while loop.
        # The loop will stop when -1 is reached in the path
        def cond_fun(i_board_tuple):
            i, board_flat = i_board_tuple
            return wire.path[i] != -1

        def body_fun(i_board_tuple):
            i, board_flat = i_board_tuple
            board_flat = board_flat.at[wire.path[i]].set(3 * wire.wire_id + PATH)
            return i + 1, board_flat

        i, board_flat = jax.lax.while_loop(cond_fun, body_fun, (0, board_flat))

        # update the start and end positions
        board_flat = board_flat.at[wire.path[i -1]].set(3 * wire.wire_id + POSITION)
        board_flat = board_flat.at[wire.path[0]].set(3 * wire.wire_id + TARGET)


        # reshape the board
        board = board_flat.reshape(board.shape)
        return board


def get_start_end_positions(board: Array, wire_num: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Function to get the start and end positions of the path.
    Args:
        board: board to get the start and end positions for
        wire_num: wire number to get the start and end positions for
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
        key: key to use for random number generation
        board: board to thin the wire on
        wire_start: start position of the wire
        wire_end: end position of the wire
        wire_num: wire number

    Returns:
        board with the wire thinned
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

    for _ in range(max_size):

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


def optimise_wire2(key: PRNGKey, board: Array, wire_num: int) -> Array:
    """Essentially the same as thin_wire but also uses empty spaces to optimise the wire
    Args:
        key: key to use for random number generation
        board: board to thin the wire on
        wire_num: wire number

    Returns:
        board with the wire optimised
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
    wire = create_wire(max_size=max_size, wire_id=wire_num, start=wire_start, end=(wire_end[0], wire_end[1]))

    # Set start and end in board to the same value as wire
    board = board.at[wire_start[0], wire_start[1]].set(grid.fill_num * 3 + 1)

    board = board.at[wire_end[0], wire_end[1]].set(grid.fill_num * 3 + 1)

    queue = jnp.zeros(max_size, dtype=jnp.int32)
    visited = -1 * jnp.ones(max_size, dtype=jnp.int32)

    # Update queue to reflect the start position
    queue = queue.at[grid.convert_tuple_to_int(wire.start)].set(1)

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


def optimise_wire(key: PRNGKey, board: Array, wire_num: int) -> Array:
    """Essentially the same as thin_wire but also uses empty spaces to optimise the wire
    Args:
        key: key to use for random number generation
        board: board to thin the wire on
        wire_num: wire number

    Returns:
        board with the wire optimised
        """

    grid = Grid(board.shape[0], board.shape[1], fill_num=wire_num)
    max_size = board.shape[0] * board.shape[1]

    start_num = 3 * wire_num + POSITION
    end_num = 3 * wire_num + TARGET

    flat_start = jnp.argmax(jnp.where(board == start_num, board, 0))
    wire_start = grid.convert_int_to_tuple(flat_start)

    flat_end = jnp.argmax(jnp.where(board == end_num, board, 0))
    wire_end = grid.convert_int_to_tuple(flat_end)

    # Initialize the wire
    wire = create_wire(max_size=max_size, wire_id=wire_num, start=wire_start, end=(wire_end[0], wire_end[1]))

    # Set start and end in board to the same value as wire
    board = board.at[wire_start[0], wire_start[1]].set(grid.fill_num * 3 + 1)

    board = board.at[wire_end[0], wire_end[1]].set(grid.fill_num * 3 + 1)

    queue = jnp.zeros(max_size, dtype=jnp.int32)
    visited = -1 * jnp.ones(max_size, dtype=jnp.int32)

    # Update queue to reflect the start position
    queue = queue.at[grid.convert_tuple_to_int(wire.start)].set(1)

    def loop_body(full_stack):
        i, bfs_stack, end_reached = full_stack
        key, wire, board, queue, visited = bfs_stack

        # Update the queue and visited arrays
        queue, visited, board = grid.update_queue_and_visited(queue, visited, board, key, use_empty=True)

        # Check if the end has been reached
        end_reached = grid.check_if_end_reached(wire, visited)

        # Return the loop condition and the updated bfs_stack
        return i + 1, (key, wire, board, queue, visited), end_reached

    i = 0

    loop_cond = lambda full_stack: jnp.logical_and(full_stack[0] < (grid.cols * grid.rows),
                                                   jnp.logical_not(full_stack[-1]))

    end_reached = False
    bfs_stack = (key, wire, board, queue, visited)
    full_stack = (i, bfs_stack, end_reached)

    final_i, final_bfs_stack, end_reached = jax.lax.while_loop(loop_cond, loop_body,
                                                               full_stack)

    _, wire, board, _, visited = final_bfs_stack

    curr_pos = grid.convert_tuple_to_int(wire.end)
    wire = grid.get_path((wire, visited, curr_pos))
    board = grid.remove_path(board, wire)

    board = grid.jax_fill_grid(wire, board)
    return board


if __name__ == '__main__':
    # Example Usage
    # boards = [jnp.array([[0, 0, 0, 0, 0], [0, 3, 1, 1, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1], [0, 0, 0, 1, 2]],
    #                     dtype=jnp.int32),
    #           jnp.array([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 0.], [1., 1., 1., 1., 1.], [1., 1., 1., 2., 1.],
    #                      [1., 1., 3., 1., 1.]],
    #                     dtype=jnp.int32)]
    #
    # for i, board in enumerate(boards):
    #     print(f'Original board {i + 1}: \n {board} \n')
    #
    #     wire_num = 0  # Wire number x, is x s.t. wire x is encoded as 3 * x + 1.
    #
    #     wire_start, wire_end = get_start_end_positions(board, wire_num)
    #
    #     seed = random.randint(0, 10000)
    #
    #     key = jax.random.PRNGKey(seed)
    #
    #     board1 = thin_wire(key, board, wire_start, wire_end, wire_num)
    #     board2 = optimise_wire(key, board, wire_num)
    #
    #     print(f'Board after thinning wire {1}: \n {board1} \n')
    #     print(f'Board after optimising wire {1}: \n {board2} \n')
    #
    # board_2 = jnp.array([[4, 4, 4, 4, 4, 4],
    #                      [4, 3, 6, 4, 4, 4],
    #                      [5, 1, 1, 1, 1, 2]], dtype=jnp.int32)
    #
    # print(f'Original board 3: \n {board_2} \n')
    #
    # for i in list(range(2)):
    #     wire_num = i
    #     wire_start, wire_end = get_start_end_positions(board_2, wire_num)
    #     seed = random.randint(0, 10000)
    #     key = jax.random.PRNGKey(seed)
    #     board_2 = thin_wire(key, board_2, wire_start, wire_end, wire_num)
    #     print(f'Board after thinning wire {i + 1}: \n {board_2} \n')

    randy_board = jnp.array([[6, 7, 7, 7, 7, 7, 7, 7, 2, 1],
                             [4, 7, 0, 0, 0, 0, 0, 7, 0, 1],
                             [5, 7, 0, 0, 9, 0, 7, 7, 1, 1],
                             [7, 7, 0, 7, 7, 0, 7, 1, 1, 0],
                             [7, 0, 7, 7, 0, 0, 7, 1, 0, 0],
                             [7, 0, 7, 0, 7, 7, 7, 1, 0, 0],
                             [8, 0, 7, 7, 7, 0, 0, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 10, 10, 10, 13, 14, 3, 0, 1],
                             [0, 11, 10, 0, 12, 15, 0, 1, 1, 1]], dtype=jnp.int32)

    num_wires = jnp.max(randy_board) // 3

    print(f'Original board 4: \n {randy_board} \n')

    for i in range(num_wires):
        wire_num = i
        # wire_start, wire_end = get_start_end_positions(randy_board, wire_num)
        seed = random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)
        # randy_board = thin_wire(key, randy_board, wire_start, wire_end, wire_num)
        randy_board = optimise_wire(key, randy_board, wire_num)
        print(f'Board after optimising wire {i + 1}: \n {randy_board} \n')

        # print(f'Board after thinning wire {i + 1}: \n {randy_board} \n')

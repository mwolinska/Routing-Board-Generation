from typing import Tuple

import chex
import jax
from chex import Array, Numeric, PRNGKey
from jumanji.environments.routing.connector.constants import DOWN, UP, LEFT, \
    RIGHT, NOOP, POSITION, PATH, TARGET, EMPTY

import jax.numpy as jnp
from jumanji.environments.routing.connector.utils import move_position, \
    move_agent, get_agent_grid, get_correction_mask, get_target, \
    connected_or_blocked, is_valid_position, get_position

from jumanji.environments.routing.connector.types import Agent


class ParallelRandomWalk:
    def __init__(self, rows, cols, num_agents):
        self.cols = cols
        self.rows = rows
        self.num_agents = num_agents

    def initialise_agents(self, key: PRNGKey, grid: Array):
        starts_flat = jax.random.choice(
            key=key,
            a=jnp.arange(self.rows * self.cols),
            shape=(1, self.num_agents),
            # Start and target positions for all agents
            replace=False,  # Start and target positions cannot overlap
        )

        # Create 2D points from the flat arrays.
        starts = jnp.divmod(starts_flat[0], self.rows)
        # targets = jnp.divmod(targets_flat, grid_size)
        targets = tuple(jnp.full((2, self.num_agents), -1))

        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )
        grid = jax.vmap(self.update_grid, in_axes=(None, 0))(grid, agents)
        grid = grid.max(axis=0)
        return grid, agents

    def update_grid(self, grid, agent: Agent):
        return grid.at[agent.start[0], agent.start[1]].set(get_position(agent.id))

    def generate_board(self, key: PRNGKey):
        grid = self.return_blank_board()
        key, step_key = jax.random.split(key)
        grid, agents = self.initialise_agents(key, grid)

        stepping_tuple = (key, grid, agents)

        _, grid, agents = jax.lax.while_loop(self.continue_stepping, self.step, stepping_tuple)

        heads = tuple(agents.start.T)
        targets = tuple(agents.position.T)

        return heads, targets, grid
        # self.step(step_key, grid, agents)

    def continue_stepping(self, stepping_tuple: Tuple[PRNGKey, Array, Agent]) -> bool:
        key, grid, agents = stepping_tuple
        dones = jax.vmap(self.blocked, in_axes=(None, 0))(grid, agents)
        return ~dones.all()

    def blocked(self, grid, agent):
        cell = self.convert_tuple_to_int(agent.position)
        return (self.available_cells(grid, cell) == -1).all()

    def return_blank_board(self) -> chex.Array:
        return jnp.zeros((self.rows, self.cols), dtype=int)

    def step(self, stepping_tuple: Tuple[PRNGKey, Array, Agent]):
        key, grid, agents = stepping_tuple
        agents, grid = self._step_agents(key, grid, agents)
        key, next_key = jax.random.split(key)
        return next_key, grid, agents

    def _step_agents(
            self, key: chex.PRNGKey, grid: Array, agents: Agent,
    ) -> Tuple[Agent, chex.Array]:
        """Steps all agents at the same time correcting for possible collisions.

        If a collision occurs we place the agent with the lower `agent_id` in its previous position.

        Returns:
            Tuple: (agents, grid) after having applied each agents' action
        """
        agent_ids = jnp.arange(self.num_agents)
        keys = jax.random.split(key, num=self.num_agents)

        actions = jax.vmap(self.select_action, in_axes=(0, None, 0))(keys, grid, agents)
        # Step all agents at the same time (separately) and return all of the grids
        agents, grids = jax.vmap(self._step_agent, in_axes=(0, None, 0))(
            agents, grid, actions
        )

        # Get grids with only values related to a single agent.
        # For example: remove all other agents from agent 1's grid. Do this for all agents.
        agent_grids = jax.vmap(get_agent_grid)(agent_ids, grids)
        joined_grid = jnp.max(agent_grids, 0)  # join the grids

        # Create a correction mask for possible collisions (see the docs of `get_correction_mask`)
        correction_fn = jax.vmap(get_correction_mask, in_axes=(None, None, 0))
        correction_masks, collided_agents = correction_fn(
            grid, joined_grid, agent_ids
        )
        correction_mask = jnp.sum(correction_masks, 0)

        # Correct state.agents
        # Get the correct agents, either old agents (if collision) or new agents if no collision
        agents = jax.vmap(
            lambda collided, old_agent, new_agent: jax.lax.cond(
                collided,
                lambda: old_agent,
                lambda: new_agent,
            )
        )(collided_agents, agents, agents)
        # Create the new grid by fixing old one with correction mask and adding the obstacles
        return agents, joined_grid + correction_mask

    def select_action(self, key: PRNGKey, grid: Array, agent: Agent):
        cell = self.convert_tuple_to_int(agent.position)
        available_cells = self.available_cells(grid=grid, cell=cell)
        step_coordinate_flat = jax.random.choice(
            key=key,
            a=available_cells,
            shape=(),
            replace=True,
            p=available_cells != -1,
        )
        # coordinate = jnp.divmod(step_coordinate_flat, self.rows)

        action = self.action_from_positions(cell, step_coordinate_flat)
        return action

    def convert_int_to_tuple(self, position: int) -> Tuple[int, int]:
        return jnp.array((position // self.cols), int(position % self.cols), int)

    def convert_tuple_to_int(self, position: Array) -> int:
        return jnp.array((position[0] * self.cols + position[1]), int)

    def action_from_positions(self, position_1: int, position_2: int) -> int:
        position_1 = self.convert_int_to_array(position_1)
        position_2 = self.convert_int_to_array(position_2)
        action_tuple = position_2 - position_1
        return self.action_from_tuple(action_tuple)

    def action_from_tuple(self, action_tuple: Array) -> int:
        # all_actions = jnp.zeros(self.num_agents)
        action_multiplier = jnp.array([UP, DOWN, LEFT, RIGHT, NOOP])
        actions = jnp.array(
            [
                (action_tuple == jnp.array([-1, 0])).all(axis=0),
                (action_tuple == jnp.array([1, 0])).all(axis=0),
                (action_tuple == jnp.array([0, -1])).all(axis=0),
                (action_tuple == jnp.array([0, 1])).all(axis=0),
                (action_tuple == jnp.array([0, 0])).all(axis=0),
            ]
        )
        actions = jnp.sum(actions* action_multiplier, axis=0)
        return actions

    def convert_int_to_array(self, position: int) -> Array:
        return jnp.array([position // self.cols, position % self.cols], int)

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
        direction_operations = jnp.array([-1 * self.rows, self.rows, -1, 1])
        # Create a mask to check 0 <= index < total size
        cells_to_check = available_moves + direction_operations
        is_id_in_grid = cells_to_check < self.rows * self.cols
        is_id_positive = 0 <= cells_to_check
        mask = is_id_positive & is_id_in_grid

        # Ensure adjacent cells doesn't involve going off the grid
        unflatten_available = jnp.divmod(cells_to_check, self.rows)
        unflatten_current = jnp.divmod(cell, self.rows)
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
        value = grid[jnp.divmod(cell, self.rows)]
        wire_id = (value - 1) // 3

        _, available_cells_mask = jax.lax.scan(self.is_cell_free, grid, adjacent_cells)
        # Also want to check if the cell is touching itself more than once
        _, touching_cells_mask = jax.lax.scan(self.is_cell_touching_self, (grid, wire_id), adjacent_cells)
        available_cells_mask = available_cells_mask & touching_cells_mask
        available_cells = jnp.where(available_cells_mask == 0, -1, adjacent_cells)
        # print("avail cells")
        # print(grid)
        # print(adjacent_cells)
        return available_cells

    def is_cell_free(
            self, grid: chex.Array, cell: int,
    ) -> Tuple[chex.Array, bool]:
        """Check if a given cell is free, i.e. has a value of 0.

        Args:
            grid: the current grid of the board.
            cell: the flat index of the cell to check.

        Returns:
            A tuple of the new grid and a boolean indicating whether the cell is free or not.
        """
        coordinate = jnp.divmod(cell, self.rows)
        return grid, jax.lax.select(cell == -1, False, grid[coordinate[0], coordinate[1]] == 0)

    def is_cell_touching_self(
            self, grid_wire_id: Tuple[chex.Array, int], cell: int,
    ) -> Tuple[Tuple[chex.Array, int], bool]:
        """Check if the cell is touching any of the wire's own cells more than once.
        This means looking for surrounding cells of value 3 * wire_id + POSITION or
        3 * wire_id + PATH.
        # TODO (MW/ OJ): Why 3?
        """
        grid, wire_id = grid_wire_id
        # Get the adjacent cells of the current cell
        adjacent_cells = self.adjacent_cells(cell)
        def is_cell_touching_self_inner(grid, cell):
            coordinate = jnp.divmod(cell, self.rows)
            cell_value = grid[coordinate[0], coordinate[1]]
            touching_self = jnp.logical_or(
                jnp.logical_or(cell_value == 3 * wire_id + POSITION, cell_value == 3 * wire_id + PATH),
                cell_value == 3 * wire_id + TARGET,
            )
            return grid, jnp.where(cell == -1, False, touching_self)

        # Count the number of adjacent cells with the same wire id
        _, touching_self_mask = jax.lax.scan(is_cell_touching_self_inner, grid, adjacent_cells)
        # If the cell is touching itself more than once, return False
        return (grid, wire_id), jnp.where(jnp.sum(touching_self_mask) > 1, False, True)


    def _step_agent(
        self, agent: Agent, grid: Array, action: int,
    ) -> Tuple[Agent, Array]:
        """Moves the agent according to the given action if it is possible.

        Returns:
            Tuple: (agent, grid) after having applied the given action.
        """
        new_pos = move_position(agent.position, action)

        new_agent, new_grid = jax.lax.cond(
            self.is_valid_position_rw(grid, agent, new_pos) & (action != NOOP),
            move_agent,
            lambda *_: (agent, grid),
            agent,
            grid,
            new_pos,
        )

        return new_agent, new_grid

    def is_valid_position_rw(self,
        grid: Array, agent: Agent, position: Array,
    ) -> Array:
        """Checks to see if the specified agent can move to `position`.

        Args:
            grid: the environment state's grid.
            agent: the agent.
            position: the new position for the agent.

        Returns:
            bool: True if the agent moving to position is valid.
        """
        row, col = position
        grid_size = grid.shape[0]

        # Within the bounds of the grid
        in_bounds = (0 <= row) & (row < grid_size) & (0 <= col) & (col < grid_size)
        # Cell is not occupied
        open_cell = (grid[row, col] == EMPTY) | (
                    grid[row, col] == get_target(agent.id))
        # Agent is not connected
        not_connected = ~agent.connected

        return in_bounds & open_cell & not_connected

    def _get_action_mask(self, agent: Agent, grid: chex.Array) -> chex.Array:
        """Gets an agent's action mask."""
        # Don't check action 0 because no-op is always valid
        actions = jnp.arange(1, 5)

        def is_valid_action(action: int) -> chex.Array:
            agent_pos = move_position(agent.position, action)
            return is_valid_position(grid, agent, agent_pos)

        mask = jnp.ones(5, dtype=bool)
        mask = mask.at[actions].set(jax.vmap(is_valid_action)(actions))
        return mask

if __name__ == '__main__':
    board_generator = ParallelRandomWalk(10, 10, 5)
    key = jax.random.PRNGKey(0)
    # grid = board_generator.return_blank_board()
    # continue_stepping = True
    # grid, agents = board_generator.initialise_agents(key, grid)
    # key, step_key = jax.random.split(key)
    # stepping_tuple = (step_key, grid, agents)
    #
    # while continue_stepping:
    #     stepping_tuple = board_generator.step(stepping_tuple)
    #     continue_stepping = board_generator.continue_stepping(stepping_tuple)
    #     print(stepping_tuple[1])


    # board.
    # board_generator.generate_board(key)
    board_generator_jit = jax.jit(board_generator.generate_board)
    heads, targets, grid = board_generator_jit(key)
    print(grid)
    # grid_size = 4
    # num_agents = 3
    # grid = jnp.array(
    #     [
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #     ]
    # )
    # cell = 9
    # agents = Agent(
    #     id=jnp.array([]),
    #     start=jnp.array([[3, 0, 0], [3, 0, 1]]),
    #     target=jnp.array([[-1, -1, -1], [-1, -1, -1]]),
    #     position=jnp.array([[3, 0, 0], [3, 0, 1]]),
    # )

    # starts_flat, targets_flat = jax.random.choice(
    #     key=key,
    #     a=jnp.arange(grid_size ** 2),
    #     shape=(2, num_agents),
    #     # Start and target positions for all agents
    #     replace=False,  # Start and target positions cannot overlap
    # )
    #
    # # Create 2D points from the flat arrays.
    # starts = jnp.divmod(starts_flat, grid_size)
    # targets = jnp.divmod(targets_flat, grid_size)
    # agents = jax.vmap(Agent)(
    #     id=jnp.arange(3),
    #     start=jnp.stack(starts, axis=1),
    #     target=jnp.stack(targets, axis=1),
    #     position=jnp.stack(starts, axis=1),
    # )

    # print(board.generate_board(key))

    # board._step_agents(key, grid, agents)
    # print(board.select_action(key, grid, agent))

    # print(board.action_from_positions(10, 0))

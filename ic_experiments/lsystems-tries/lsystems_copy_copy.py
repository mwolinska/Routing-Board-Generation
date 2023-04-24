import jax.numpy as np
from jax import random
import jax
from ic_routing_board_generation.board_generator.jax_data_model.deque import (
    Agent, 
    create_stack, 
    stack_push_head, 
    stack_push_tail, 
    stack_pop_head, 
    stack_pop_tail)

from typing import List, NamedTuple, SupportsFloat as Numeric, Tuple
import grid_utils
from post_processing_utils import extend_wires_jax
from functools import partial


# class Connector(Environment[State]):
#     """The `Connector` environment is a multi-agent gridworld problem where each agent must connect a
#     start to a target. However, when moving through this gridworld the agent leaves an impassable
#     trail behind it. Therefore, agents must connect to their targets without overlapping the routes
#     taken by any other agent.

#     - observation - `Observation`
#         - action mask: jax array (bool) of shape (num_agents, 5).
#         - step_count: jax array (int32) of shape ()
#             the current episode step.
#         - grid: jax array (int32) of shape (num_agents, size, size)
#             - each 2d array (size, size) along axis 0 is the agent's local observation.
#             - agents have ids from 0 to (num_agents - 1)
#             - with 2 agents you might have a grid like this:
#               4 0 1
#               5 0 1
#               6 3 2
#               which means agent 1 has moved from the top right of the grid down and is currently in
#               the bottom right corner and is aiming to get to the middle bottom cell. Agent 2
#               started in the top left and moved down once towards its target in the bottom left.

#               This would just be agent 0's view, the numbers would be flipped for agent 1's view.
#               So the full observation would be of shape (2, 3, 3).

#     - action: jax array (int32) of shape (num_agents,):
#         - can take the values [0,1,2,3,4] which correspond to [No Op, Up, Right, Down, Left].
#         - each value in the array corresponds to an agent's action.

#     - reward: jax array (float) of shape ():
#         - dense: each agent is given 1.0 if it connects on that step, otherwise 0.0. Additionally,
#             each agent that has not connected receives a penalty reward of -0.03.

#     - episode termination: if an agent can't move, or the time limit is reached, or the agent
#         connects to its target, it is considered done. Once all agents are done, the episode
#         terminates. The timestep discounts are of shape (num_agents,).

#     - state: State:
#         - key: jax PRNG key used to randomly spawn agents and targets.
#         - grid: jax array (int32) of shape (size, size) which corresponds to agent 0's observation.
#         - step_count: jax array (int32) of shape () number of steps elapsed in the current episode.

#     """

#     def __init__(
#         self,
#         generator: Optional[Generator] = None,
#         reward_fn: Optional[RewardFn] = None,
#         time_limit: int = 50,
#         viewer: Optional[Viewer[State]] = None,
#     ) -> None:
#         """Create the `Connector` environment.

#         Args:
#             generator: `Generator` whose `__call__` instantiates an environment instance.
#                 Implemented options are [`UniformRandomGenerator`].
#                 Defaults to `UniformRandomGenerator` with `grid_size=10` and `num_agents=5`.
#             reward_fn: class of type `RewardFn`, whose `__call__` is used as a reward function.
#                 Implemented options are [`DenseRewardFn`]. Defaults to `DenseRewardFn`.
#             time_limit: the number of steps allowed before an episode terminates. Defaults to 50.
#             viewer: `Viewer` used for rendering. Defaults to `ConnectorViewer` with "human" render
#                 mode.
#         """
#         self._generator = generator or UniformRandomGenerator(
#             grid_size=10, num_agents=5
#         )
#         self.time_limit = time_limit
#         self.num_agents = self._generator.num_agents
#         self.grid_size = self._generator.grid_size
#         self._agent_ids = jnp.arange(self.num_agents)

#     def step(
#         self, state: State, action: chex.Array
#     ) -> Tuple[State, TimeStep[Observation]]:
#         """Perform an environment step.

#         Args:
#             state: State object containing the dynamics of the environment.
#             action: Array containing the actions to take for each agent.
#                 - 0 no op
#                 - 1 move up
#                 - 2 move right
#                 - 3 move down
#                 - 4 move left

#         Returns:
#             state: `State` object corresponding to the next state of the environment.
#             timestep: `TimeStep` object corresponding the timestep returned by the environment.
#         """
#         agents, grid = self._step_agents(state, action)
#         new_state = State(
#             grid=grid, step_count=state.step_count + 1, agents=agents, key=state.key
#         )

#         # Construct timestep: get observations, rewards, discounts
#         grids = self._obs_from_grid(grid)
#         reward = self._reward_fn(state, action, new_state)
#         action_mask = jax.vmap(self._get_action_mask, (0, None))(agents, grid)
#         observation = Observation(
#             grid=grids, action_mask=action_mask, step_count=new_state.step_count
#         )

#         dones = jax.vmap(connected_or_blocked)(agents, action_mask)
#         discount = jnp.asarray(jnp.logical_not(dones), dtype=float)
#         extras = self._get_extras(new_state)
#         timestep = jax.lax.cond(
#             dones.all() | (new_state.step_count >= self.time_limit),
#             lambda: termination(
#                 reward=reward,
#                 observation=observation,
#                 extras=extras,
#                 shape=self.num_agents,
#             ),
#             lambda: transition(
#                 reward=reward,
#                 observation=observation,
#                 discount=discount,
#                 extras=extras,
#                 shape=self.num_agents,
#             ),
#         )

#         return new_state, timestep

#     def _step_agents(self, state: State, action: chex.Array) -> Tuple[Agent, chex.Array]:
#         """Steps all agents at the same time correcting for possible collisions.

#         If a collision occurs we place the agent with the lower `agent_id` in its previous position.

#         Returns:
#             Tuple: (agents, grid) after having applied each agents' action
#         """
#         agent_ids = jnp.arange(self.num_agents)
#         # Step all agents at the same time (separately) and return all of the grids
#         agents, grids = jax.vmap(self._step_agent, in_axes=(0, None, 0))(state.agents, state.grid, action)

#         # Get grids with only values related to a single agent.
#         # For example: remove all other agents from agent 1's grid. Do this for all agents.
#         agent_grids = jax.vmap(get_agent_grid)(agent_ids, grids)
#         joined_grid = jnp.max(agent_grids, 0)  # join the grids

#         # Create a correction mask for possible collisions (see the docs of `get_correction_mask`)
#         correction_fn = jax.vmap(get_correction_mask, in_axes=(None, None, 0))
#         correction_masks, collided_agents = correction_fn(
#             state.grid, joined_grid, agent_ids
#         )
#         correction_mask = jnp.sum(correction_masks, 0)

#         # Correct state.agents
#         # Get the correct agents, either old agents (if collision) or new agents if no collision
#         agents = jax.vmap(
#             lambda collided, old_agent, new_agent: jax.lax.cond(
#                 collided,
#                 lambda: old_agent,
#                 lambda: new_agent,
#             )
#         )(collided_agents, state.agents, agents)
#         # Create the new grid by fixing old one with correction mask and adding the obstacles
#         return agents, joined_grid + correction_mask

#     def _step_agent(
#         self, agent: Agent, grid: chex.Array, action: chex.Numeric
#     ) -> Tuple[Agent, chex.Array]:
#         """Moves the agent according to the given action if it is possible.

#         Returns:
#             Tuple: (agent, grid) after having applied the given action.
#         """
#         new_pos = move_position(agent.position, action)

#         new_agent, new_grid = jax.lax.cond(
#             is_valid_position(grid, agent, new_pos) & (action != NOOP),
#             move_agent,
#             lambda *_: (agent, grid),
#             agent,
#             grid,
#             new_pos)
#         return new_agent, new_grid

#     def _obs_from_grid(self, grid: chex.Array) -> chex.Array:
#         """Gets the observation vector for all agents."""
#         return jax.vmap(switch_perspective, (None, 0, None))(
#             grid, self._agent_ids, self.num_agents
#         )

#     def _get_action_mask(self, agent: Agent, grid: chex.Array) -> chex.Array:
#         """Gets an agent's action mask."""
#         # Don't check action 0 because no-op is always valid
#         actions = jnp.arange(1, 5)

#         def is_valid_action(action: int) -> chex.Array:
#             agent_pos = move_position(agent.position, action)
#             return is_valid_position(grid, agent, agent_pos)

#         mask = jnp.ones(5, dtype=bool)
#         mask = mask.at[actions].set(jax.vmap(is_valid_action)(actions))
#         return mask

#     def observation_spec(self) -> specs.Spec[Observation]:
#         """Specifications of the observation of the `Connector` environment.

#         Returns:
#             Spec for the `Observation` whose fields are:
#             - grid: BoundedArray (int32) of shape (num_agents, grid_size, grid_size).
#             - action_mask: BoundedArray (bool) of shape (num_agents, 5).
#             - step_count: BoundedArray (int32) of shape ().
#         """
#         grid = specs.BoundedArray(
#             shape=(self.num_agents, self.grid_size, self.grid_size),
#             dtype=jnp.int32,
#             name="grid",
#             minimum=0,
#             maximum=self.num_agents * 3 + AGENT_INITIAL_VALUE,
#         )
#         action_mask = specs.BoundedArray(
#             shape=(self.num_agents, 5),
#             dtype=bool,
#             minimum=False,
#             maximum=True,
#             name="action_mask",
#         )
#         step_count = specs.BoundedArray(
#             shape=(),
#             dtype=jnp.int32,
#             minimum=0,
#             maximum=self.time_limit,
#             name="step_count",
#         )
#         return specs.Spec(
#             Observation,
#             "ObservationSpec",
#             grid=grid,
#             action_mask=action_mask,
#             step_count=step_count,
#         )

#     def action_spec(self) -> specs.MultiDiscreteArray:
#         """Returns the action spec for the Connector environment.

#         5 actions: [0,1,2,3,4] -> [No Op, Up, Right, Down, Left]. Since this is a multi-agent
#         environment, the environment expects an array of actions of shape (num_agents,).

#         Returns:
#             observation_spec: `MultiDiscreteArray` of shape (num_agents,).
#         """
#         return specs.MultiDiscreteArray(
#             num_values=jnp.array([5] * self.num_agents),
#             dtype=jnp.int32,
#             name="action",
#         )


class LSystemBoardGen:
    def __init__(self,
                 rows: int,
                 cols: int = None,
                 num_agents: int = None,
                 n_iters: int = 5,
                 pushpullnone_ratios: Tuple[Numeric] = (2, 0.5, 1)) -> None:

        self.rows = rows
        self.cols = cols
        self.max_size = rows*cols
        self.num_agents = num_agents
        self.n_iters = n_iters
        self.pushpullnone_ratios = pushpullnone_ratios

    def generate_indices(self,
                         key: random.PRNGKey) -> List[Agent]:
        """Create a list of agents with random initial starting and ending (as neighbours) locations."""
        row_indices, col_indices = jax.numpy.meshgrid(jax.numpy.arange(1, self.rows, 2), jax.numpy.arange(1, self.cols, 2), indexing='ij')
        index_choice = jax.numpy.stack((row_indices, col_indices), axis=-1).reshape(-1, 2)
        head_indices = jax.random.choice(key, index_choice, (self.num_agents,), replace=False)
        randomness_type = jax.random.randint(key, (), 0, 2)
        offset_array = jax.lax.select(randomness_type == 0, jax.numpy.array([[0, 1], [1, 0]]), jax.numpy.array([[1, 0], [0, 1]]))
        tail_offsets = jax.random.choice(key, offset_array, (self.num_agents,))
        tail_indices = head_indices + tail_offsets
        return head_indices, tail_indices

    def initialise_starting_board_for_one_agent(self,
                                                agent_num: int,
                                                head_indices: jax.Array,
                                                tail_indices: jax.Array) -> List[Agent]:
        """Create a list of agents with random initial starting and ending (as neighbours) locations."""
        new_agent = create_stack(self.max_size, 2)
        new_agent = stack_push_head(new_agent, head_indices[agent_num])
        new_agent = stack_push_tail(new_agent, tail_indices[agent_num])
        return new_agent

    def initialise_starting_board(self, key):
        head_indices, tail_indices = self.generate_indices(key)
        agents = jax.vmap(self.initialise_starting_board_for_one_agent, in_axes=(0, None, None))(
            jax.numpy.arange(self.num_agents), head_indices, tail_indices)
        grid = self.fill_solved_board(agents)
        return agents, grid

    def heads_or_tails(self, key: random.PRNGKey) -> int:
        """Given `key`choose a random growth or shrink direction for an Agent."""
        return random.randint(key, (), -1, 1)

    def find_empty_neighbours(self, loc: Tuple[int, int], board) -> np.ndarray:
        """Given a location tuple, find the empty neighbours of a board location.
        
        loc: a tuple of shape 1x2.
        returns: a boolean array of shape 1x4.
        """
        i, j = loc
        neighbors = []
        neighbour_locs = [(i-1,j), (i,j+1), (i+1,j), (i,j-1)]

        v1 = jax.lax.select(i != 0, np.equal(board[i - 1, j], 0), False)                 # north
        v2 = jax.lax.select(j != self.cols - 1, np.equal(board[i, j + 1], 0), False)     # east
        v3 = jax.lax.select(i != self.rows - 1, np.equal(board[i + 1, j], 0), False)     # south
        v4 = jax.lax.select(j != 0, np.equal(board[i, j - 1], 0), False)                 # west
        
        neighbors = [v1, v2, v3, v4]
        return np.asarray(neighbors, dtype=int), np.asarray(neighbour_locs)

    def growth_neighbour_loc(self, key: random.PRNGKey, neighbour_locs: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
        """Chooses the location of the next growth of an Agent according to free space in neighbouring cells."""
        which_neighbour = jax.lax.select(neighbours.sum()>0, jax.random.choice(key, 4, (), p=neighbours), -1)
        loc = jax.lax.select(which_neighbour >= 0, neighbour_locs[which_neighbour], np.asarray((-1, -1)))
        return loc

    def grow_agent(self, side:int, loc: np.ndarray, agent: Agent) -> Agent:
        """Given a location tuple and an Agent, grow the Agent in the given direction."""        
        grown_agent = jax.lax.cond(side == 0, lambda _: stack_push_head(agent, loc), lambda _: stack_push_tail(agent, loc), None)
        return grown_agent

    def push(self, key: random.PRNGKey, board: jax.Array, agent: Agent) -> Agent:
        """Pushes (grows) an Agent in a random direction."""
        side = self.heads_or_tails(key)                                                                                                     # stochastic
        pointer = jax.lax.select(side == 0, (agent.head_insertion_index-1)%self.max_size, (agent.tail_insertion_index+1)%self.max_size)     # deterministic
        loc = agent.data[pointer]
        neighbours, neighbour_locs = self.find_empty_neighbours(loc, board)                                                                 # deterministic
        loc = self.growth_neighbour_loc(key, neighbour_locs, neighbours)        
        agent = jax.lax.cond(loc[0] >= 0, lambda _: self.grow_agent(side, loc, agent), lambda _: agent, None)                               # deterministic
        return agent

    def shrink_agent(self, side: int, agent: Agent) -> Agent:
        """Given a direction and an Agent, shrink the Agent in the given direction."""
        agent = jax.lax.cond(side == 0, stack_pop_head, stack_pop_tail, agent)
        return agent

    def do_nothing(self, side: int, agent: Agent) -> Agent:
        return agent

    def pull(self, key:random.PRNGKey, board, agent: Agent) -> Agent:
        """Pulls (shrinks) the agent in a random direction."""
        side = self.heads_or_tails(key)                                                                                                 # stochastic
        agent_length = np.count_nonzero(agent.data[:,0]+1)   
        agent = jax.lax.cond(agent_length > 1, self.shrink_agent, self.do_nothing, side, agent)
        return agent

    def fill(self,
             key: random.PRNGKey,
             agents: Agent,
             board: jax.Array) -> Agent:
        return self.fill_method((key, agents, board))

    def are_agents_good(self, inp) -> bool:
        _, agents = inp
        is_board_good = np.asarray([np.count_nonzero(agent.data[:,0]+1)>1 for agent in agents])
        all_good = np.all(is_board_good)
        return ~all_good

    def fill_method_inner_inner(self, key, agent, board) -> Agent:
        key, _ = random.split(key)
        action = jax.random.choice(key, 3, (), p=np.asarray(self.pushpullnone_ratios))
        return jax.lax.cond(action == 0, self.push, self.pull, key, board, agent)

    def fill_method_inner(self, i, inp) -> Agent:
        key, agents, board = inp
        return jax.vmap(self.fill_method_inner_inner, in_axes=(None, None, 0, None))(key, agents, board)

    def fill_method(self, inp) -> None:
        return jax.lax.fori_loop(0, self.n_iters, self.fill_method_inner, inp)


    # ------------------- Board Filling ------------------- #
    # everything below here is OK.


    def fill_unsolved_board(self, agents: Agent) -> np.ndarray:
        """Fills an `unsolved` board with Agents."""
        board = np.zeros((self.rows, self.cols))
        
        for i in range(len(agents)):
            for j in range(self.max_size):
                left, middle, right = self.is_empty_self_and_neighbours(j, agents, i)
                board = jax.lax.select(self.is_head(left, middle, right), board.at[tuple(agents.data[i][j])].set(3*i+2), board)
                board = jax.lax.select(self.is_tail(left, middle, right), board.at[tuple(agents.data[i][j])].set(3*i+3), board)
        return np.asarray(board, dtype=int)
    
    def is_empty_self_and_neighbours(self, idx: np.ndarray, agent: Agent, i: int) -> bool:
        """Tests if a location and its neighbours are empty."""
        return np.all(-1 == agent.data[i][(idx-1)%self.max_size]), np.all(-1 == agent.data[i][idx]), np.all(-1 == agent.data[i][(idx+1)%self.max_size])

    def is_body(self, left, mid, right) -> bool:
        """Tests if a location is a body."""
        return ~ (left | mid | right)

    def is_head(self, left, mid, right) -> bool:
        """Tests if a location is a head."""
        return (~(left | mid)) & right

    def is_tail(self, left, mid, right) -> bool:
        """Tests if a location is a tail."""
        return left & (~ (mid | right))

    def fill_solved_board(self, agents: Agent) -> np.ndarray:
        """Fills a `solved` board with Agents."""
        board = np.zeros((self.rows, self.cols))
        
        for i in range(len(agents)):
            for j in range(self.max_size):
                left, middle, right = self.is_empty_self_and_neighbours(j, agents, i)
                board = jax.lax.select(self.is_body(left, middle, right), board.at[tuple(agents.data[i][j])].set(3*i+1), board)
                board = jax.lax.select(self.is_head(left, middle, right), board.at[tuple(agents.data[i][j])].set(3*i+2), board)
                board = jax.lax.select(self.is_tail(left, middle, right), board.at[tuple(agents.data[i][j])].set(3*i+3), board)
        return np.asarray(board, dtype=int)

    def return_training_board(self, key: random.PRNGKey) -> np.ndarray:
        agents = self.initialise_starting_board(key)
        modified_agents = self.fill(key, agents, n_iters=5, pushpullnone_ratios=[2, 0.5, 1])
        self.agents = modified_agents
        return self.fill_unsolved_board(modified_agents)
    
    def return_solved_board(self) -> np.ndarray:
        return np.asarray(self.fill_solved_board(self.agents), int)

    def declutter(self, key: random.PRNGKey, wire_num: int, agents: List[Agent]):
        agent       = agents[wire_num]
        wire_start  = agent.data[(agent.head_insertion_index - 1)%self.max_size]
        wire_end    = agent.data[(agent.tail_insertion_index + 1)%self.max_size]
        key, subkey = jax.random.split(key)
        return grid_utils.thin_wire(key, self.fill_solved_board(agents), wire_start, wire_end, wire_num)
    
    def wire_extend(self, key: random.PRNGKey, agents: List[Agent]) -> jax.Array:
        """Method to extend all wires in the board."""
        return extend_wires_jax(self.fill_solved_board(agents), key)

import time
if __name__ == '__main__':
    
    # Example usage
    key = random.PRNGKey(int(time.time()))
    board = LSystemBoardGen(rows=5, cols=5, num_agents=5)
    agents = board.initialise_starting_board(key)
    new_agents = board.fill(key, agents, n_iters=5, pushpullnone_ratios=[2, 0.5, 1])
    print('board after fill\n', board.fill_solved_board(new_agents))

    # alternatively, select an individual agent to push or pull
    
    # # ie) this below code works
    # for _ in range(5):
    #     for i in range(5):
    #         which_agent = random.randint(key, (1,), 0, 5)[0]
    #         print('pushing agent: ', which_agent)
    #         agents[which_agent] = board.push(key, agents[which_agent], agents)
    #         key, _ = random.split(key)
    #         print('board after push\n', board.fill_solved_board(agents))

    #     which_agent = random.randint(key, (1,), 0, 2)[0]
    #     agents[which_agent] = board.pull(key, agents[which_agent], agents)
    #     key, _ = random.split(key)
    #     print('board after pull\n', board.fill_solved_board(agents))

    # is_board_good = np.asarray([np.count_nonzero(nagent.data[:,0]+1)>1 for nagent in agents])
    # print('is board good? ', np.all(is_board_good))

    # alternatively, run a single command once
    # board = LSystemBoardGen(rows=5, cols=5, num_agents=5)
    # print('board after fill\n', board.return_training_board(key))

from dataclasses import dataclass
from functools import partial

from chex import PRNGKey, Array

# import random
from typing import Tuple
import jax.numpy as jnp
import jax
from jumanji.environments.routing.connector.constants import POSITION, TARGET, \
    PATH

from ic_routing_board_generation.board_generator.jax_data_model.wire import Wire, create_wire, \
    stack_push

STARTING_POSITION = POSITION  # My internal variable to disambiguate the word "position"
# Also available to import from constants NOOP, LEFT, LEFT, UP, RIGHT, DOWN

class ParallelRandomWalk:
    """The `Connector` environment is a multi-agent gridworld problem where each agent must connect a
    start to a target. However, when moving through this gridworld the agent leaves an impassable
    trail behind it. Therefore, agents must connect to their targets without overlapping the routes
    taken by any other agent.

    - observation - `Observation`
        - action mask: jax array (bool) of shape (num_agents, 5).
        - step_count: jax array (int32) of shape ()
            the current episode step.
        - grid: jax array (int32) of shape (num_agents, size, size)
            - each 2d array (size, size) along axis 0 is the agent's local observation.
            - agents have ids from 0 to (num_agents - 1)
            - with 2 agents you might have a grid like this:
              4 0 1
              5 0 1
              6 3 2
              which means agent 1 has moved from the top right of the grid down and is currently in
              the bottom right corner and is aiming to get to the middle bottom cell. Agent 2
              started in the top left and moved down once towards its target in the bottom left.

              This would just be agent 0's view, the numbers would be flipped for agent 1's view.
              So the full observation would be of shape (2, 3, 3).

    - action: jax array (int32) of shape (num_agents,):
        - can take the values [0,1,2,3,4] which correspond to [No Op, Up, Right, Down, Left].
        - each value in the array corresponds to an agent's action.

    - reward: jax array (float) of shape ():
        - dense: each agent is given 1.0 if it connects on that step, otherwise 0.0. Additionally,
            each agent that has not connected receives a penalty reward of -0.03.

    - episode termination: if an agent can't move, or the time limit is reached, or the agent
        connects to its target, it is considered done. Once all agents are done, the episode
        terminates. The timestep discounts are of shape (num_agents,).

    - state: State:
        - key: jax PRNG key used to randomly spawn agents and targets.
        - grid: jax array (int32) of shape (size, size) which corresponds to agent 0's observation.
        - step_count: jax array (int32) of shape () number of steps elapsed in the current episode.

    ```python
    from jumanji.environments import Connector
    env = Connector()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(self, rows: int, cols: int, num_agents: int = 3, key: PRNGKey = None):
        # super().__init__(rows, cols, num_agents)
        self._rows = rows
        self._cols = cols
        self._num_agents = num_agents
        self._key = key
        self.agents = jnp.arange(num_agents)
        self.grid = jnp.zeros((rows,cols, num_agents), dtype=jnp.int32)


    def pick_heads_and_targets(self):
        """Randomly pick starting positions and targets for each agent.
        Heads and targets must be next to each other and unique for each agent."""
        pass

    def pick_head_and_target(self):
        """Randomly pick starting position and target for an agent.
        Head and target must be next to each other and unique for each agent."""
        # Select a random position for the head
        head = jax.random.randint(self._key, (2,), 0, self._rows * self._cols)









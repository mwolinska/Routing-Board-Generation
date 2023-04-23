import chex
import jax
from jax import numpy as jnp

from ic_routing_board_generation.board_generator.jax_board_generation.board_generator_random_seed_rb import \
    RandomSeedBoard
from ic_routing_board_generation.ic_rl_training.online_generators.uniform_generator import \
    Generator
from jumanji.environments.routing.connector import State
from jumanji.environments.routing.connector.types import Agent
from jumanji.environments.routing.connector.utils import get_position, \
    get_target


class RandomSeedGenerator(Generator):
    """Randomly generates `Connector` grids that may or may not be solvable. This generator places
    start and target positions uniformly at random on the grid.
    """

    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Instantiates a `UniformRandomGenerator`.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)
        self.board_generator = RandomSeedBoard(grid_size, grid_size, num_agents)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
        key, pos_key = jax.random.split(key)

        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        starts, targets = self.board_generator.generate_starts_ends(key)
        # jax.debug.print("our starts: {x}", x=starts)
        # jax.debug.print("our targets: {x}", x=targets)
        agent_position_values = jax.vmap(get_position)(
            jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))
        # jax.debug.print("agent_position_values: {x}", x=agent_position_values)
        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)

        # Create the agent pytree that corresponds to the grid.

        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),

            position=jnp.stack(starts, axis=1),
        )

        step_count = jnp.array(0, jnp.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)

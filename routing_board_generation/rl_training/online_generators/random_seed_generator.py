import chex
import jax
from jax import numpy as jnp
from jumanji.environments.routing.connector import State
from jumanji.environments.routing.connector.types import Agent
from jumanji.environments.routing.connector.utils import get_position, get_target

from routing_board_generation.board_generation_methods.jax_implementation.board_generation.seed_extension import (
    SeedExtensionBoard,
)
from routing_board_generation.rl_training.online_generators.uniform_generator import (
    Generator,
)


class SeedExtensionGenerator(Generator):
    def __init__(self, grid_size: int, num_agents: int) -> None:
        """Instantiates a `SeedExtensionGenerator`.

        Args:
            grid_size: size of the square grid to generate.
            num_agents: number of agents/paths on the grid.
        """
        super().__init__(grid_size, num_agents)
        self.board_generator = SeedExtensionBoard(grid_size, grid_size, num_agents)
        self.board_generator_call = jax.jit(self.board_generator.generate_starts_ends)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
        key, pos_key = jax.random.split(key)

        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        starts, targets = self.board_generator_call(key)
        agent_position_values = jax.vmap(get_position)(jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

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

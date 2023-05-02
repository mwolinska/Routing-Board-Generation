### This file mirrors the agent class from types.py in Connector within the Jumanji package.
# The equality dunder method was overloaded and was added as a feature into a PR into Jumanji
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import chex
import jax.numpy as jnp

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

AgentT = TypeVar("AgentT", bound="Agent")


@dataclass
class Agent:
    """
    id: unique number representing only this agent.
    start: start position of this agent.
    target: goal position of this agent.
    position: the current position of this agent.
    """

    id: chex.Array  # ()
    start: chex.Array  # (2,)
    target: chex.Array  # (2,)
    position: chex.Array  # (2,)

    @property
    def connected(self) -> chex.Array:
        """returns: True if the agent has reached its target."""
        return jnp.all(self.position == self.target, axis=-1)

    def __eq__(self: AgentT, agent_2: Any) -> chex.Array:
        if not isinstance(agent_2, Agent):
            return NotImplemented
        same_ids = (agent_2.id == self.id).all()
        same_starts = (agent_2.start == self.start).all()
        same_targets = (agent_2.target == self.target).all()
        same_position = (agent_2.position == self.position).all()
        return same_ids & same_starts & same_targets & same_position

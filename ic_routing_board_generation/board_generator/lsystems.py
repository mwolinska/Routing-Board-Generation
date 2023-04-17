import jax.numpy as np
from jax import random
import jax
from ic_routing_board_generation.board_generator.deque import (
    Agent, 
    create_stack, 
    stack_push_head, 
    stack_push_tail, 
    stack_pop_head, 
    stack_pop_tail)

from typing import List, NamedTuple, SupportsFloat as Numeric


class LSystemBoardGen:
    def __init__(self,
                 rows: int,
                 cols: int = None,
                 num_agents: int = None) -> None:

        self.rows = rows
        self.cols = cols
        self.max_size = rows*cols
        self.num_agents = num_agents

    def initialise_starting_board(self,
                                  key: random.PRNGKey) -> List[Agent]:
        """Create a list of agents with random initial starting locations."""

        arr = np.zeros((self.rows, self.cols))
        numbers = np.arange(1, self.num_agents + 1)
        permuted_numbers = random.permutation(key, numbers)
        indices = random.permutation(key, np.arange(self.rows * self.cols))[:self.num_agents]
        row_indices, col_indices = np.unravel_index(indices, (self.rows, self.cols))
        arr = arr.at[row_indices, col_indices].set(permuted_numbers)

        agents = []
        for i in range(self.num_agents):
            new_agent = create_stack(self.max_size, 2)
            agents.append(stack_push_head(new_agent, np.asarray([row_indices[i], col_indices[i]]))) # initial coords given here

        return agents

    def heads_or_tails(self, key: random.PRNGKey) -> int:
        """Given `key`choose a random growth or shrink direction for an Agent."""
        return random.randint(key, (1,), -1, 1)[0]
    
    def find_empty_neighbours(self, loc: tuple, agents: list[Agent]) -> np.ndarray:
        """Given a location tuple, find the empty neighbours of a board location.
        
        loc: a tuple of shape 1x2.
        returns: a boolean array of shape 1x4.
        """
        i, j = loc
        neighbors = []
        neighbour_locs = [(i-1,j), (i,j+1), (i+1,j), (i,j-1)]
        board = self.fill_solved_board(agents)

        v1 = jax.lax.select(i - 1 > 0, np.equal(board[i - 1, j], 0), False)             # north
        v2 = jax.lax.select(j < self.cols - 1, np.equal(board[i, j + 1], 0), False)     # east
        v3 = jax.lax.select(i < self.rows - 1, np.equal(board[i + 1, j], 0), False)     # south
        v4 = jax.lax.select(j > 0, np.equal(board[i, j - 1], 0), False)                 # west
        
        neighbors = [v1, v2, v3, v4]
        return np.asarray(neighbors, dtype=int), np.asarray(neighbour_locs)

    def growth_neighbour_loc(self, key: random.PRNGKey, neighbour_locs: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
        """Chooses the location of the next growth of an Agent according to free space in neighbouring cells."""
        which_neighbour = jax.lax.select(neighbours.sum()>0, jax.random.choice(key, 4, (1,), p=neighbours)[0], -1)
        loc = jax.lax.select(which_neighbour >= 0, neighbour_locs[which_neighbour], np.asarray((-1, -1)))
        return loc
    
    def grow_agent(self, side:int, loc: np.ndarray, agent: Agent) -> Agent:
        """Given a location tuple and an Agent, grow the Agent in the given direction."""        
        grown_agent = jax.lax.cond(side == 0, lambda _: stack_push_head(agent, loc), lambda _: stack_push_tail(agent, loc), None)
        return grown_agent

    def push(self, key: random.PRNGKey, agent: Agent, agents: list[Agent]) -> Agent:
        """Pushes (grows) an Agent in a random direction."""
        side = self.heads_or_tails(key)                                                                                                 # stochastic
        pointer = jax.lax.select(side == 0, (agent.head_insertion_index-1)%self.max_size, (agent.tail_insertion_index+1)%self.max_size) # deterministic
        loc = agent.data[pointer]
        neighbours, neighbour_locs = self.find_empty_neighbours(loc, agents)                                                            # deterministic
        loc = self.growth_neighbour_loc(key, neighbour_locs, neighbours)                                                                # stochastic
        agent = jax.lax.cond(loc[0] >= 0, lambda _: self.grow_agent(side, loc, agent), lambda _: agent, None)                           # deterministic
        return agent

    def shrink_agent(self, side: int, agent: Agent) -> Agent:
        """Given a direction and an Agent, shrink the Agent in the given direction."""
        agent = jax.lax.cond(side == 0, stack_pop_head, stack_pop_tail, agent)
        return agent

    def do_nothing(self, side: int, agent: Agent) -> Agent:
        return agent

    def pull(self, key:random.PRNGKey, agent: Agent, agents: list[Agent]) -> Agent:
        """Pulls (shrinks) the agent in a random direction."""
        side = self.heads_or_tails(key)                                                                                                 # stochastic
        agent_length = np.count_nonzero(agent.data[:,0]+1)   
        agent = jax.lax.cond(agent_length > 1, self.shrink_agent, self.do_nothing, side, agent)
        return agent

    def fill(self,
             key: random.PRNGKey,
             agents: List[Agent],
             n_iters: int = 5,
             pushpullnone_ratios: List[Numeric] = [2, 1, 1]) -> None:
        
        for _ in range(n_iters):
            new_agents = []
            for agent in agents:
                key, subkey = random.split(key)
                action = jax.random.choice(key, 3, (1,), p=np.asarray(pushpullnone_ratios))[0]
                nagent = jax.lax.cond(action == 0, self.push, self.pull, key, agent, agents)
                new_agents.append(nagent)
            agents = new_agents
        
        is_board_good = np.asarray([np.count_nonzero(nagent.data[:,0]+1)>1 for nagent in new_agents])
        return jax.lax.select(np.all(is_board_good), new_agents, self.fill(key, agents, n_iters, pushpullnone_ratios))

    def fill_unsolved_board(self, agents: Agent) -> np.ndarray:
        """Fills an `unsolved` board with Agents."""
        board = np.zeros((self.rows, self.cols))
        
        for i in range(len(agents)):
            for j in range(len(agents[0].data)):
                left, middle, right = self.is_empty_self_and_neighbours(j, agents[i])
                board = jax.lax.select(self.is_head(left, middle, right), board.at[tuple(agents[i].data[j])].set(3*i+2), board)
                board = jax.lax.select(self.is_tail(left, middle, right), board.at[tuple(agents[i].data[j])].set(3*i+3), board)
        return board
    
    def is_empty_self_and_neighbours(self, idx: np.ndarray, agent: Agent) -> bool:
        """Tests if a location and its neighbours are empty."""
        return np.all(-1 == agent.data[idx-1]), np.all(-1 == agent.data[idx]), np.all(-1 == agent.data[idx+1])

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
            for j in range(len(agents[0].data)):
                left, middle, right = self.is_empty_self_and_neighbours(j, agents[i])
                board = jax.lax.select(self.is_body(left, middle, right), board.at[tuple(agents[i].data[j])].set(3*i+1), board)
                board = jax.lax.select(self.is_head(left, middle, right), board.at[tuple(agents[i].data[j])].set(3*i+2), board)
                board = jax.lax.select(self.is_tail(left, middle, right), board.at[tuple(agents[i].data[j])].set(3*i+3), board)
        return board

    def return_training_board(self, key: random.PRNGKey) -> np.ndarray:

        agents = self.initialise_starting_board(key)
        modified_agents = self.fill(key, agents, n_iters=5, pushpullnone_ratios=[2, 0.5, 1])
        self.agents = modified_agents
        return self.fill_unsolved_board(modified_agents)

    def return_solved_board(self) -> np.ndarray:
        return self.fill_solved_board(self.agents)


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
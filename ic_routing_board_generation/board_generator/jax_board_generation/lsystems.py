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
        return random.randint(key, (), -1, 1)

    
    def find_empty_neighbours(self, loc: Tuple[int, int], agents: List[Agent]) -> np.ndarray:
        """Given a location tuple, find the empty neighbours of a board location.
        
        loc: a tuple of shape 1x2.
        returns: a boolean array of shape 1x4.
        """
        i, j = loc
        neighbors = []
        neighbour_locs = [(i-1,j), (i,j+1), (i+1,j), (i,j-1)]
        board = self.fill_solved_board(agents)

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

    def push(self, key: random.PRNGKey, agent: Agent, agents: List[Agent]) -> Agent:
        """Pushes (grows) an Agent in a random direction."""
        side = self.heads_or_tails(key)                                                                                                     # stochastic
        pointer = jax.lax.select(side == 0, (agent.head_insertion_index-1)%self.max_size, (agent.tail_insertion_index+1)%self.max_size)     # deterministic
        loc = agent.data[pointer]
        neighbours, neighbour_locs = self.find_empty_neighbours(loc, agents)                                                                # deterministic
        loc = self.growth_neighbour_loc(key, neighbour_locs, neighbours)        
        agent = jax.lax.cond(loc[0] >= 0, lambda _: self.grow_agent(side, loc, agent), lambda _: agent, None)                               # deterministic
        return agent

    
    def shrink_agent(self, side: int, agent: Agent) -> Agent:
        """Given a direction and an Agent, shrink the Agent in the given direction."""
        agent = jax.lax.cond(side == 0, stack_pop_head, stack_pop_tail, agent)
        return agent

    def do_nothing(self, side: int, agent: Agent) -> Agent:
        return agent

    def pull(self, key:random.PRNGKey, agent: Agent, agents: List[Agent]) -> Agent:
        """Pulls (shrinks) the agent in a random direction."""
        side = self.heads_or_tails(key)                                                                                                 # stochastic
        agent_length = np.count_nonzero(agent.data[:,0]+1)   
        agent = jax.lax.cond(agent_length > 1, self.shrink_agent, self.do_nothing, side, agent)
        return agent

    def fill(self,
             key: random.PRNGKey,
             agents: List[Agent]) -> None:
        return jax.lax.while_loop(self.are_agents_good, self.fill_method, (key, agents))

    def are_agents_good(self, inp) -> bool:
        key, agents = inp
        is_board_good = np.asarray([np.count_nonzero(agent.data[:,0]+1)>1 for agent in agents])
        all_good = np.all(is_board_good)
        return ~all_good

    def fill_method_inner(self, i, inp) -> None:
        chosen_key, agents = inp
        new_agents = agents
        for i in range(len(agents)):
            key, _ = random.split(chosen_key)
            action = jax.random.choice(key, 3, (), p=np.asarray(self.pushpullnone_ratios))
            nagent = jax.lax.cond(action == 0, self.push, self.pull, key, agents[i], new_agents)
            new_agents[i] = nagent
        # agents = new_agents
        return (key, new_agents)

    
    def fill_method(self, inp) -> None:
        return jax.lax.fori_loop(0, self.n_iters, self.fill_method_inner, inp)

    
    def fill_unsolved_board(self, agents: Agent) -> np.ndarray:
        """Fills an `unsolved` board with Agents."""
        board = np.zeros((self.rows, self.cols))
        for i in range(len(agents)):
            for j in range(len(agents[0].data)):
                left, middle, right = self.is_empty_self_and_neighbours(j, agents[i])
                board = jax.lax.select(self.is_head(left, middle, right), board.at[tuple(agents[i].data[j])].set(3*i+2), board)
                board = jax.lax.select(self.is_tail(left, middle, right), board.at[tuple(agents[i].data[j])].set(3*i+3), board)
        return np.asarray(board, dtype=int)
    
    
    def is_empty_self_and_neighbours(self, idx: np.ndarray, agent: Agent) -> bool:
        """Tests if a location and its neighbours are empty."""
        return np.all(-1 == agent.data[(idx-1)%self.max_size]), np.all(-1 == agent.data[idx]), np.all(-1 == agent.data[(idx+1)%self.max_size])

    
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

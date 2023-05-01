from enum import Enum

from ic_routing_board_generation.board_generator.jax_board_generation.parallel_random_walk import \
    ParallelRandomWalk
from ic_routing_board_generation.board_generator.numpy_board_generation.bfs_board import BFSBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.bfs_board_variations import \
    BFSBoardMinBends, BFSBoardFifo, BFSBoardShortest, BFSBoardLongest
from ic_routing_board_generation.board_generator.numpy_board_generation.board_generator_numberlink_oj import \
    NumberLinkBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.board_generator_random_walk_rb import \
    RandomWalkBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.board_generator_wfc_oj import \
    WFCBoard

from ic_routing_board_generation.board_generator.jax_board_generation.board_generator_random_seed_rb import \
    RandomSeedBoard
from ic_routing_board_generation.board_generator.numpy_board_generation.lsystems_numpy import \
    LSystemBoardGen
from ic_routing_board_generation.ic_rl_training.online_generators.uniform_generator import \
    UniformRandomGenerator


class BoardName(str, Enum):
    """Enum of implemented board generators."""
    RANDOM_WALK = "RandomWalk"
    JAX_PARALLEL_RW = "ParallelRandomWalk"
    BFS_BASE = "BFSBase"
    BFS_MIN_BENDS = "BFSMin_Bends"
    BFS_FIFO = "BFSFIFO"
    BFS_SHORTEST = "BFSShort"
    BFS_LONGEST = "BFSLong"
    LSYSTEMS = "L-Systems"
    WFC = "WFC"
    NUMBERLINK = "NumberLink"

    # JAX_SEED_EXTENSION = "JAX Seed Extension"
    # JAX_UNIFORM = "JAX Uniform"


class BoardGenerator:
    """Maps BoardGeneratorType to class of generator."""
    board_generator_dict = {
        BoardName.RANDOM_WALK: RandomWalkBoard,
        BoardName.JAX_PARALLEL_RW: ParallelRandomWalk,
        BoardName.BFS_BASE: BFSBoard,
        BoardName.BFS_MIN_BENDS: BFSBoardMinBends,
        BoardName.BFS_FIFO: BFSBoardFifo,
        BoardName.BFS_SHORTEST: BFSBoardShortest,
        BoardName.BFS_LONGEST: BFSBoardLongest,
        BoardName.LSYSTEMS: LSystemBoardGen,
        BoardName.WFC: WFCBoard,
        BoardName.NUMBERLINK: NumberLinkBoard,

        # BoardName.JAX_SEED_EXTENSION: RandomSeedBoard,
        # BoardName.JAX_UNIFORM: UniformRandomGenerator,
    }

    @classmethod
    def get_board_generator(cls, board_enum: BoardName):
        """Return class of desired board generator."""
        return cls.board_generator_dict[board_enum]

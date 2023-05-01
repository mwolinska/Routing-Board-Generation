from enum import Enum

from routing_board_generation.board_generation_methods.jax_implementation.board_generation.lsystems import \
    JAXLSystemBoard
from routing_board_generation.board_generation_methods.jax_implementation.board_generation.parallel_random_walk import \
    ParallelRandomWalkBoard
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.bfs_board import \
    BFSBoard
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.bfs_board_variations import \
    BFSBoardMinBends, BFSBoardFifo, BFSBoardShortest, BFSBoardLongest
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.numberlink import \
    NumberLinkBoard
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.random_walk import \
    RandomWalkBoard
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation.wave_function_collapse import \
    WFCBoard


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
        BoardName.JAX_PARALLEL_RW: ParallelRandomWalkBoard,
        BoardName.BFS_BASE: BFSBoard,
        BoardName.BFS_MIN_BENDS: BFSBoardMinBends,
        BoardName.BFS_FIFO: BFSBoardFifo,
        BoardName.BFS_SHORTEST: BFSBoardShortest,
        BoardName.BFS_LONGEST: BFSBoardLongest,
        BoardName.LSYSTEMS: JAXLSystemBoard,
        BoardName.WFC: WFCBoard,
        BoardName.NUMBERLINK: NumberLinkBoard,

        # BoardName.JAX_SEED_EXTENSION: RandomSeedBoard,
        # BoardName.JAX_UNIFORM: UniformRandomGenerator,
    }

    @classmethod
    def get_board_generator(cls, board_enum: BoardName):
        """Return class of desired board generator."""
        return cls.board_generator_dict[board_enum]

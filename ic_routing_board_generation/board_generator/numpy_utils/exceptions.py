# Define exceptions for the board validity checks
class IncorrectBoardSizeError(Exception):
    """ Raised when a board size does not match the specified dimensions."""
    pass


class NumAgentsOutOfRangeError(Exception):
    """ Raised when self._wires_on_board is negative."""
    pass


class EncodingOutOfRangeError(Exception):
    """ Raised when one or more cells on the board have an invalid index."""
    pass


class DuplicateHeadsTailsError(Exception):
    """ Raised when one of the heads or tails of a wire is duplicated."""
    pass


class MissingHeadTailError(Exception):
    """ Raised when one of the heads or tails of a wire is missing."""
    pass


class InvalidWireStructureError(Exception):
    """ Raised when one or more of the wires has an invalid structure, e.g. looping or branching."""
    pass


class PathNotFoundError(Exception):
    """ Raised when a path cannot be found between a head and a target."""
    pass

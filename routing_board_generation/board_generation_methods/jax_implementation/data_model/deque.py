from typing import NamedTuple, Tuple
import chex
import jax.numpy as jnp
import jax


"""Define a stack data structure which can be used with Jax.

To be usable with Jax transformations, data structures must have fixed shape.
A stack can be represented by a 2D array, each row containing the flatten representation of an
element. For example, in a stack of chambers (as in the maze generation), each row should
contain 4 digits: x0, y0, width and height.
We also need an upper bound to the number of elements the stack can contain, so that we always
have a fixed number of row.
To create the stack we need two parameters: the size of one element, and the maximum number
of elements.

For example, here is an empty stack containing elements of size 4 with at most 5 elements:

[[. . . .]
[. . . .]
[. . . .]
[. . . .]
[. . . .]]

Originally the stack is empty, data only contains padding. Elements can be pushed on the stack.
Say an element `[a,b,c,d]` is pushed on the stack for example:

[[a b c d]
[. . . .]
[. . . .]
[. . . .]
[. . . .]]

In this 2D array, how do we differentiate between the first row, which actually contains an element,
and the other rows, which only contain padding ?
An `insertion_index` can be used, which contains the index at which the next element should
be inserted. All rows up to this index are elements of the stack, all rows after contain padding.

[[a b c d]
[. . . .] <- insertion_index # everything from this row is padding and should be ignored
[. . . .]
[. . . .]
[. . . .]]

"""

class LSystemsAgent(NamedTuple):
    """Define a stack usable with Jax transformations.

    - data: array of fixed-shape, each row up to insertion_index containing an element of the stack.
        Rows after insertion_index should be ignored, they only contain padding to make sure data
        is of fixed shape and can be used with Jax transformations.
        The width of the data is the number of features in an element, the height is the maximum
        number of elements the stack can contain.
    - head_insertion_index: the index of the row at which to insert the next element in data.
                            Should be 0 for an empty stack.
    - tail_insertion_index: the index of the row at which to insert the next element in data. 
                            Should be max_size-1 (data.shape[0] - 1) for an empty stack.
    """

    data: jnp.array
    head_insertion_index: int
    tail_insertion_index: int

def create_stack(max_size: int, num_features: int) -> LSystemsAgent:
    """Create an empty stack.

    Args:
        max_size: maximum number of elements the stack can contain.
        num_features: number of features in an element.

    Returns:
        stack: the created stack.
    """
    return LSystemsAgent(jnp.zeros((max_size, num_features), dtype=int) - 1, 0, max_size - 1)

def stack_push_head(stack: LSystemsAgent, element: chex.Array) -> LSystemsAgent:
    """Push an element on top of the stack.

    Args:
        stack: the stack on which to push element.
        element: the element to push on the stack.

    Returns:
        stack: the stack containing the new element.
    """
    return LSystemsAgent(
        stack.data.at[stack.head_insertion_index].set(element),
        (stack.head_insertion_index + 1) % stack.data.shape[0],
        stack.tail_insertion_index
        )

def stack_push_tail(stack: LSystemsAgent, element: chex.Array) -> LSystemsAgent:
    """Push an element on top of the stack.

    Args:
        stack: the stack on which to push element.
        element: the element to push on the stack onto the end.

    Returns:
        stack: the stack containing the new element.
    """
    return LSystemsAgent(
        stack.data.at[stack.tail_insertion_index].set(element),
        stack.head_insertion_index,
        (stack.tail_insertion_index - 1) % stack.data.shape[0],
    )

def stack_pop_head(stack: LSystemsAgent) -> Tuple[LSystemsAgent, chex.Array]:
    """Pop the last element from the stack.

    Args:
        stack: the stack from which to pop the last element.

    Returns:
        stack: the stack without the last element.
        element: the last element from the stack.
    """
    last_element_idx = (stack.head_insertion_index - 1) % stack.data.shape[0]
    element = stack.data[last_element_idx]
    stack = LSystemsAgent(
        stack.data.at[last_element_idx].set(jnp.zeros_like(element)-1),
        last_element_idx,
        stack.tail_insertion_index,
    )
    return stack

def stack_pop_tail(stack: LSystemsAgent) -> Tuple[LSystemsAgent, chex.Array]:
    """Pop the last element from the stack.

    Args:
        stack: the stack from which to pop the last tail element.

    Returns:
        stack: the stack without the last element (tail-end).
        element: the last element from the stack.
    """
    last_element_idx = (stack.tail_insertion_index + 1) % stack.data.shape[0]
    element = stack.data[last_element_idx]
    stack = LSystemsAgent(
        stack.data.at[last_element_idx].set(jnp.zeros_like(element)-1),
        stack.head_insertion_index,
        last_element_idx,
    )
    return stack

def empty_stack(stack: LSystemsAgent) -> bool:
    """Check if a stack is empty.

    Args:
        stack: the stack to check.

    Returns:
        Boolean stating whether the stack is empty.
    """
    return stack.head_insertion_index == 0 and stack.tail_insertion_index == stack.data.shape[0]-1

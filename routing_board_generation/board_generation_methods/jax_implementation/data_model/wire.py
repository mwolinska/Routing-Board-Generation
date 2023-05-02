# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### This file contains the following methods adapted from implementations in Jumanji:
# This follows Stack class from maze utils in Jumanji adapted for this repo's use case
###

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
from typing import NamedTuple, Tuple

import chex
import jax.numpy as jnp


class Wire(NamedTuple):
    """Define a stack usable with Jax transformations.

    - data: array of fixed-shape, each row up to insertion_index containing an element of the stack.
        Rows after insertion_index should be ignored, they only contain padding to make sure data
        is of fixed shape and can be used with Jax transformations.
        The width of the data is the number of features in an element, the height is the maximum
        number of elements the stack can contain.
    - insertion_index: the index of the row at which to insert the next element in data. Should be
        0 for an empty stack.
    """
    path: chex.Array
    insertion_index: int
    wire_id: int
    start: Tuple[int, int]
    end: Tuple[int, int]

def create_wire(
    max_size: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    wire_id: int
) -> Wire:
    """Create an empty stack.

    Args:
        max_size: maximum number of elements the stack can contain. GRID SIZE


    Returns:
        stack: the created stack of size grid size x 2 - i.e. rows and columns
    """
    return Wire(jnp.full((max_size), fill_value=-1, dtype=int), 0, wire_id=wire_id, start=start, end=end)


def create_wire_for_sequential_rw(
    max_size: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    wire_id: int
) -> Wire:
    """Create an empty stack.

    Args:
        max_size: maximum number of elements the stack can contain. GRID SIZE


    Returns:
        stack: the created stack of size grid size x 2 - i.e. rows and columns
    """
    return Wire(jnp.full((max_size, 2), fill_value=-1, dtype=int), 0, wire_id=wire_id, start=start, end=end)


def stack_push(stack: Wire, element: chex.Array) -> Wire:
    """Push an element on top of the stack.

    Args:
        stack: the stack on which to push element.
        element: the element to push on the stack.

    Returns:
        stack: the stack containing the new element.
    """
    return Wire(start=stack.start, end = stack.end,
                wire_id=stack.wire_id,
                path=stack.path.at[stack.insertion_index].set(element),
                insertion_index=stack.insertion_index + 1)


def stack_pop(stack: Wire) -> Tuple[Wire, chex.Array]:
    """Pop the last element from the stack.

    Args:
        stack: the stack from which to pop the last element.

    Returns:
        stack: the stack without the last element.
        element: the last element from the stack.
    """
    element = stack.path[stack.insertion_index]
    return Wire(
        stack.path[:stack.insertion_index],
        insertion_index=stack.insertion_index - 1,
        wire_id=stack.wire_id,
        start=stack.start,
        end=stack.end
    ), element
    return stack, element

def stack_reverse(stack: Wire) -> Wire:
    """Reverse the items in the stack before the insertion index.

    Args:
        stack: the stack to reverse.

    Returns:
        stack: the reversed stack.
    """
    return Wire(
        jnp.concatenate([jnp.flip(stack.path[:stack.insertion_index], axis=0), stack.path[stack.insertion_index:]]),
        insertion_index=stack.insertion_index,
        wire_id=stack.wire_id,
        start=stack.start,
        end=stack.end
    )

def stack_clip(stack:Wire) -> Wire:
    """Clip the stack.

    Args:
        stack: the stack to clip.

    Returns:
        stack: the clipped stack.
    """
    return Wire(
        stack.path[:stack.insertion_index+1],
        insertion_index=stack.insertion_index,
        wire_id=stack.wire_id,
        start=stack.start,
        end=stack.end
    )

def empty_stack(stack: Wire) -> bool:
    """Check if a stack is empty.

    Args:
        stack: the stack to check.

    Returns:
        Boolean stating whether the stack is empty.
    """
    return stack.insertion_index == 0

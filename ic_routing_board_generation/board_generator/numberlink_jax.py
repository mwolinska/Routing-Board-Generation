import sys
import random
import collections
from copy import deepcopy
import itertools
import argparse
import numpy as np
import sys
import collections
import string
# from colorama import Fore, Style
# from colorama import init as init_colorama

from ic_routing_board_generation.board_generator.abstract_board import AbstractBoard


# Number of tries at adding loops to the grid before redrawing the side paths.
LOOP_TRIES = 1000


"""
Functions from mitm
"""
import random
from collections import Counter, defaultdict
import itertools


"""
Checklist for making the board generator jitable:
1. Use jax.numpy instead of numpy
2. Use jax.random instead of random
3. Replace conditionals and loops with jax specific functions
4. Set no properties of a method that will change between runs
5. Remember that jax arrays are immutable 
6. Remember that we want pure functions - no side effects
7. We can't use sets etc., create boolean masks instead

"""

import jax
import chex
import jax.numpy as jnp
import jax.random as jrandom

#TODO: test whether this works!!
#Think this might be done?
class Path:
    # What is this actually storing?
    # Think it's storing the path as a list of steps
    # 0 = T, 1 = L, 2 = R

    def __init__(self, steps):
        self.steps = steps
        self.T, self.L, self.R = 0, 1, 2

    def xys(self, dx=0, dy=1):
        """ Yields all positions on path """
        x, y = 0, 0
        # Want this to be a jax array, so need to
        # Create a jax array of the same size as steps
        all_positions = jnp.zeros((len(self.steps) + 1, 2))
        all_positions = jax.ops.index_update(all_positions, jax.ops.index[0, :], jnp.array([x, y]))


        def step_function(carry, step):
            x, y, dx, dy, idx = carry
            x, y = x + dx, y + dy

            all_positions_updated = jax.ops.index_update(all_positions, jax.ops.index[idx, :], jnp.array([x, y]))

            dx_new = jax.lax.cond(step == self.L, lambda _: -dy, lambda _: dx, None)
            dy_new = jax.lax.cond(step == self.L, lambda _: dx, lambda _: dy, None)

            dx_new = jax.lax.cond(step == self.R, lambda _: dy, lambda _: dx_new, None)
            dy_new = jax.lax.cond(step == self.R, lambda _: -dx, lambda _: dy_new, None)

            x, y = jax.lax.cond(step == self.T, lambda _: (x + dx_new, y + dy_new), lambda _: (x, y), None)
            idx_new = jax.lax.cond(step == self.T, lambda _: idx + 2, lambda _: idx + 1, None)

            return (x, y, dx_new, dy_new, idx_new), all_positions_updated

        _, final_positions = jax.lax.scan(step_function, (x, y, dx, dy, 1), self.steps)
        return final_positions

    def test(self):
        """ Tests path is non-overlapping. """
        ps = self.xys()

        # Sort the positions lexicographically
        sorted_ps = jnp.sort(ps, axis=0)

        # Compute differences between consecutive elements
        diff_ps = jnp.diff(sorted_ps, axis=0)

        # Check if any consecutive elements are equal (i.e., difference is 0)
        has_duplicates = jnp.any(jnp.all(diff_ps == 0, axis=1))

        # Return True if there are no duplicates (non-overlapping)
        return ~has_duplicates

    def test_loop(self):
        """ Tests path is non-overlapping, except for first and last. """
        ps = self.xys()

        # Sort the positions lexicographically
        sorted_ps = jnp.sort(ps, axis=0)

        # Compute differences between consecutive elements
        diff_ps = jnp.diff(sorted_ps, axis=0)

        # Check if any consecutive elements are equal (i.e., difference is 0)
        duplicate_count = jnp.sum(jnp.all(diff_ps == 0, axis=1))

        # Check if the path is non-overlapping, except for first and last
        is_loop = jnp.logical_or(duplicate_count == 0, jnp.logical_and(duplicate_count == 1, jnp.all(ps[0] == ps[-1])))

        return is_loop

    def winding(self):
        return jnp.sum(self.steps == self.R) - jnp.sum(self.steps == self.L)

#TODO: test whether this works!!
def unrotate(x, y, dx, dy):
    """ Inverse rotate x, y by (dx,dy), where dx,dy=0,1 means 0 degrees.
        Basically rotate(dx,dy, dx,dy) = (0, 1). """

    def cond_fun(carry):
        _, _, dx, dy = carry
        return jnp.logical_not(jnp.logical_and(dx == 0, dy == 1))

    def body_fun(carry):
        x, y, dx, dy = carry
        x_new, y_new = -y, x
        dx_new, dy_new = -dy, dx
        return x_new, y_new, dx_new, dy_new

    x, y, _, _ = jax.lax.while_loop(cond_fun, body_fun, (x, y, dx, dy))
    return x, y


class Mitm:
    def __init__(self, lr_price, t_price, key, h=10, w=10):
        self.lr_price = lr_price
        self.t_price = t_price
        #TODO: see where these are used, might be some issues with these
        # and how they are used in the code
        #self.inv = defaultdict(list)
        #self.list = []
        self.T, self.L, self.R = 0, 1, 2
        self.key = key
        self.h = h
        self.w = w

    def prepare(self, budget):
        # NOTE: self.grid_size is the size of the mitm grid, not the
        # size of the final board. Can probably optimise the size of the grid,
        # but for now going to use an upper bound.
        self.grid_size = 4 * max(self.h, self.w) + budget + 1

        dx0, dy0 = 0, 1
        # Create zeros array of shape (grid_size, grid_size)
        start_grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        good_paths = self._good_paths(0, 0, dx0, dy0, budget, start_grid)

        # Initialize the list and inv arrays
        max_paths = good_paths.shape[0]
        self.list = jnp.zeros((max_paths, 5), dtype=jnp.int32)
        self.inv = jnp.zeros((self.grid_size, self.grid_size, 2, 2, max_paths), dtype=jnp.int32)

        def loop_body(i, state):
            list_, inv = state
            path, x, y, dx, dy = good_paths[i]
            
            list_ = jax.ops.index_update(list_, jax.ops.index[i, :], jnp.array([path, x, y, dx, dy]))

            def update_inv(_):
                inv_ = jax.ops.index_update(inv, jax.ops.index[x, y, dx, dy], jnp.array([path]))
                return inv_

            inv = jax.lax.cond(i < max_paths, update_inv, lambda _: inv, operand=None)
            
            return list_, inv
        # TODO: I think this is ok, as these are not random,
        # but could be an issue with self. stuff
        self.list, self.inv = jax.lax.fori_loop(0, max_paths, loop_body, (self.list, self.inv))
        return self.list, self.inv

    def rand_path2(self, xn, yn, dxn, dyn, key):
        """ Like rand_path, but uses a combination of a fresh random walk and
            the lookup table. This allows for even longer paths. """

        def loop_condition(state):
            _, _, _, _, _, _, x, y, _ = state
            return jnp.logical_not((x == xn) & (y == yn))

        def loop_body(state):
            key, steps, seen, step_choices, step_weights, x, y, dx, dy = state

            key, subkey = jrandom.split(key)
            step = jrandom.choice(subkey, step_choices, p=step_weights)
            steps = steps.at[len(steps)].set(step)

            x, y = x + dx, y + dy
            seen = seen.at[x, y].set(True)

            dx, dy = jax.lax.cond(
                step == self.L,
                lambda _: (-dy, dx),
                lambda _: jax.lax.cond(step == self.R, lambda _: (dy, -dx), lambda _: (dx, dy)),
                operand=None
            )

            x, y = jax.lax.cond(step == self.T, lambda _: (x + dx, y + dy), lambda _: (x, y), operand=None)

            return key, steps, seen, step_choices, step_weights, x, y, dx, dy

        step_choices = jnp.array([self.L, self.R, self.T], dtype=jnp.int32)
        step_weights = jnp.array([1 / self.lr_price, 1 / self.lr_price, 2 / self.t_price], dtype=jnp.float32)

        init_state = (key, jnp.zeros((2 * (abs(xn) + abs(yn)),), dtype=jnp.int32), jnp.zeros((2 * xn + 1, 2 * yn + 1), dtype=jnp.bool_),
                    step_choices, step_weights, 0, 0, 0, 1)
        _, steps, _, _, _, x, y, dx, dy = jax.lax.while_loop(loop_condition, loop_body, init_state)

        ends = self._lookup(dx, dy, xn - x, yn - y, dxn, dyn)

        def ends_nonempty(_):
            key, subkey = jrandom.split(key)
            end_idx = jrandom.randint(subkey, (1,), 0, len(ends))
            end = ends[end_idx]
            return Path(steps + end)

        def ends_empty(_):
            return Path(steps)

        return jax.lax.cond(ends.size > 0, ends_nonempty, ends_empty, operand=None)

    def rand_loop(self, clock=0, key=None):
        """ Set clock = 1 for clockwise, -1 for anti clockwise. 0 for don't care. """

        def loop_condition(state):
            _, path, _, _, _, _, _, _, winding = state
            return jnp.logical_not((clock == 0) | (winding == clock * 4))

        def loop_body(state):
            key, path, x, y, dx, dy, steps, path2s, winding = state

            key, subkey = jrandom.split(key)
            idx = jrandom.randint(subkey, (1,), 0, len(self.list))
            path, x, y, dx, dy = self.list[idx]
            path2s = self._lookup(dx, dy, -x, -y, 0, 1)

            key, subkey = jrandom.split(key)
            idx = jrandom.randint(subkey, (1,), 0, len(path2s))
            path2 = path2s[idx]

            joined = Path(path + path2)
            winding = joined.winding()

            return key, path, x, y, dx, dy, steps, path2s, winding

        init_state = (key, None, 0, 0, 0, 1, 0, 0, 0)
        _, path, x, y, dx, dy, _, path2s, _ = jax.lax.while_loop(loop_condition, loop_body, init_state)

        key, subkey = jrandom.split(key)
        idx = jrandom.randint(subkey, (1,), 0, len(path2s))
        path2 = path2s[idx]
        joined = Path(path + path2)

        return jax.lax.cond(joined.test_loop(), lambda _: joined, lambda _: self.rand_loop(clock, key), operand=None)


    def _good_paths(self, x, y, dx, dy, budget, seen):
        # Note: changed this so that we call this with seen being a jnp array of zeros
        # to start. Shouldn't be a problem, but could be.

        def generate_paths(x, y, dx, dy, budget, seen):
            def budget_negative(_):
                return jnp.zeros((0, 1, 4), dtype=jnp.int32)

            def budget_non_negative(_):
                seen_updated = seen.at[(x, y)].set(True)
                x1, y1 = x + dx, y + dy

                def not_seen_true(_):
                    seen_L = seen_updated.at[(x1, y1)].set(True)
                    seen_R = seen_L
                    print("budget is: ", budget)
                    paths_L = generate_paths(x1, y1, -dy, dx, budget - self.lr_price, seen_L)
                    paths_R = generate_paths(x1, y1, dy, -dx, budget - self.lr_price, seen_R)
                    path_LR = jnp.concatenate([paths_L, paths_R], axis=0)
                    path_LR = path_LR.at[:, 0, :].set(jnp.array([self.L, self.R]))

                    x2, y2 = x1 + dx, y1 + dy

                    def not_seen_true_inner(_):
                        seen_T = seen_updated.at[(x2, y2)].set(True)
                        paths_T = generate_paths(x2, y2, dx, dy, budget - self.t_price, seen_T)
                        path_T = paths_T.at[:, 0, :].set(jnp.array([self.T]))

                        return jnp.concatenate([path_LR, path_T], axis=0)

                    def not_seen_false_inner(_):
                        return path_LR

                    path = jax.lax.cond(jax.lax.eq(seen_updated[x2, y2], 0), not_seen_true_inner, not_seen_false_inner, operand=None)

                    return path

                def not_seen_false(_):
                    return jnp.zeros((0, 1, 4), dtype=jnp.int32)

                paths = jax.lax.cond(jax.lax.eq(seen_updated[x1, y1], 0), not_seen_true, not_seen_false, operand=None)
                paths = paths.at[:, :, 2:].set(jnp.array([dx, dy]))

                return paths

            return jax.lax.cond(budget < 0, budget_negative, budget_non_negative, operand=None)

        return generate_paths(x, y, dx, dy, budget, seen)

    def _lookup(self, dx, dy, xn, yn, dxn, dyn):
        """ Return cached paths coming out of (0,0) with direction (dx,dy)
            and ending up in (xn,yn) with direction (dxn,dyn). """
        # Give me a path, pointing in direction (0,1) such that when I rotate
        # it to (dx, dy) it ends at xn, yn in direction dxn, dyn.
        xt, yt = unrotate(xn, yn, dx, dy)
        dxt, dyt = unrotate(dxn, dyn, dx, dy)
        return self.inv[xt, yt, dxt, dyt]





class NumberLinkBoard(AbstractBoard):
    def __init__(self, rows, cols, num_agents):
        self.width = rows
        self.height = cols
        self.num_agents = num_agents
        self.no_colors = True
        self.zero = True
        self.solve = False
        self.no_pipes = True
        self.terminal_only = True
        self.verbose = False

        self.w, self.h = self.width, self.height
        # TODO: include check that width and height are each greater than 3
    

    def __call__(self, key: chex.PRNGKey) -> jnp.array:
        """
        Generate a random, unsolved NumberLink board.
        """
        board = self.main(key)
        return board
    
    def main(self, key: chex.PRNGKey) -> jnp.array:
        mitm = Mitm(lr_price=2, t_price=1, key=key)
        # This might be ok: preparation should be constant 
        mitm.prepare(min(20, max(self.h, 6)))
        

        


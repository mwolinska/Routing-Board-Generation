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




class Mitm:
    def __init__(self, lr_price, t_price, key):
        self.lr_price = lr_price
        self.t_price = t_price
        self.inv = defaultdict(list)
        self.list = []
        self.T, self.L, self.R = 0, 1, 2
        self.key = key

    def prepare(self, budget):
        dx0, dy0 = 0, 1
        good_paths = self._good_paths(0, 0, dx0, dy0, budget)
        for path, (x, y, dx, dy) in good_paths:
            self.list.append((path, x, y, dx, dy))
            self.inv[x, y, dx, dy].append(path)

    def rand_path(self, xn, yn, dxn, dyn):
        """ Returns a path, starting at (0,0) with dx,dy = (0,1)
            and ending at (xn,yn) with direction (dxn, dyn) """
        while True:
            path, x, y, dx, dy = random.choice(self.list)
            path2s = self._lookup(dx, dy, xn - x, yn - y, dxn, dyn)
            if path2s:
                path2 = random.choice(path2s)
                joined = Path(path + path2)
                if joined.test():
                    return joined

    def rand_path2(self, xn, yn, dxn, dyn):
        """ Like rand_path, but uses a combination of a fresh random walk and
            the lookup table. This allows for even longer paths. """
        seen = set()
        path = []
        while True:
            seen.clear()
            del path[:]
            x, y, dx, dy = 0, 0, 0, 1
            seen.add((x, y))
            for _ in range(2 * (abs(xn) + abs(yn))):
                # We sample with weights proportional to what they are in _good_paths()
                step, = random.choices(
                    [self.L, self.R, self.T], [1 / self.lr_price, 1 / self.lr_price, 2 / self.t_price])
                path.append(step)
                x, y = x + dx, y + dy
                if (x, y) in seen:
                    break
                seen.add((x, y))
                if step == self.L:
                    dx, dy = -dy, dx
                if step == self.R:
                    dx, dy = dy, -dx
                elif step == self.T:
                    x, y = x + dx, y + dy
                    if (x, y) in seen:
                        break
                    seen.add((x, y))
                if (x, y) == (xn, yn):
                    return Path(path)
                ends = self._lookup(dx, dy, xn - x, yn - y, dxn, dyn)
                if ends:
                    return Path(tuple(path) + random.choice(ends))

    def rand_loop(self, clock=0):
        """ Set clock = 1 for clockwise, -1 for anti clockwise. 0 for don't care. """
        while True:
            # The list only contains 0,1 starting directions
            path, x, y, dx, dy = random.choice(self.list)
            # Look for paths ending with the same direction
            path2s = self._lookup(dx, dy, -x, -y, 0, 1)
            if path2s:
                path2 = random.choice(path2s)
                joined = Path(path + path2)
                # A clockwise path has 4 R's more than L's.
                if clock and joined.winding() != clock * 4:
                    continue
                if joined.test_loop():
                    return joined
    
    def add_point(seen, pair):
        return seen | pair

    def _good_paths(self, x, y, dx, dy, budget, seen=None):
        """ Returns a list of paths that end at (x,y) with direction (dx,dy)
            and cost at most budget. (According to copilot, might be true?!)"""
        if seen is None:
            seen = set()
        if budget >= 0:
            yield (), (x, y, dx, dy)
        if budget <= 0:
            return
        
        # Initialise output array, which will be used to store the paths
        # This will be a fixed size array, so we can jit it
        # It will then just be a boolean mask
        good_paths = jnp

        seen.add((x, y))  # Remember cleaning this up (A)
        x1, y1 = x + dx, y + dy
        if (x1, y1) not in seen:
            for path, end in self._good_paths(
                    x1, y1, -dy, dx, budget - self.lr_price, seen):
                yield (self.L,) + path, end
            for path, end in self._good_paths(
                    x1, y1, dy, -dx, budget - self.lr_price, seen):
                yield (self.R,) + path, end
            seen.add((x1, y1))  # Remember cleaning this up (B)
            x2, y2 = x1 + dx, y1 + dy
            if (x2, y2) not in seen:
                for path, end in self._good_paths(
                        x2, y2, dx, dy, budget - self.t_price, seen):
                    yield (self.T,) + path, end
            seen.remove((x1, y1))  # Clean up (B)
        seen.remove((x, y))  # Clean up (A)

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
    

    def __call__(self, key: chex.PRNGKey) -> jnp.Array:
        """
        Generate a random, unsolved NumberLink board.
        """
        board = self.main(key)
        return board
    
    def main(self, key: chex.PRNGKey) -> jnp.Array:
        mitm = Mitm(lr_price=2, t_price=1, key=key)
        # This might be ok: preparation should be constant 
        mitm.prepare(min(20, max(self.h, 6)))
        

        


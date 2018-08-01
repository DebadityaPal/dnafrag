from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .constants import GENOME_DOMAIN_NAME, COUNTS_RANGE_NAME, INSERT_DOMAIN_NAME


class NumpyBackend(object):
    """A dummy, in-memmory implementation of the TileDB API using numpy arrays.
    We only use this for unit tests on small arrays when we don't want to write
    them to disk. Array access is O(n)."""

    def __init__(self, n, m, n_idxs, m_idxs, values):
        self.n = n
        self.m = m
        ind = np.lexsort((m_idxs, n_idxs))
        self.n_idxs = n_idxs[ind]
        self.m_idxs = m_idxs[ind]
        self.values = values[ind]

    def __getitem__(self, key):
        r, c = key

        if not isinstance(r, slice):
            r = slice(r, r + 1)
        if not isinstance(c, slice):
            c = slice(c, c + 1)

        if r.start == None:
            r = slice(0, r.stop)
        if r.stop == None:
            r = slice(r.start, self.n)
        if c.start == None:
            c = slice(0, c.stop)
        if c.stop == None:
            c = slice(c.start, self.m)

        r_idxs = np.bitwise_and(self.n_idxs >= r.start, self.n_idxs < r.stop)
        c_idxs = np.bitwise_and(self.m_idxs >= c.start, self.m_idxs < c.stop)
        idxs = np.bitwise_and(r_idxs, c_idxs)

        ret = {
            "coords": {
                GENOME_DOMAIN_NAME: self.n_idxs[idxs],
                INSERT_DOMAIN_NAME: self.m_idxs[idxs],
            },
            COUNTS_RANGE_NAME: self.values[idxs],
        }

        return ret

    def nonempty_domain(self):
        return ((self.n_idxs[0], self.m_idxs[0]), (self.n_idxs[-1], self.m_idxs[-1]))

    @property
    def shape(self):
        return (self.n, self.m)

    @property
    def ndim(self):
        return 2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tiledb
from scipy.sparse import coo_matrix
from operator import itemgetter

from .constants import GENOME_DOMAIN_NAME, COUNTS_RANGE_NAME, INSERT_DOMAIN_NAME


class DNAFragArray(object):

    def __init__(self, array, mode="r"):
        if isinstance(array, tiledb.SparseArray):
            self._arr = array
        else:
            ctx = tiledb.Ctx()
            self._arr = tiledb.SparseArray.load(ctx, array)

    def __getitem__(self, key):
        r, c = key
        return self._arr[c, r]

    def __setitem__(self, key, item):
        raise NotImplemented("DNAFragArrays are read-only")

    def fill_array(self, start_pos, a, zero=True):
        assert a.shape[0] == self._arr.shape[1]
        assert start_pos + a.shape[1] < self._arr.shape[0]
        assert start_pos >= 0

        if zero:
            a[:, :] = 0

        d = self._arr[start_pos : start_pos + a.shape[1], :]
        c = d["coords"]
        i = c[INSERT_DOMAIN_NAME]  # transpose of on-disk array
        j = c[GENOME_DOMAIN_NAME] - start_pos  # coords are offset from start
        a[i, j] = d[COUNTS_RANGE_NAME]

    def add_to_array(self, start_pos, a):
        """Add `vplot[:, start_pos:start_pos + a.shape[1]]` to `a`."""
        assert a.shape[0] == self._arr.shape[1]
        assert start_pos + a.shape[1] < self._arr.shape[0]
        assert start_pos >= 0

        d = self._arr[start_pos : start_pos + a.shape[1], :]
        c = d["coords"]
        i = c[INSERT_DOMAIN_NAME]  # transpose of on-disk array
        j = c[GENOME_DOMAIN_NAME] - start_pos  # coords are offset from start
        a[i, j] += d[COUNTS_RANGE_NAME]

    @property
    def shape(self):
        return self._arr.shape[::-1]

    @property
    def ndim(self):
        return self._arr.ndim

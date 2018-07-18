from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tiledb
from scipy.sparse import coo_matrix
from operator import itemgetter

from .context import context as ctx
from .constants import GENOME_DOMAIN_NAME, COUNTS_RANGE_NAME, INSERT_DOMAIN_NAME


class DNAFragArray(object):

    def __init__(self, array, mode="r", subsample_rate=None):
        """subsample_rate is the probability of *keeping* an entry."""
        if array is None:
            # empty array -- this is used by DNAFragMultiArray
            self._arr = None
        elif isinstance(array, DNAFragArray):
            # copy construcor -- this is used by DNAFragMultiArray
            self._arr = array._arr
            self.subsample_rate = array.subsample_rate
        elif isinstance(array, tiledb.SparseArray):
            self._arr = array
        else:
            assert os.path.exists(array)
            self._arr = tiledb.SparseArray(ctx, array, mode="r")

        if self._arr is not None:
            (n, m) = self._arr.shape
            nonempty = self._arr.nonempty_domain()

            if nonempty is None:
                raise ValueError("Array is empty")

            else:
                ((si, sj), (sti, stj)) = nonempty
                if n >= m:
                    _ = self._arr[si, :]
                    _ = self._arr[sti, :]
                else:
                    _ = self._arr[:, sj]
                    _ = self._arr[:, stj]

        if subsample_rate is not None:
            assert subsample_rate > 0. and subsample_rate <= 1.
        self.subsample_rate = subsample_rate

    def __getitem__(self, key):
        r, c = key
        d = self._arr[c, r]
        if self.subsample_rate is not None:
            v = d[COUNTS_RANGE_NAME]
            v = self.subsample(v)
            d[COUNTS_RANGE_NAME] = v
        return d

    def __setitem__(self, key, item):
        raise NotImplemented("DNAFragArray is read-only")

    def subsample(self, v):
        assert self.subsample_rate > 0. and self.subsample_rate <= 1.
        dropout_rate = 1.0 - self.subsample_rate
        for uval in v.unique():
            idxs = v == uval
            n_idxs = idxs.sum()
            d = np.random.binomial(uval, dropout_rate, n_idxs)
            v[idxs] = v[idxs] - d
        return v

    def fill_array(self, start_pos, a, zero=True):
        assert a.shape[0] == self.shape[0]
        assert start_pos + a.shape[1] < self.shape[1]
        assert start_pos >= 0

        if zero:
            a[:, :] = 0

        d = self[:, start_pos : start_pos + a.shape[1]]
        c = d["coords"]
        i = c[INSERT_DOMAIN_NAME]  # transpose of on-disk array
        j = c[GENOME_DOMAIN_NAME] - start_pos  # coords are offset from start
        v = d[COUNTS_RANGE_NAME]
        a[i, j] = v

    def add_to_array(self, start_pos, a):
        """Add `vplot[:, start_pos:start_pos + a.shape[1]]` to `a`."""
        assert a.shape[0] == self.shape[0]
        assert start_pos + a.shape[1] < self.shape[1]
        assert start_pos >= 0

        d = self[:, start_pos : start_pos + a.shape[1]]
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

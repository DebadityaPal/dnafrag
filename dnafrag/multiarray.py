from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.sparse import coo_matrix

from .array import DNAFragArray
from .constants import GENOME_DOMAIN_NAME, COUNTS_RANGE_NAME, INSERT_DOMAIN_NAME


class DNAFragMultiArray(DNAFragArray):

    def __init__(self, arrays, subsample_rate=None):
        """Arrays should either be a list of DNAFragArrays, or a dictionary
        of `{array_i_path: subsampling_rate_i}`.

        If subsample_rate is specified, it is applied to the combined output,
        after any subsampling of individual arrays."""
        super(DNAFragMultiArray, self).__init__(
            array=None, subsample_rate=subsample_rate
        )

        self._arrays = []

        if not isinstance(arrays, dict):
            # First case: a list of tileDB arrays, DNAFragArrays, or array paths
            for arr in arrays:
                arr = DNAFragArray(arr)
                self._arrays.append(arr)
        else:
            # Second case: a dict of {array_path -> subsampling_rate}
            for array_path, subsample_rate in arrays.items():
                arr = DNAFragArray(array_path, subsample_rate=subsample_rate)
                self._arrays.append(arr)

    def __getitem__(self, key):
        ai = []
        aj = []
        av = []

        d = None

        for arr in self._arrays:
            d = arr[key]
            c = d["coords"]
            ai.append(c[GENOME_DOMAIN_NAME])
            aj.append(c[INSERT_DOMAIN_NAME])
            av.append(d[COUNTS_RANGE_NAME])

        ai = np.concatenate(ai)
        aj = np.concatenate(aj)
        av = np.concatenate(av)

        # Case where all of the arrays are empty for this slice
        if av.shape[0] == 0:
            return d

        row_offset = ai.min()
        ai = ai - row_offset

        a = coo_matrix((av, (ai, aj)), dtype=np.uint8)
        a = a.tocsc(copy=False).tocoo(copy=False)

        ret = {}

        ret["coords"] = {
            GENOME_DOMAIN_NAME: a.row + row_offset,
            INSERT_DOMAIN_NAME: a.col,
        }

        if self.subsample_rate is not None:
            v = self.subsample(a.data)
            ret[COUNTS_RANGE_NAME] = v

        else:
            ret[COUNTS_RANGE_NAME] = a.data

        return ret

    @property
    def shape(self):
        return self._arrays[0].shape

    @property
    def ndim(self):
        return self._arrays[0].ndim

import os.path
from itertools import groupby
from operator import itemgetter
import json
import glob
import os

import numpy as np
import pandas as ps
from pybedtools import BedTool
from six.moves import range
from scipy.sparse import coo_matrix, csc_matrix
import tiledb

from .array import DNAFragArray
from .context import context as ctx
from .constants import GENOME_DOMAIN_NAME, COUNTS_RANGE_NAME, INSERT_DOMAIN_NAME
from .multiarray import DNAFragMultiArray


DEFAULT_GENOME_TILE_EXTENT = 50000
DEFAULT_MAX_FRAGLEN = 400
DEFAULT_COMPRESSOR = "lz4"
VPLOT_MAX_VALUE = (2 ** 8) - 1  # we use unsigned 8-bit ints


def write_fragbed(fragment_bed, output_dir, genome_file, max_fraglen, overwrite=False):
    if os.path.exists(output_dir) and not overwrite:
        raise FileExistsError("Output directory {} exists".format(output_dir))

    chr2size = {}
    with open(genome_file, "r") as fp:
        for line in fp:
            chrom, size = line.split()
            chr2size[chrom] = int(size)

    file_shapes = {}  # only write file shapes for chroms we have data for

    # Assume that the file is SORTED so we can write out the matrices one
    # chromosome at a time if necessary
    frags = BedTool(fragment_bed)

    os.makedirs(output_dir)

    for chrom, intervals in groupby(frags, itemgetter(0)):
        print("Processing %s" % chrom)
        chr_dir = os.path.join(output_dir, chrom)
        chr_len = chr2size[chrom]
        file_shapes[chrom] = (max_fraglen, chr_len)

        data, row, col = [], [], []
        for interval in intervals:
            fraglen = interval.end - interval.start
            midpoint = int(0.5 * (interval.start + interval.end))

            if fraglen <= max_fraglen:
                data.append(1)
                row.append(fraglen - 1)
                col.append(midpoint)

        row = np.array(row, dtype=np.int32)
        col = np.array(col, dtype=np.int32)
        data = np.array(data, dtype=np.int32)

        # NB: We pass the *transpose* of the vplot
        # (so that the layout on disk is coordinate-major)
        write_sparse_array(chr_dir, chr_len, max_fraglen, col, row, data)

    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(
            {
                "file_shapes": file_shapes,
                "type": "vplot_tiledb",
                "source": fragment_bed,
            },
            fp,
        )


def write_sparse_array(path, n, m, n_idxs, m_idxs, values, clip=True):
    if os.path.exists(path):
        raise FileExistsError("{} already exists".format(path))

    if n_idxs.min() < 0 or n_idxs.max() >= n:
        raise ValueError("row indexes must be in range [0, n - 1]")

    if m_idxs.min() < 0 or m_idxs.max() >= m:
        raise ValueError("column indexes must in in range [0, m - 1]")

    sparse = coo_matrix((values, (n_idxs, m_idxs)), dtype=np.int32)
    sparse = sparse.tocsc(copy=False).tocoo(copy=False)

    n_idxs = sparse.row
    m_idxs = sparse.col
    values = sparse.data

    if clip:
        values = np.minimum(values, VPLOT_MAX_VALUE)

    if values.min() < 0 or values.max() > VPLOT_MAX_VALUE:
        raise ValueError(
            "vplot values must be in range [0, {}]".format(VPLOT_MAX_VALUE)
        )

    # ctx = tiledb.Ctx()

    n_tile_extent = min(DEFAULT_GENOME_TILE_EXTENT, n)

    d1 = tiledb.Dim(
        ctx, GENOME_DOMAIN_NAME, domain=(0, n - 1), tile=n_tile_extent, dtype="uint32"
    )
    d2 = tiledb.Dim(ctx, INSERT_DOMAIN_NAME, domain=(0, m - 1), tile=m, dtype="uint32")

    domain = tiledb.Domain(ctx, d1, d2)

    v = tiledb.Attr(ctx, "v", compressor=("lz4", -1), dtype="uint8")

    schema = tiledb.ArraySchema(
        ctx,
        domain=domain,
        attrs=(v,),
        capacity=1000,
        cell_order="row-major",
        tile_order="row-major",
        sparse=True,
    )

    tiledb.SparseArray.create(path, schema)

    with tiledb.SparseArray(ctx, path, mode="w") as A:
        values = values.astype(np.uint8)
        # A[n_idxs, m_idxs] = {"v": values}
        A[n_idxs, m_idxs] = values


def load(directory, subsample_rate=None, probe=False):
    assert os.path.isdir(directory)
    chrom_dirs = glob.glob(os.path.join(directory, "*"))
    chrom_dirs = filter(os.path.isdir, chrom_dirs)
    data = {
        os.path.basename(c): DNAFragArray(c, subsample_rate=subsample_rate, probe=probe)
        for c in chrom_dirs
    }
    return data


def load_multi(directories, probe=False):
    """directories: dictionary of `{array_i_path: subsampling_rate_i}`."""

    if len(directories.keys()) == 1:
        path, rate = next(iter(directories.items()))
        return load(path, subsample_rate=rate, probe=probe)

    chrom_dirs = glob.glob(os.path.join(next(iter(directories)), "*"))
    chrom_names = map(os.path.basename, filter(os.path.isdir, chrom_dirs))

    data = {}
    for chrom_name in chrom_names:
        chrom_arrs = {os.path.join(k, chrom_name): v for k, v in directories.items()}
        data[chrom_name] = DNAFragMultiArray(chrom_arrs, probe=probe)

    return data


def load_sparse_array(path):
    # ctx = tiledb.Ctx()
    return tiledb.SparseArray(ctx, path, mode="r")

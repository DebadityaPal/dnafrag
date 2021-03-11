from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import hub

from .array import DNAFragArray
from .context import context as ctx
from .constants import GENOME_DOMAIN_NAME, COUNTS_RANGE_NAME, INSERT_DOMAIN_NAME
from .multiarray import DNAFragMultiArray


DEFAULT_GENOME_TILE_EXTENT = 50000
DEFAULT_MAX_FRAGLEN = 400
DEFAULT_COMPRESSOR = "lz4"
VPLOT_MAX_VALUE = (2 ** 8) - 1  # we use unsigned 8-bit ints


def write_fragbed(fragment_bed, output_dir, genome_file, max_fraglen, overwrite=False, backend="tiledb"):
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
        if backend=="tiledb":
            write_sparse_array(chr_dir, chr_len, max_fraglen, col, row, data)
        elif backend=="hub":
            write_hub_dataset(chr_dir, chr_len, max_fraglen, col, row, data)

    if backend=="tiledb":
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
        GENOME_DOMAIN_NAME,
        domain=(0, n - 1),
        tile=n_tile_extent,
        dtype="uint32",
        ctx=ctx,
    )
    d2 = tiledb.Dim(
        INSERT_DOMAIN_NAME, domain=(0, m - 1), tile=m, dtype="uint32", ctx=ctx
    )

    domain = tiledb.Domain(d1, d2, ctx=ctx)

    v = tiledb.Attr(
        "v",
        filters=tiledb.FilterList([tiledb.LZ4Filter(level=-1)]),
        dtype="uint8",
        ctx=ctx,
    )

    schema = tiledb.ArraySchema(
        ctx=ctx,
        domain=domain,
        attrs=(v,),
        capacity=1000,
        cell_order="row-major",
        tile_order="row-major",
        sparse=True,
    )

    tiledb.SparseArray.create(path, schema)

    with tiledb.SparseArray(path, mode="w", ctx=ctx) as A:
        values = values.astype(np.uint8)
        # A[n_idxs, m_idxs] = {"v": values}
        A[n_idxs, m_idxs] = values


def write_hub_dataset(path, n, m, n_idxs, m_idxs, values, clip=True):
    if os.path.exists(path):
        raise FileExistsError("{} already exists".format(path))
    if n_idxs.min() < 0 or n_idxs.max() >= n:
        raise ValueError("row indexes must be in range [0, n - 1]")

    if m_idxs.min() < 0 or m_idxs.max() >= m:
        raise ValueError("column indexes must in in range [0, m - 1]")

    sparse = coo_matrix((values, (n_idxs, m_idxs)), dtype=np.int32)

    n_idxs = sparse.row
    m_idxs = sparse.col
    values = sparse.data

    if clip:
        values = np.minimum(values, VPLOT_MAX_VALUE)

    if values.min() < 0 or values.max() > VPLOT_MAX_VALUE:
        raise ValueError(
            "vplot values must be in range [0, {}]".format(VPLOT_MAX_VALUE)
        )
    n_tile_extent = min(DEFAULT_GENOME_TILE_EXTENT, n)

    schema = {
        "v": hub.schema.Tensor(len(values), "int64"),
        "row": hub.schema.Tensor(len(n_idxs), "int64"),
        "col": hub.schema.Tensor(len(m_idxs), "int64")
    }

    tag = path + "_hub"

    ds = hub.Dataset(tag, shape=(1,), schema=schema)
    ds["v", 0] = values
    ds["row", 0] = n_idxs
    ds["col", 0] = m_idxs

    ds.close()


def load(directory, subsample_rate=None, chroms=None):
    assert os.path.isdir(directory)

    if chroms is None:
        chrom_dirs = glob.glob(os.path.join(directory, "*"))
        chrom_dirs = filter(os.path.isdir, chrom_dirs)
    else:
        chrom_dirs = [os.path.join(directory, chrom) for chrom in chroms]

    data = {
        os.path.basename(c): DNAFragArray(c, subsample_rate=subsample_rate)
        for c in chrom_dirs
    }
    return data


def load_multi(directories, chroms=None):
    """directories: dictionary of `{array_i_path: subsampling_rate_i}`."""

    if len(directories.keys()) == 1:
        path, rate = next(iter(directories.items()))
        return load(path, subsample_rate=rate, chroms=chroms)

    if chroms is None:
        chrom_dirs = glob.glob(os.path.join(next(iter(directories)), "*"))
        chrom_names = map(os.path.basename, filter(os.path.isdir, chrom_dirs))
    else:
        chrom_names = chroms

    data = {}
    for chrom_name in chrom_names:
        chrom_arrs = {os.path.join(k, chrom_name): v for k, v in directories.items()}
        data[chrom_name] = DNAFragMultiArray(chrom_arrs)

    return data


def load_sparse_array(path):
    # ctx = tiledb.Ctx()
    return tiledb.SparseArray(path, mode="r", ctx=ctx)

# dnafrag: oprtimized storage for DNA fragmentation pattern V-plots
This module provides a compressed datastore for DNA fragmentation pattern V-plots.
The backend is [tiledb](https://www.tiledb.io), which is a compressed
column-oriented embedded database with support for sparse coordinate-format matrices.

The library is focused on efficient random interval query patterns for fragmentation
"V-plots", which are 2-D integer arrays of shape `(insert_size, query_range)`.
For typical applications, these are shape `(~300-400, ~1000-10000)` with sparsity
`~ 1e-5 - 1e-2`. The default compression/storage parameters are optimized
for these ranges, and they likely need to be modified for substantially different use cases.

## Dependencies
Dependencies can be installed in the current Anaconda environment with
`tools/install_dependencies.bash`.
`dnafrag` relies on [`Py-TileDB`](https://github.com/TileDB-Inc/TileDB-Py), which
currently supports Python 3.6 only. For this reason, `dnafrag` currently requires
Python 3.6.

## Format
The file format is simple: a directory of tiledb SparseArray files, with a single
metadata json file specifying the file shapes and format:

```
directory_root/
    chr1/
    chr2/
    ...
    chrN/
    metadata.json
```

Tiledb SparseArray files can be loaded with `tiledb.SparseArray.load` or
`dnafrag.core.load_sparse_array` (a wrapper for the former). The overall directory
can be loaded with `dnafrag.load` (see usage below).

## Example usage:
The input is a fragment bedfile, which is a TSV file where the first 3 columns
correspond to `(chrom, start, end)` (all other columns are ignored).
`dnafrag-from-fragbed` provides a command
line script to convert a fragment bed file to a dnafrag array:
```
$ dnafrag-from-fragbed --help
usage: dnafrag-from-fragbed [-h] [--max-fraglen MAX_FRAGLEN]
                            fragbed outdir genome

positional arguments:
  fragbed
  outdir
  genome

optional arguments:
  -h, --help            show this help message and exit
  --max-fraglen MAX_FRAGLEN
```

After conversion to a dnafrag array (a directory that contains tiledb files),
the array can be loaded with `dnafrag.load`:
```
array = dnafrag.load(output_dir)
```

The recommended query pattern is to use a `numpy.ndarray` as the buffer to query
the dnafrag array:

```
data = dnafrag.load(output_dir)

# data is dict of {chrom (str) -> array (dnafrag.DNAFragArray)}:
data_chr1 = data['chr1']

# Allocate a numpy.ndarray to query the dnafrag file
a = np.zeros((MAX_INTERVAL_LEN, 100))

# The coordinate where the query should start
start_coord = 0

data_chr1.fill_array(start_coord, a)
```

## Tests
Tests are in `dnafrag/tests` and can be run with `pytest` (`dnafrag` must be
installed in the current python environment)

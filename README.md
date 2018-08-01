# dnafrag: optimized storage for DNA fragmentation V-plots
This module provides a compressed datastore for DNA fragmentation pattern V-plots.
The backend is [tiledb](https://www.tiledb.io), which is a compressed
column-oriented embedded database with support for sparse coordinate-format matrices.

The library is focused on efficient random interval query patterns for fragmentation
"V-plots", which are 2-D integer arrays of shape `(insert_size, query_range)`.
For typical applications, these are shape `(~300-400, ~1000-10000)` with sparsity
`~ 1e-6 - 1e-2`. The default compression/storage parameters are optimized
for these ranges, and they likely need to be modified for substantially different use cases.

## Installation
The requirements are `numpy`, `scipy`, `tqdm`, and `tiledb`.

Install `dnafrag` with `python setup.py install`, or `pip install .`

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
```python
array = dnafrag.load(output_dir)
```

The recommended query pattern is to use a `numpy.ndarray` as the buffer to query
the dnafrag array:

```python
data = dnafrag.load(output_dir)

# data is dict of {chrom (str) -> array (dnafrag.DNAFragArray)}:
data_chr1 = data['chr1']

# Allocate a numpy.ndarray to query the dnafrag file
a = np.zeros((MAX_INTERVAL_LEN, 100))

# The coordinate where the query should start
start_coord = 0

data_chr1.fill_array(start_coord, a)
```

## TensorFlow sparse tensor integration
One of the slowest operations in tensorflow is feeding large numpy arrays into
the graph, because they are copied and serialized as protocolbuffers, and then
copied and deserialized into tensor buffers. With sparse vplots arrays,
one can improve performance using `tf.sparse_to_dense`, which allows feeding
coordinates of non-zero entries. This works well with the dnafrag API:

```python
# Shape: [batch_index, row_index, column_index]
sparse_indices_ph = tf.placeholder(tf.int32, shape=[None, 3])

# Shape: [value_index]
sparse_values_ph = tf.placeholder(tf.int32, shape=[None])

Xi = tf.sparse_to_dense(
    sparse_indices=sparse_indices_ph,
    output_shape=[batch_size, vplot_shape[0], input_width],
    sparse_values=sparse_values_ph,
    default_value=0,
    validate_indices=False,
)

X = Xi[:, :, :, None]

# Optionally clip the V-plot values
if maxvalue is not None:
    X = tf.minimum(X, maxvalue)

X = tf.cast(X, tf.float32)

# This example is constant filter cross-correlation, but regular conv2d layers work too
f = tf.constant(kernel[:, :, None, None])
y = tf.squeeze(tf.nn.conv2d(X, f, [1] * 4, "VALID"))
```

Then, to run:

```python
sess.run(tf.global_variables_initializer())
run_conv = sess.make_callable(y, [sparse_indices_ph, sparse_values_ph])

d = vplot[:, start:stop]
c = d["coords"]
i = c[dnafrag.constants.INSERT_DOMAIN_NAME]
j = c[dnafrag.constants.GENOME_DOMAIN_NAME] - start
v = d[dnafrag.constants.COUNTS_RANGE_NAME]

sp_indices = np.hstack([i[:, None], j[:, None]])

out_start = p - conv_halfwidth
out_stop = p + conv_halfwidth + 1

output[out_start:out_stop] = run_conv(sp_indices, v)
```

## Tests
Tests are in `dnafrag/tests` and can be run with `pytest` (`dnafrag` must be
installed in the current python environment)

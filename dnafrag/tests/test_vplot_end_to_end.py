import os

import numpy as np

import dnafrag

MODULE_BASE_PATH = os.path.abspath(os.path.dirname(dnafrag.__file__))

FRAGBED_PATH = os.path.join(MODULE_BASE_PATH, "tests/test_fragbed_100k.fragbed.gz")
CHRSZ_PATH = os.path.join(MODULE_BASE_PATH, "tests/test_fragbed_hg19.chrom.sizes")


def test_vplot_end_to_end(tmpdir):
    dest = os.path.join(tmpdir, "100kfragbed_array")

    assert not os.path.exists(dest)

    run_extract = "dnafrag-from-fragbed {} {} {}".format(FRAGBED_PATH, dest, CHRSZ_PATH)
    os.system(run_extract)

    assert os.path.isdir(dest)

    data = dnafrag.load(dest)
    assert len(data.keys()) == 1
    assert "chr1" in data.keys()
    data = data["chr1"]

    assert data[0, 0]["v"].shape[0] == 0
    assert data[:1000, :1000]["v"].shape[0] == 0

    assert data[:, :]["v"].sum() == 100000

    assert data[0, 0]["v"].shape[0] == 0
    assert data[:1000, :1000]["v"].shape[0] == 0

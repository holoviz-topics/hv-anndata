from __future__ import annotations

import holoviews as hv
import pandas as pd

from hv_anndata import labeller


def test_labeller() -> None:
    df = pd.DataFrame({
        "UMAP1": [0, 1, 2, 3, 10],
        "UMAP2": [0, 1, 2, 3, 10],
        "cell_type": ["a", "a", "b", "b", "b"],
    })
    dataset = hv.Dataset(df, kdims=["UMAP1", "UMAP2"], vdims=("cell_type"))
    ldm = labeller(dataset, min_count=0)
    labels = ldm[()]
    expected_data = pd.DataFrame({
        "cell_type": ["b", "a"],
        "count": [3, 2],
        "x": [5, 0.5],
        "y": [5, 0.5],
    })
    pd.testing.assert_frame_equal(
        labels.data.sort_values("cell_type"),
        expected_data.sort_values("cell_type"),
    )

#!/usr/bin/env python3
"""Verify the bundled Perturb-FISH cancer cohort before training."""
import argparse
import hashlib
from pathlib import Path

import anndata
import numpy as np
from scipy import sparse


FILES = {
    "fish_cancer_raw.h5ad": "3d14674de211f7cd1f446ae947b2c10f7dc119722286d98d05f359c74cde6c00",
    "fish_coordinates.csv": "4aade2a964c6cf56dc448ff1e2817899940c8a01504e9de12a2c2d707745a5b6",
}


def prepare(directory):
    for name, expected in FILES.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled data: {path}. Clone the complete experiment2 repository including data/.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Bundled file checksum mismatch: {path}; no files were overwritten")
        print(f"SHA-256 verified: {path}", flush=True)
    data = anndata.read_h5ad(directory / "fish_cancer_raw.h5ad")
    coordinates = np.loadtxt(directory / "fish_coordinates.csv", delimiter=",")
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X)
    if data.shape != (7496, 500) or coordinates.shape != (7496, 2):
        raise ValueError("Unexpected bundled dataset/coordinate shape")
    if not np.isfinite(values).all() or (values < 0).any() or not np.isfinite(coordinates).all():
        raise ValueError("Invalid expression/coordinate values")
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Duplicate cell/gene IDs")
    eligible = data.obs["perturbation"].eq("Control")
    if int(eligible.sum()) != 3452 or data.obs["celltype2"].isna().any():
        raise ValueError("Unexpected bundled metadata")
    print(f"Ready: cells={data.n_obs}, genes={data.n_vars}, eligible_receivers={int(eligible.sum())}")
    print("Coordinates are row-aligned by the verified bundled file version. Tissue identity is not verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    prepare(parser.parse_args().data_dir)

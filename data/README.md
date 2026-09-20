# Bundled FISH data

These two files are the prepared Perturb-FISH cancer cohort already used and
verified in Exp1 and Exp2. Commit both files with the code; they total about 1.8 MB
and do not need Git LFS. They are deliberately not excluded by `.gitignore`.

- `fish_cancer_raw.h5ad`: 7,496 cells × 500 genes, raw counts and cell metadata.
- `fish_coordinates.csv`: 7,496 × 2 coordinates in exactly the same cell order.

Local provenance: `experiment1/data/`, commit
`f6fe00931b91abd2e22b623fd026df0887a91c23`. The public raw GitHub URLs for that
repository returned 404 when checked on 2026-09-20, so runtime setup does not
depend on that repository or claim it is publicly downloadable.

`python scripts/prepare_fish.py` verifies SHA-256 checksums, dimensions, finite
nonnegative expression and the expected 3,452 Control receivers. This is a
verification step: no additional preprocessing or separate train/test file is
needed. It does not verify the single-tissue assumption or infer tissue IDs.

This is the prepared cancer subset, not the complete Perturb-FISH release, raw
images, or a new derivation of tissue metadata. Gene modules and receiver splits
are constructed later by Exp2 training using its training-only rules.

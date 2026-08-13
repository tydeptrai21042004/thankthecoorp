# DT1D WHC Kaggle package — fresh standalone cells

This package uses the **current uploaded code snapshots** and the renamed repositories:

- CNN / dense: `https://github.com/tydeptrai21042004/dt1d.git`
- ViT: `https://github.com/tydeptrai21042004/DT1D-vit.git`

## Standalone rule

Every **training** `.sh` / Kaggle code cell is independent. Paste any one training cell into a new Kaggle notebook with **Internet ON** and **GPU ON**. The cell performs, in this order:

1. checks Python/git,
2. fresh-clones the correct GitHub repository (3 retry attempts),
3. optionally checks out `DT1D_CNN_COMMIT` or `DT1D_VIT_COMMIT`,
4. SHA-256 validates the cloned implementation against the current uploaded source snapshot,
5. installs the dependencies required by that repository,
6. runs the proposal/source tests and GPU check,
7. restores only its own optional result ZIP for resume,
8. downloads/prepares its own dataset(s),
9. downloads/caches the pretrained backbone/checkpoint needed by that cell,
10. generates and validates the exact execution plan,
11. dry-runs the generated configurations,
12. runs training on up to two GPUs,
13. aggregates and writes `<SESSION>_results.zip`.

The data/model roots are session-specific (`/kaggle/working/data_<SESSION>` and, for ViT, `/kaggle/working/models_<SESSION>`), so no training cell needs a dataset or model cache produced by another cell.

## DRIVE exception (P02 only)

The current dense code uses the historical DRIVE layout with vessel annotations under both `training/1st_manual` and `test/1st_manual`. DRIVE distribution requires controlled/account access, so the package deliberately does **not** hard-code an unofficial mirror. P02 is still independent of every other cell, but you must provide DRIVE to that cell in one of these ways:

- attach it as a Kaggle Input using the expected layout; or
- set `DRIVE_DATA_DIR=/path/to/DRIVE`; or
- set `DRIVE_ARCHIVE_URL=<your authorized direct zip URL>`.

P02 downloads PennFudan itself and downloads the required pretrained model weights.

## Optional exact Git commit pin

Before a CNN/dense cell:

```bash
export DT1D_CNN_COMMIT=<exact sha>
```

Before a ViT cell:

```bash
export DT1D_VIT_COMMIT=<exact sha>
```

If no commit is supplied, the cell uses the current default branch but still performs the SHA-256 source-snapshot check. A mismatch fails before GPU training; only set `DT1D_ALLOW_SOURCE_MISMATCH=1` if you intentionally changed the repository after this package was generated.

## Session count and splitting

The compact grouping is preserved: experiments are combined when the estimate remains within a Kaggle 12-hour session and split only for the workloads estimated to exceed that limit. Every session has an 11 h 55 min safety cutoff and writes a resumable ZIP.

`FINAL_MERGE_ALL_RESULTS.sh` is intentionally different: it is an aggregation-only cell, so it does not clone a training repo or download datasets. Attach the result ZIPs and run the merge after the training sessions are complete.

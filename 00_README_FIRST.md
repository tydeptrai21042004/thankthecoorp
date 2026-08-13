# DT1D WHC Kaggle standalone v3

This package fixes the second V01 failure observed on Kaggle. The prior cell reached the ViT checkpoint download but `$MODEL_ROOT` had not been created, so `curl` failed with exit code 23.

## v3 fix
- P03, V01 and V02 now create `OUTPUT_ROOT`, `DATA_ROOT`, and `MODEL_ROOT` before any download.
- The 330 MB ViT-B/16 checkpoint is downloaded atomically to `*.part` and renamed only after a successful transfer.
- Every notebook training cell is also exported under `sessions/` as a standalone `.sh` file.
- Every session remains fresh-session standalone: clone -> install -> validate -> GPU check -> optional restore -> download dataset/weights -> dry-run -> train -> aggregate -> ZIP.

For the failed run, use `HOTFIX_V01_vtab_dtd_all5_methods.sh` in a fresh Kaggle session. No previous V01 result ZIP is useful because training had not started.

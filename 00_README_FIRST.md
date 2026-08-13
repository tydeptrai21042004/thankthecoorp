# DT1D WHC compact Kaggle rerun package — 12-hour split policy

This is the compact version of the previous 45-session package.

## Split rule

- A workload is **kept in one Kaggle Bash cell whenever the conservative 2xT4 estimate is <= 12 hours**.
- It is split only when the estimated wall time would exceed 12 hours.
- Every cell has an internal **11 h 55 min safety deadline**, writes a resumable `*_results.zip`, and skips already-completed runs when that ZIP is attached on a retry.
- Estimates are based on the previous package's per-session 2xT4 estimates. For merged cells I use the **sum of the previous estimates**, so the estimate is conservative because clone/install/dataset setup is now shared.

## Result

- Previous package: **45 training `.sh` files**.
- This package: **23 training `.sh` files** + 1 final merge cell.
- Conservative summed wall-time estimates across all work: **202.6 session-hours**. This is not sequential elapsed time if you run Kaggle sessions in parallel.

## Sessions near the 12-hour ceiling

`C08`, `C09`, `C10`, `C11`, `V01`, and `V02` are deliberately packed close to the 12-hour limit. Runtime can vary by Kaggle VM/load, so the resumable ZIP mechanism is retained. `V01/V02` are 12.0 h by the old summed estimates, but merging removes duplicated clone/install/setup overhead and should bring the real wall time below that sum.

## Main grouping

- `A01-A02`: one ablation/hyper session per dataset, all 3 seeds together.
- `P01`: all updated-proposal-only CNN reruns from existing experiments (~7.4 h).
- `P02`: all updated-proposal-only binary dense reruns (~10.4 h).
- `P03`: all updated-proposal-only existing ViT reruns (~9.0 h).
- `C01-C07`: full CNN experiments that fit below 12 h are each a single session.
- `C08-C10`: SVHN is split into the minimum 3 balanced sessions because the full estimate is ~35 h.
- `C11-C12`: Food-101/EfficientNet-B0 is split into 2 sessions because the full estimate is ~21.6 h.
- `C13-C14`: Food-101/ResNet-18 is split into 2 sessions because the full estimate is ~19.0 h.
- `D01-D02`: semantic segmentation and detection each already fit below 12 h.
- `V01-V02`: one all-five-method session per new VTAB dataset.

Repositories:

```text
tydeptrai21042004/dt1d
tydeptrai21042004/DT1D-vit
```

Optional exact source pinning before a cell:

```bash
export DT1D_CNN_COMMIT="<final CNN commit>"
export DT1D_VIT_COMMIT="<final ViT commit>"
```

Run cells in `RUN_ORDER.csv`. When one is interrupted, attach that cell's result ZIP as Kaggle Input and rerun the same cell.

#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: V02 | VTAB-EuroSAT / ViT-B/16 / BS32 — all 5 methods | all methods in one <=12h session
# Methods: whc_dt1d,vpt,pfeiffer,full,linear
# Fair protocol: 10 LR candidates/method on tune seed 42 -> final seeds 0/1/2 -> test once at best-validation checkpoint.
# Estimated 2xT4 wall time: ~11.50 h conservative (shared setup should reduce this); hard cutoff is 11 h 55 min with resumable ZIP.

SESSION_ID="V02"; DATASET="vtab-eurosat"; BATCH_SIZE="32"; METHOD="whc_dt1d,vpt,pfeiffer,full,linear"; VPT_TOKENS="10"
REPO_URL="https://github.com/tydeptrai21042004/DT1D-vit.git"; REPO_COMMIT="${DT1D_VIT_COMMIT:-}"; WORKDIR="/kaggle/working";REPO_DIR="$WORKDIR/DT1D-vit-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID";MODEL_ROOT="$WORKDIR/models_$SESSION_ID";OUTPUT_ROOT="$WORKDIR/vit_$SESSION_ID";RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID DATASET BATCH_SIZE METHOD VPT_TOKENS REPO_DIR DATA_ROOT MODEL_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

pack_results() {
  if [[ -d "$OUTPUT_ROOT" ]]; then
    python - <<'PYZIP'
import os,zipfile
from pathlib import Path
root=Path(os.environ['OUTPUT_ROOT']);dst=Path(os.environ['RESULT_ZIP'])
if dst.exists():dst.unlink()
with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() not in {'.pth','.pt','.ckpt'}:
            z.write(p,p.relative_to(root.parent))
print('RESULT_ZIP=',dst)
PYZIP
    ls -lh "$RESULT_ZIP" || true
  fi
}
trap 'rc=$?; trap - EXIT; if [[ ! -f "$RESULT_ZIP" ]]; then pack_results || true; fi; exit $rc' EXIT

# 1) Fresh clone of the UPDATED repository. Set DT1D_VIT_COMMIT=<sha> to pin an exact Git commit.
# 0) Fresh-session prerequisites. Kaggle: Internet ON + GPU accelerator ON.
command -v git >/dev/null || { echo "ERROR: git is unavailable" >&2; exit 2; }
command -v python >/dev/null || { echo "ERROR: python is unavailable" >&2; exit 2; }
python -V
echo "BOOTSTRAP SESSION=$SESSION_ID REPO=$REPO_URL"
for _clone_try in 1 2 3; do
  rm -rf "$REPO_DIR"
  if git clone --depth 1 "$REPO_URL" "$REPO_DIR"; then break; fi
  echo "git clone attempt $_clone_try failed; retrying..." >&2
  sleep $((_clone_try*5))
done
[[ -d "$REPO_DIR/.git" ]] || { echo "ERROR: failed to clone $REPO_URL" >&2; exit 2; }
cd "$REPO_DIR"
if [[ -n "$REPO_COMMIT" ]]; then
  git fetch --depth 1 origin "$REPO_COMMIT"
  git checkout --detach "$REPO_COMMIT"
  [[ "$(git rev-parse HEAD)" == "$REPO_COMMIT" ]]
fi
SOURCE_COMMIT="$(git rev-parse HEAD)"; export SOURCE_COMMIT
echo "SOURCE_COMMIT=$SOURCE_COMMIT"
python - <<'PYSOURCE'
import hashlib, os
from pathlib import Path
expected={
    "configs/ablations/whc_p2_fixed_gate_vit.yaml": "ead24bebaa3c5bf9c973bac97659154481cc90cdd00fa4f20cb447eec4aa2768",
    "run_fair_vit_comparison.py": "216a4fd69b628628ad29bcd9f35d27f2d22259e2d4a76d6f4d722298b0a85dd2",
    "run_whc_p2_vit_ablation.py": "d36a28362f2ecf977310d1eb24700b9852a8370a7d72ff49b7df69a8d837faa1",
    "src/configs/config.py": "8726621300ce3dbf724247871874d978b3c2378cf81f6d214181ad0ed04a7cd1",
    "src/models/vit_adapter/whc_compact_dt1d_adapter.py": "710b03d3b98fcc32cbe6145bcbd2a80aea2683b2a76a4008282408499c101b92"
}
bad=[]
for rel,want in expected.items():
    p=Path(rel); got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub DT1D-vit source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('VIT SOURCE SNAPSHOT PASS')
PYSOURCE

python -m pip install -q --upgrade-strategy only-if-needed scipy scikit-learn pandas Pillow fvcore iopath yacs simplejson termcolor tabulate tqdm ml-collections 'timm>=1.0.0,<2' PyYAML tensorflow-datasets six
if ! python - <<'PYTFIMPORT'
import tensorflow
print('tensorflow:', tensorflow.__version__)
PYTFIMPORT
then
  python -m pip install -q 'tensorflow>=2.16,<2.20'
fi
python - <<'PYTFIMPORT2'
import tensorflow, tensorflow_datasets
print('tensorflow/tfds import: PASS')
PYTFIMPORT2
python validate_whc_p2_vit.py
python -m py_compile run_fair_vit_comparison.py train.py verify_fair_protocol.py verify_vpt_original.py
python -m pytest -q tests/test_whc_compact_dt1d_token_adapter.py tests/test_fair_protocol.py tests/test_dt1d_token_adapter.py
python verify_vpt_original.py
python verify_fair_protocol.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# Restore best previous session ZIP.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input');zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0;summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            ns=f.namelist();summaries=sum(n.endswith('run_summary.json') or n.endswith('_fair_three_seed.csv') for n in ns)
            for n in ns:
                if n.endswith('SESSION_STATUS.json'):
                    try:complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                    except Exception:pass
    except Exception:pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'vit_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT" "$MODEL_ROOT"

# Add the two additional VTAB dataset registry entries at runtime only; model/baseline implementations remain unchanged.
python - <<'PYPATCH'
from pathlib import Path
p=Path('run_fair_vit_comparison.py');s=p.read_text()
if '"vtab-dtd"' not in s:
    NL='\n';anchor=NL*3+'# SHA256 hashes';idx=s.find(anchor);assert idx!=-1,'DATASETS terminator changed'
    head=s[:idx];close=head.rfind('}');assert close!=-1
    lines=[
      '    "vtab-dtd": {"name":"vtab-dtd","classes":47,"default_batches":(32,),"vpt_tokens":10,"protocol":"train800->val200->official-test@best-val"},',
      '    "vtab-eurosat": {"name":"vtab-eurosat","classes":10,"default_batches":(32,),"vpt_tokens":10,"protocol":"train800->val200->official-test@best-val"},',
    ]
    s=head[:close]+NL.join(lines)+NL+head[close:]+s[idx:];p.write_text(s)
print('VTAB registry patch PASS')
PYPATCH

# Exact ViT-B/16 checkpoint.
WEIGHT_FILE="$MODEL_ROOT/ViT-B_16-224.npz"
if [[ ! -s "$WEIGHT_FILE" ]];then curl -L --fail --retry 5 --retry-delay 5 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz' -o "$WEIGHT_FILE";fi
python - "$WEIGHT_FILE" <<'PYWEIGHT'
import sys,numpy as np
z=np.load(sys.argv[1]);assert len(z.files)==200,len(z.files);print('ViT weight tensors=',len(z.files))
PYWEIGHT

# Dataset pre-download.
if [[ "$DATASET" == "flowers102" ]];then
 FLOWERS_ROOT="$DATA_ROOT/flowers_download";mkdir -p "$FLOWERS_ROOT"
 DATA_PATH="$(python - "$FLOWERS_ROOT" <<'PYFLOWERS'
import json,sys
from pathlib import Path
from torchvision.datasets import Flowers102
root=Path(sys.argv[1]);ds={sp:Flowers102(str(root),split=sp,download=True) for sp in ('train','val','test')};image=root/'flowers-102'/'jpg';assert image.is_dir();expected={'train':1020,'val':1020,'test':6149}
for sp,d in ds.items():
 m={Path(str(p)).name:int(y) for p,y in zip(d._image_files,d._labels)};assert len(m)==expected[sp];(image/f'{sp}.json').write_text(json.dumps(m))
print(image)
PYFLOWERS
 )"
else
 DATA_PATH="$DATA_ROOT/tfds";mkdir -p "$DATA_PATH";export DATA_PATH
 python - <<'PYTFDS'
import os,tensorflow_datasets as tfds
spec={'vtab-caltech101':'caltech101:3.*.*','vtab-dtd':'dtd:3.*.*','vtab-eurosat':'eurosat/rgb:2.*.*'}[os.environ['DATASET']]
b=tfds.builder(spec,data_dir=os.environ['DATA_PATH']);b.download_and_prepare();print('TFDS READY',b.info.full_name)
PYTFDS
fi
export DATA_PATH

# Exact protocol dry-run.
PREFLIGHT="$WORKDIR/_preflight_$SESSION_ID";rm -rf "$PREFLIGHT"
python run_fair_vit_comparison.py --dataset "$DATASET" --data-path "$DATA_PATH" --model-root "$MODEL_ROOT" --output-root "$PREFLIGHT" --batch-sizes "$BATCH_SIZE" --methods "$METHOD" --epochs 10 --resolution 224 --seeds 0,1,2 --tune-seed 42 --weight-decay 1e-4 --warmup-epoch 1 --patience 20 --vpt-tokens "$VPT_TOKENS" --allow-boundary-best --gpus cpu --dry-run
rm -rf "$PREFLIGHT"

# Real fair run as its own process group for safe time-cap termination.
LOG="$OUTPUT_ROOT/session.log";mkdir -p "$OUTPUT_ROOT";set +e
setsid python run_fair_vit_comparison.py --dataset "$DATASET" --data-path "$DATA_PATH" --model-root "$MODEL_ROOT" --output-root "$OUTPUT_ROOT" --batch-sizes "$BATCH_SIZE" --methods "$METHOD" --epochs 10 --resolution 224 --seeds 0,1,2 --tune-seed 42 --weight-decay 1e-4 --warmup-epoch 1 --patience 20 --vpt-tokens "$VPT_TOKENS" --allow-boundary-best --gpus auto >"$LOG" 2>&1 &
RUN_PID=$!
while kill -0 "$RUN_PID" 2>/dev/null;do
 if (( $(date +%s) >= DEADLINE_EPOCH ));then echo "TIME CAP reached; terminating fair-run process group" | tee -a "$LOG";kill -TERM -- "-$RUN_PID" 2>/dev/null || true;sleep 30;kill -KILL -- "-$RUN_PID" 2>/dev/null || true;break;fi
 sleep 10
done
wait "$RUN_PID";RUN_RC=$?;set -e

python - <<'PYSTATUS'
import json,os,pandas as pd
from pathlib import Path
out=Path(os.environ['OUTPUT_ROOT']);expected={x for x in os.environ['METHOD'].split(',') if x};bs=int(os.environ['BATCH_SIZE']);cp=out/'aggregated'/f"{os.environ['DATASET'].replace('-','_')}_fair_three_seed.csv"
complete=False;actual=set()
if cp.is_file():
    df=pd.read_csv(cp);df=df[df['batch_size'].astype(int)==bs];actual=set(df['method_key'].astype(str));complete=(actual==expected and len(df)==len(expected))
status={'session':os.environ['SESSION_ID'],'family':'vit','dataset':os.environ['DATASET'],'methods':sorted(expected),'batch_size':bs,'actual_methods':sorted(actual),'complete':bool(complete),'source_commit':os.environ.get('SOURCE_COMMIT')}
(out/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
PYSTATUS
{ echo "session=$SESSION_ID";echo "family=vit";echo "dataset=$DATASET";echo "methods=$METHOD";echo "batch_size=$BATCH_SIZE";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$RUN_RC" -eq 0 ]] && python - <<'PYCHECK'
import json,os,sys
from pathlib import Path
p=Path(os.environ['OUTPUT_ROOT'])/'SESSION_STATUS.json';sys.exit(0 if json.loads(p.read_text()).get('complete') else 1)
PYCHECK
then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun the SAME cell; compatible completed tuning/final runs are skipped.";fi
exit 0

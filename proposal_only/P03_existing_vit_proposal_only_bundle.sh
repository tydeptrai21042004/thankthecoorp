#!/usr/bin/env bash
set -Eeuo pipefail
# Standalone Kaggle cell: P03 | proposal-only ViT fair reruns: Flowers102 BS16+BS32 and VTAB-Caltech101 BS32
# Method whc_dt1d only; 10 LR candidates on tune seed 42 -> final seeds 0,1,2 -> test at best-val checkpoint.
# Conservative summed estimate ~9.0 h on 2xT4; shared setup reduces overhead. Hard cutoff 11 h 55 min; resumable.
SESSION_ID="P03";METHOD="whc_dt1d";REPO_URL="https://github.com/tydeptrai21042004/DT1D-vit.git";REPO_COMMIT="${DT1D_VIT_COMMIT:-}";WORKDIR="/kaggle/working";REPO_DIR="$WORKDIR/DT1D-vit-$SESSION_ID";DATA_ROOT="$WORKDIR/vit_shared_data";MODEL_ROOT="$WORKDIR/vit_weights";OUTPUT_ROOT="$WORKDIR/vit_$SESSION_ID";RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip";CELL_START_EPOCH="$(date +%s)";DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))";export SESSION_ID METHOD REPO_DIR DATA_ROOT MODEL_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
pack_results() { if [[ -d "$OUTPUT_ROOT" ]];then python - <<'PYZIP'
import os,zipfile
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);d=Path(os.environ['RESULT_ZIP']);d.unlink(missing_ok=True)
with zipfile.ZipFile(d,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in r.rglob('*'):
  if p.is_file() and p.suffix.lower() not in {'.pth','.pt','.ckpt'}:z.write(p,p.relative_to(r.parent))
print(d)
PYZIP
ls -lh "$RESULT_ZIP" || true;fi; }
trap 'rc=$?; trap - EXIT; [[ -f "$RESULT_ZIP" ]] || pack_results || true; exit $rc' EXIT
rm -rf "$REPO_DIR";git clone --depth 1 "$REPO_URL" "$REPO_DIR";cd "$REPO_DIR";if [[ -n "$REPO_COMMIT" ]];then git fetch --depth 1 origin "$REPO_COMMIT";git checkout --detach "$REPO_COMMIT";[[ "$(git rev-parse HEAD)" == "$REPO_COMMIT" ]];fi;SOURCE_COMMIT="$(git rev-parse HEAD)";export SOURCE_COMMIT
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
 p=Path(rel);got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
 if got!=want:bad.append((rel,want,got))
if bad:
 print('SOURCE SNAPSHOT MISMATCH:',bad)
 if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0')!='1':raise SystemExit('GitHub DT1D-vit source mismatch')
print('VIT SOURCE SNAPSHOT PASS')
PYSOURCE
python -m pip install -q --upgrade-strategy only-if-needed scipy scikit-learn pandas Pillow fvcore iopath yacs simplejson termcolor tabulate tqdm ml-collections 'timm>=1.0.0,<2' PyYAML tensorflow-datasets six
python validate_whc_p2_vit.py;python -m py_compile run_fair_vit_comparison.py train.py verify_fair_protocol.py verify_vpt_original.py;python -m pytest -q tests/test_whc_compact_dt1d_token_adapter.py tests/test_fair_protocol.py tests/test_dt1d_token_adapter.py;python verify_vpt_original.py;python verify_fair_protocol.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available();print('GPU count=',torch.cuda.device_count())
PYGPU
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);zs=list(Path('/kaggle/input').rglob(f'{sid}_results.zip')) if Path('/kaggle/input').exists() else []
def score(z):
 c=s=0
 try:
  with zipfile.ZipFile(z) as f:
   ns=f.namelist();s=sum(n.endswith('_fair_three_seed.csv') or n.endswith('run_summary.json') for n in ns)
   for n in ns:
    if n.endswith('SESSION_STATUS.json'):
     try:c=max(c,int(bool(json.loads(f.read(n)).get('complete'))))
     except:pass
 except:pass
 return(c,s,z.stat().st_mtime)
if zs:
 z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);f=list(tmp.rglob(f'vit_{sid}'))
 if len(f)==1:shutil.move(str(f[0]),str(out));print('RESTORED',z,score(z))
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT" "$MODEL_ROOT"
WEIGHT_FILE="$MODEL_ROOT/ViT-B_16-224.npz";if [[ ! -s "$WEIGHT_FILE" ]];then curl -L --fail --retry 5 --retry-delay 5 'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz' -o "$WEIGHT_FILE";fi
python - "$WEIGHT_FILE" <<'PYW'
import sys,numpy as np
z=np.load(sys.argv[1]);assert len(z.files)==200;print('weights OK')
PYW
FLOWERS_ROOT="$DATA_ROOT/flowers_download";mkdir -p "$FLOWERS_ROOT";FLOWERS_PATH="$(python - "$FLOWERS_ROOT" <<'PYF'
import json,sys
from pathlib import Path
from torchvision.datasets import Flowers102
r=Path(sys.argv[1]);ds={sp:Flowers102(str(r),split=sp,download=True) for sp in ('train','val','test')};im=r/'flowers-102'/'jpg';exp={'train':1020,'val':1020,'test':6149}
for sp,d in ds.items():
 m={Path(str(p)).name:int(y) for p,y in zip(d._image_files,d._labels)};assert len(m)==exp[sp];(im/f'{sp}.json').write_text(json.dumps(m))
print(im)
PYF
)"
TFDS_ROOT="$DATA_ROOT/tfds";mkdir -p "$TFDS_ROOT";export TFDS_ROOT
python - <<'PYTF'
import os,tensorflow_datasets as tfds
b=tfds.builder('caltech101:3.*.*',data_dir=os.environ['TFDS_ROOT']);b.download_and_prepare();print('TFDS READY',b.info.full_name)
PYTF
run_fair() {
 DATASET="$1";DATA_PATH="$2";BATCHES="$3";VPT_TOKENS="$4";export DATASET DATA_PATH BATCHES VPT_TOKENS;echo "===== P03 $DATASET BS=$BATCHES ====="
 PREF="$WORKDIR/_preflight_${SESSION_ID}_${DATASET}";rm -rf "$PREF";python run_fair_vit_comparison.py --dataset "$DATASET" --data-path "$DATA_PATH" --model-root "$MODEL_ROOT" --output-root "$PREF" --batch-sizes "$BATCHES" --methods "$METHOD" --epochs 10 --resolution 224 --seeds 0,1,2 --tune-seed 42 --weight-decay 1e-4 --warmup-epoch 1 --patience 20 --vpt-tokens "$VPT_TOKENS" --allow-boundary-best --gpus cpu --dry-run;rm -rf "$PREF"
 LOG="$OUTPUT_ROOT/${DATASET}.log";set +e;setsid python run_fair_vit_comparison.py --dataset "$DATASET" --data-path "$DATA_PATH" --model-root "$MODEL_ROOT" --output-root "$OUTPUT_ROOT" --batch-sizes "$BATCHES" --methods "$METHOD" --epochs 10 --resolution 224 --seeds 0,1,2 --tune-seed 42 --weight-decay 1e-4 --warmup-epoch 1 --patience 20 --vpt-tokens "$VPT_TOKENS" --allow-boundary-best --gpus auto >"$LOG" 2>&1 & PID=$!
 while kill -0 "$PID" 2>/dev/null;do if (( $(date +%s) >= DEADLINE_EPOCH ));then kill -TERM -- "-$PID" 2>/dev/null || true;sleep 30;kill -KILL -- "-$PID" 2>/dev/null || true;break;fi;sleep 10;done;wait "$PID";RC=$?;set -e;return "$RC"
}
OVERALL=0;run_fair flowers102 "$FLOWERS_PATH" "16,32" 5 || OVERALL=$?;if [[ "$OVERALL" -eq 0 ]];then run_fair vtab-caltech101 "$TFDS_ROOT" "32" 5 || OVERALL=$?;fi
python - <<'PYSTATUS'
import os,json,pandas as pd
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);checks=[('flowers102',[16,32]),('vtab-caltech101',[32])];rows=[]
for ds,bss in checks:
 cp=r/'aggregated'/f"{ds.replace('-','_')}_fair_three_seed.csv";ok=False;actual=[]
 if cp.is_file():
  df=pd.read_csv(cp);sub=df[df['method_key'].astype(str)==os.environ['METHOD']];actual=sorted(set(sub['batch_size'].astype(int)));ok=actual==bss and len(sub)==len(bss)
 rows.append({'dataset':ds,'expected_batches':bss,'actual_batches':actual,'complete':ok})
st={'session':os.environ['SESSION_ID'],'family':'vit_bundle','method':os.environ['METHOD'],'checks':rows,'complete':all(x['complete'] for x in rows),'source_commit':os.environ.get('SOURCE_COMMIT')};(r/'SESSION_STATUS.json').write_text(json.dumps(st,indent=2));print(json.dumps(st,indent=2))
PYSTATUS
pack_results;trap - EXIT;echo "P03 bundle finished; attach P03_results.zip and rerun if incomplete.";exit 0

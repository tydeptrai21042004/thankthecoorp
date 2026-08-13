# DT1D WHC — fresh standalone Kaggle cells

Every training cell below contains the full clone → install → dataset/weights → plan/dry-run → train → aggregate/ZIP workflow.

## A01 — DTD / ResNet-18 ablation + hyper

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: A01 | DTD / ResNet-18 ablation + hyperparameter sensitivity | all 3 seeds in one <=12h session
# Methods/variants: 30 variants
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~7.50 h conservative; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="A01"
TARGET="whc_p2_fixed_gate_reviewer_dtd_r18"
METHODS="30 variants"
SEEDS="0,1,2"
MANIFEST_REL="configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml"
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"

# Augment the committed reviewer matrix with the old hyperparameter-sensitivity group/support controls.
RUNTIME_MANIFEST="$OUTPUT_ROOT/runtime_ablation_hyper_manifest.yaml"; export RUNTIME_MANIFEST
python - <<'PYMAN'
import copy,os,yaml
from pathlib import Path
src=Path('configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml');d=yaml.safe_load(src.read_text())
extra={
 'hyper_group8': {'method_preset':'whc_final','args':{'dt_alpha_group':8}},
 'hyper_group32': {'method_preset':'whc_final','args':{'dt_alpha_group':32}},
 'hyper_support1': {'method_preset':'whc_final','args':{'whc_active_offsets':'1'}},
 'hyper_support12': {'method_preset':'whc_final','args':{'whc_active_offsets':'1,2'}},
}
for t in d['targets'].values():
    t['variants']=copy.deepcopy(t['variants'])
    t['variants'].update(copy.deepcopy(extra))
    order=list(t.get('variant_order',[]))
    for k in extra:
        if k not in order: order.append(k)
    t['variant_order']=order
d['variant_order']=list(d.get('variant_order',[]))+[k for k in extra if k not in d.get('variant_order',[])]
Path(os.environ['RUNTIME_MANIFEST']).write_text(yaml.safe_dump(d,sort_keys=False))
print('AUGMENTED ABLATION/HYPER VARIANTS=',len(d['targets'][os.environ['TARGET']]['variants']))
PYMAN
MANIFEST_REL="$RUNTIME_MANIFEST"

# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "target" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## A02 — Flowers102 / EfficientNet-B0 ablation + hyper

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: A02 | Flowers102 / EfficientNet-B0 ablation + hyperparameter sensitivity | all 3 seeds in one <=12h session
# Methods/variants: 30 variants
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~9.00 h conservative; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="A02"
TARGET="whc_p2_fixed_gate_reviewer_flowers_b0"
METHODS="30 variants"
SEEDS="0,1,2"
MANIFEST_REL="configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml"
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"

# Augment the committed reviewer matrix with the old hyperparameter-sensitivity group/support controls.
RUNTIME_MANIFEST="$OUTPUT_ROOT/runtime_ablation_hyper_manifest.yaml"; export RUNTIME_MANIFEST
python - <<'PYMAN'
import copy,os,yaml
from pathlib import Path
src=Path('configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml');d=yaml.safe_load(src.read_text())
extra={
 'hyper_group8': {'method_preset':'whc_final','args':{'dt_alpha_group':8}},
 'hyper_group32': {'method_preset':'whc_final','args':{'dt_alpha_group':32}},
 'hyper_support1': {'method_preset':'whc_final','args':{'whc_active_offsets':'1'}},
 'hyper_support12': {'method_preset':'whc_final','args':{'whc_active_offsets':'1,2'}},
}
for t in d['targets'].values():
    t['variants']=copy.deepcopy(t['variants'])
    t['variants'].update(copy.deepcopy(extra))
    order=list(t.get('variant_order',[]))
    for k in extra:
        if k not in order: order.append(k)
    t['variant_order']=order
d['variant_order']=list(d.get('variant_order',[]))+[k for k in extra if k not in d.get('variant_order',[])]
Path(os.environ['RUNTIME_MANIFEST']).write_text(yaml.safe_dump(d,sort_keys=False))
print('AUGMENTED ABLATION/HYPER VARIANTS=',len(d['targets'][os.environ['TARGET']]['variants']))
PYMAN
MANIFEST_REL="$RUNTIME_MANIFEST"

# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "target" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## P01 — Updated proposal in existing CNN experiments

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).
# Standalone Kaggle cell: P01 | UPDATED proposal-only rerun across 5 existing CNN experiments
# Targets: table_05, table_09, figure_04, table_14_15, figure_01
# Method: dt1d (updated WHC-Compact-DT1D implementation), seeds 0,1,2
# Conservative summed estimate: ~7.40 h on 2xT4; shared clone/install/preload overhead makes actual wall time typically lower.
# Only bundled because total estimate is below 12 h. Hard cutoff: 11 h 55 min; resumable ZIP.
SESSION_ID="P01"; TARGETS="table_05,table_09,figure_04,table_14_15,figure_01"; METHODS="dt1d"; SEEDS="0,1,2"
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"; OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGETS METHODS SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
pack_results() {
  if [[ -d "$OUTPUT_ROOT" ]]; then python - <<'PYZIP'
import os,zipfile
from pathlib import Path
root=Path(os.environ['OUTPUT_ROOT']);dst=Path(os.environ['RESULT_ZIP'])
if dst.exists():dst.unlink()
with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() not in {'.pth','.pt','.ckpt'}: z.write(p,p.relative_to(root.parent))
print('RESULT_ZIP=',dst)
PYZIP
  ls -lh "$RESULT_ZIP" || true; fi
}
trap 'rc=$?; trap - EXIT; [[ -f "$RESULT_ZIP" ]] || pack_results || true; exit $rc' EXIT
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
if [[ -n "$REPO_COMMIT" ]]; then git fetch --depth 1 origin "$REPO_COMMIT"; git checkout --detach "$REPO_COMMIT"; [[ "$(git rev-parse HEAD)" == "$REPO_COMMIT" ]]; fi
SOURCE_COMMIT="$(git rev-parse HEAD)"; export SOURCE_COMMIT; echo "SOURCE_COMMIT=$SOURCE_COMMIT"
python - <<'PYSOURCE'
import hashlib, os
from pathlib import Path
expected={
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
bad=[]
for rel,want in expected.items():
    p=Path(rel); got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU
# Restore best prior P01 bundle if attached.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input');zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
 c=s=0
 try:
  with zipfile.ZipFile(z) as f:
   ns=f.namelist();s=sum(n.endswith('test_summary.json') for n in ns)
   for n in ns:
    if n.endswith('SESSION_STATUS.json'):
     try:c=max(c,int(bool(json.loads(f.read(n)).get('complete',False))))
     except:pass
 except:pass
 return (c,s,z.stat().st_mtime)
if zs:
 z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
 with zipfile.ZipFile(z) as f:f.extractall(tmp)
 found=list(tmp.rglob(f'run_{sid}'))
 if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"
# Preload all datasets/backbones once.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
root=os.environ['DATA_ROOT'];m=yaml.safe_load(Path('configs/paper/cnn_three_seed_manifest.yaml').read_text());targets=os.environ['TARGETS'].split(',')
seen_ds=set();seen_bb=set()
for t in targets:
 s=m['targets'][t];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',t,ds,bb)
 if ds not in seen_ds:
  if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
  elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
  elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
  elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
  elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
  elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
  elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
  elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
  else:raise RuntimeError(ds)
  seen_ds.add(ds)
 if bb not in seen_bb:
  if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
  elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
  elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
  else:raise RuntimeError(bb)
  seen_bb.add(bb)
print('PRELOAD PASS')
PYPRELOAD

run_target() {
 TARGET="$1"; export TARGET; echo "========== P01 TARGET $TARGET =========="
 if (( $(date +%s) >= DEADLINE_EPOCH )); then return 20; fi
 python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT" --plan-only
 PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
 python - <<'PYPLAN'
import json,os
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',')];ms=os.environ['METHODS'].split(',')
assert p['run_count']==len(ss)*len(ms),(p['run_count'],len(ss)*len(ms));assert p['seeds']==ss;print('PLAN PASS',os.environ['TARGET'],p['run_count'])
PYPLAN
 python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
 k=Path(j['output_dir']).parts[-2]
 if k in seen:continue
 seen.add(k);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+os.environ['TARGET']+'_'+k);shutil.rmtree(tmp,ignore_errors=True)
 subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True);shutil.rmtree(tmp,ignore_errors=True)
print('DRY PASS',os.environ['TARGET'],seen)
PYDRY
 set +e
 python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event();ng=torch.cuda.device_count();assert ng>=1
def killpg(p):
 if p.poll() is not None:return
 try:os.killpg(p.pid,signal.SIGTERM)
 except:pass
 try:p.wait(timeout=30)
 except:
  try:os.killpg(p.pid,signal.SIGKILL)
  except:pass
def worker(gpu):
 while not stop.is_set():
  if time.time()>=deadline:stop.set();return
  try:j=q.get_nowait()
  except queue.Empty:return
  od=Path(j['output_dir']);sm=od/'test_summary.json'
  if sm.is_file():print('[SKIP]',sm);q.task_done();continue
  shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True);exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
  cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
  with log.open('w') as lf:
   p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
   with lock:active[gpu]=p
   while p.poll() is None:
    if time.time()>=deadline:stop.set();killpg(p);break
    time.sleep(5)
   rc=p.poll()
   with lock:active.pop(gpu,None)
  if rc!=0 or not sm.is_file():errors.append({'experiment':exp,'rc':rc});stop.set()
  q.task_done()
ts=[threading.Thread(target=worker,args=(i,)) for i in range(min(2,ng))];[t.start() for t in ts];[t.join() for t in ts]
with lock:[killpg(p) for p in active.values()]
st={'target':os.environ['TARGET'],'complete':not errors and q.qsize()==0,'errors':errors,'remaining':q.qsize()};Path(os.environ['OUTPUT_ROOT'],f"target_status_{os.environ['TARGET']}.json").write_text(json.dumps(st,indent=2));print(st);sys.exit(0 if st['complete'] else 20)
PYRUN
 RC=$?; set -e
 if [[ "$RC" -eq 0 ]]; then python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_${TARGET}.txt"; fi
 return "$RC"
}
OVERALL=0
IFS=',' read -ra TARGET_ARRAY <<< "$TARGETS"
for T in "${TARGET_ARRAY[@]}"; do run_target "$T" || { OVERALL=$?; break; }; done
export OVERALL
python - <<'PYSTATUS'
import json,os
from pathlib import Path
root=Path(os.environ['OUTPUT_ROOT']);targets=os.environ['TARGETS'].split(',');rows=[]
for t in targets:
 p=root/f'target_status_{t}.json'; rows.append(json.loads(p.read_text()) if p.exists() else {'target':t,'complete':False,'missing':True})
st={'session':os.environ['SESSION_ID'],'family':'cnn_bundle','targets':targets,'methods':os.environ['METHODS'].split(','),'seeds':[0,1,2],'target_status':rows,'complete':all(r.get('complete') for r in rows),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(st,indent=2));print(json.dumps(st,indent=2))
PYSTATUS
{ echo "session=$SESSION_ID";echo "family=cnn_bundle";echo "targets=$TARGETS";echo "methods=$METHODS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results; trap - EXIT
echo "P01 bundle finished. If incomplete, attach P01_results.zip and rerun the SAME cell; completed targets/seeds are skipped."
exit 0
```

## P02 — Updated proposal in existing binary dense experiments

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).
# Standalone Kaggle cell: P02 | UPDATED proposal-only rerun across 4 existing binary dense-prediction experiments
# Targets: binary_vit_drive,binary_deeplab_drive,binary_vit_pennfudan,binary_deeplab_pennfudan
# Method whc_dt, seeds 0,1,2. Conservative summed estimate ~10.40 h on 2xT4; hard cutoff 11 h 55 min, resumable.
SESSION_ID="P02"; TARGETS="binary_vit_drive,binary_deeplab_drive,binary_vit_pennfudan,binary_deeplab_pennfudan"; METHODS="whc_dt"; SEEDS="0,1,2"
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git";REPO_COMMIT="${DT1D_CNN_COMMIT:-}";WORKDIR="/kaggle/working";REPO_DIR="$WORKDIR/dt1d-$SESSION_ID";DATA_ROOT="$WORKDIR/data_$SESSION_ID";OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID";RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)";DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))";export SESSION_ID TARGETS METHODS SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
cd "$REPO_DIR";if [[ -n "$REPO_COMMIT" ]];then git fetch --depth 1 origin "$REPO_COMMIT";git checkout --detach "$REPO_COMMIT";[[ "$(git rev-parse HEAD)" == "$REPO_COMMIT" ]];fi
SOURCE_COMMIT="$(git rev-parse HEAD)";export SOURCE_COMMIT
python - <<'PYSOURCE'
import hashlib, os
from pathlib import Path
expected={
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
bad=[]
for rel,want in expected.items():
    p=Path(rel); got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_dense_paper.py tools/run_dense_from_config.py tools/aggregate_dense_results.py dense_main.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_dense_models.py tests/test_dense_manifest.py tests/test_run_dense_paper_selection.py
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
   ns=f.namelist();s=sum(n.endswith('summary.json') for n in ns)
   for n in ns:
    if n.endswith('SESSION_STATUS.json'):
     try:c=max(c,int(bool(json.loads(f.read(n)).get('complete'))))
     except:pass
 except:pass
 return(c,s,z.stat().st_mtime)
if zs:
 z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir();zipfile.ZipFile(z).extractall(tmp);f=list(tmp.rglob(f'run_{sid}'))
 if len(f)==1:shutil.move(str(f[0]),str(out));print('RESTORED',z,score(z))
 shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"
# Preload all required official datasets and weights once.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
m=yaml.safe_load(Path('configs/dense/dense_prediction_manifest.yaml').read_text());root=Path(os.environ['DATA_ROOT']);root.mkdir(parents=True,exist_ok=True);targets=os.environ['TARGETS'].split(',');seen=set()
for t in targets:
 s=m['targets'][t];ds=s['dataset'];pipe=s['pipeline'];print('PRELOAD',t,ds,pipe)
 if ds=='drive' and 'drive' not in seen:
  dst=root/'DRIVE';req=lambda b:[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
  def valid(b): return b.is_dir() and all(p.is_dir() for p in req(b))
  found=None
  # 1) Explicit authorized/local source path (recommended for reproducibility).
  env_dir=os.environ.get('DRIVE_DATA_DIR','').strip()
  if env_dir and valid(Path(env_dir)): found=Path(env_dir)
  # 2) Attached Kaggle Input containing the historical DRIVE training/test + 1st_manual layout.
  if found is None and Path('/kaggle/input').exists():
   for tr in Path('/kaggle/input').rglob('training'):
    if valid(tr.parent): found=tr.parent; break
  # 3) Optional authorized direct archive URL supplied by the user.
  #    The official Grand Challenge distribution requires account access, so no unauthenticated mirror is hard-coded.
  if found is None:
   url=os.environ.get('DRIVE_ARCHIVE_URL','').strip()
   if url:
    arc=root/'drive_authorized_download.zip';urllib.request.urlretrieve(url,arc);tmp=root/'_drive_extract';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    shutil.unpack_archive(str(arc),str(tmp))
    for tr in tmp.rglob('training'):
     if valid(tr.parent): found=tr.parent; break
  if found is None:
   raise RuntimeError('DRIVE access is required for P02. Attach the authorized DRIVE dataset as a Kaggle Input (training/images, training/1st_manual, test/images, test/1st_manual), or set DRIVE_DATA_DIR / DRIVE_ARCHIVE_URL. This cell does not depend on any earlier cell.')
  shutil.copytree(found,dst,dirs_exist_ok=True)
  assert valid(dst), dst
  print('DRIVE READY:',dst)
  seen.add('drive')
 if ds=='pennfudan' and 'pennfudan' not in seen:
  dst=root/'PennFudanPed'
  if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
   arc=root/'PennFudanPed.zip';urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc);shutil.unpack_archive(str(arc),str(root))
  seen.add('pennfudan')
 if pipe=='vit_b16_dense' and 'vit' not in seen:models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT);seen.add('vit')
 if pipe=='deeplab_mobilenet_v3' and 'mob' not in seen:models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT);seen.add('mob')
print('PRELOAD PASS')
PYPRELOAD
run_target() {
 TARGET="$1";export TARGET;RUNTIME_MANIFEST="$OUTPUT_ROOT/runtime_${TARGET}.yaml";export RUNTIME_MANIFEST;echo "====== P02 $TARGET ======"
 python - <<'PYMAN'
import os,copy,yaml
from pathlib import Path
d=yaml.safe_load(Path('configs/dense/dense_prediction_manifest.yaml').read_text());t=os.environ['TARGET'];ms=os.environ['METHODS'].split(',');d['targets']={t:copy.deepcopy(d['targets'][t])};d['targets'][t]['methods']=ms;d['target_order']=[t];d['official_seeds']=[0,1,2];Path(os.environ['RUNTIME_MANIFEST']).write_text(yaml.safe_dump(d,sort_keys=False))
PYMAN
 python tools/run_dense_paper.py --manifest "$RUNTIME_MANIFEST" --target "$TARGET" --methods target --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cuda --download --plan-only
 PLAN="$OUTPUT_ROOT/execution_plan.json";export PLAN
 python - <<'PYPLAN'
import json,os
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());assert p['run_count']==3;assert p['seeds']==[0,1,2];print('PLAN PASS',os.environ['TARGET'])
PYPLAN
 python - <<'PYDRY'
import json,os,subprocess,sys,shutil,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);dm=yaml.safe_load(Path(os.environ['RUNTIME_MANIFEST']).read_text());sub=dm['targets'][os.environ['TARGET']].get('data_subdir','');j=p['runs'][0];tmp=Path('/kaggle/working')/('_dry_'+os.environ['TARGET']);shutil.rmtree(tmp,ignore_errors=True);subprocess.run([sys.executable,str(repo/'tools/run_dense_from_config.py'),j['config'],'--data-path',str(Path(os.environ['DATA_ROOT'])/sub),'--output-dir',str(tmp),'--device','cuda','--dry-run'],cwd=repo,check=True);shutil.rmtree(tmp,ignore_errors=True)
PYDRY
 set +e
 python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH']);dm=yaml.safe_load(Path(os.environ['RUNTIME_MANIFEST']).read_text());sub=dm['targets'][os.environ['TARGET']].get('data_subdir','');import torch
q=queue.Queue();[q.put(x) for x in p['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event();ng=torch.cuda.device_count()
def killpg(p):
 if p.poll() is not None:return
 try:os.killpg(p.pid,signal.SIGTERM)
 except:pass
 try:p.wait(timeout=30)
 except:
  try:os.killpg(p.pid,signal.SIGKILL)
  except:pass
def w(g):
 while not stop.is_set():
  if time.time()>=deadline:stop.set();return
  try:j=q.get_nowait()
  except queue.Empty:return
  od=Path(j['output_dir']);sm=od/'summary.json'
  if sm.is_file():q.task_done();continue
  shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(g);log=root/f"{j['target']}_{j['method']}_seed{j['seed']}.log";cmd=[sys.executable,str(repo/'tools/run_dense_from_config.py'),j['config'],'--data-path',str(Path(os.environ['DATA_ROOT'])/sub),'--output-dir',str(od),'--device','cuda']
  with log.open('w') as lf:
   pr=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
   with lock:active[g]=pr
   while pr.poll() is None:
    if time.time()>=deadline:stop.set();killpg(pr);break
    time.sleep(5)
   rc=pr.poll()
   with lock:active.pop(g,None)
  if rc!=0 or not sm.is_file():errors.append({'target':j['target'],'seed':j['seed'],'rc':rc});stop.set()
  q.task_done()
ts=[threading.Thread(target=w,args=(i,)) for i in range(min(2,ng))];[t.start() for t in ts];[t.join() for t in ts]
st={'target':os.environ['TARGET'],'complete':not errors and q.qsize()==0,'errors':errors,'remaining':q.qsize()};Path(root,f"target_status_{os.environ['TARGET']}.json").write_text(json.dumps(st,indent=2));print(st);sys.exit(0 if st['complete'] else 20)
PYRUN
 RC=$?;set -e;return "$RC"
}
OVERALL=0;IFS=',' read -ra TA <<< "$TARGETS";for T in "${TA[@]}";do run_target "$T" || { OVERALL=$?;break; };done
if [[ "$OVERALL" -eq 0 ]];then python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt";fi
python - <<'PYSTATUS'
import json,os
from pathlib import Path
r=Path(os.environ['OUTPUT_ROOT']);ts=os.environ['TARGETS'].split(',');rows=[]
for t in ts:
 p=r/f'target_status_{t}.json';rows.append(json.loads(p.read_text()) if p.exists() else {'target':t,'complete':False})
st={'session':os.environ['SESSION_ID'],'family':'dense_bundle','targets':ts,'methods':os.environ['METHODS'].split(','),'seeds':[0,1,2],'target_status':rows,'complete':all(x.get('complete') for x in rows),'source_commit':os.environ.get('SOURCE_COMMIT')};(r/'SESSION_STATUS.json').write_text(json.dumps(st,indent=2));print(json.dumps(st,indent=2))
PYSTATUS
pack_results;trap - EXIT;echo "P02 bundle finished; rerun with its ZIP attached if incomplete.";exit 0
```

## P03 — Updated proposal in existing ViT experiments

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).
# Standalone Kaggle cell: P03 | proposal-only ViT fair reruns: Flowers102 BS16+BS32 and VTAB-Caltech101 BS32
# Method whc_dt1d only; 10 LR candidates on tune seed 42 -> final seeds 0,1,2 -> test at best-val checkpoint.
# Conservative summed estimate ~9.0 h on 2xT4; shared setup reduces overhead. Hard cutoff 11 h 55 min; resumable.
SESSION_ID="P03";METHOD="whc_dt1d";REPO_URL="https://github.com/tydeptrai21042004/DT1D-vit.git";REPO_COMMIT="${DT1D_VIT_COMMIT:-}";WORKDIR="/kaggle/working";REPO_DIR="$WORKDIR/DT1D-vit-$SESSION_ID";DATA_ROOT="$WORKDIR/data_$SESSION_ID";MODEL_ROOT="$WORKDIR/models_$SESSION_ID";OUTPUT_ROOT="$WORKDIR/vit_$SESSION_ID";RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip";CELL_START_EPOCH="$(date +%s)";DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))";export SESSION_ID METHOD REPO_DIR DATA_ROOT MODEL_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH
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
cd "$REPO_DIR";if [[ -n "$REPO_COMMIT" ]];then git fetch --depth 1 origin "$REPO_COMMIT";git checkout --detach "$REPO_COMMIT";[[ "$(git rev-parse HEAD)" == "$REPO_COMMIT" ]];fi;SOURCE_COMMIT="$(git rev-parse HEAD)";export SOURCE_COMMIT
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
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT" "$MODEL_ROOT"
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
```

## C01 — DTD / ResNet-50 / 100ep / BS128 — all 9 methods

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C01 | DTD / ResNet-50 / 100ep / BS128 — all 9 methods | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~10.30 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C01"
TARGET="table_03"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C02 — Flowers102 / ResNet-50 / 100ep / BS128 — all 9 methods

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C02 | Flowers102 / ResNet-50 / 100ep / BS128 — all 9 methods | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~5.80 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C02"
TARGET="table_04"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C03 — Flowers102 / ResNet-18 / 100ep / BS64 — all 9 methods

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C03 | Flowers102 / ResNet-18 / 100ep / BS64 — all 9 methods | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~3.20 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C03"
TARGET="table_06"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C04 — Flowers102 / ResNet-18 / 100ep / BS32 — all 9 methods

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C04 | Flowers102 / ResNet-18 / 100ep / BS32 — all 9 methods | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~4.00 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C04"
TARGET="table_07"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C05 — Oxford-IIIT Pet / EfficientNet-B0 / 10ep / BS64 — all 9 methods

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C05 | Oxford-IIIT Pet / EfficientNet-B0 / 10ep / BS64 — all 9 methods | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~1.50 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C05"
TARGET="table_12"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C06 — Oxford-IIIT Pet / EfficientNet-B0 / 100ep / BS64 — all 9 methods

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C06 | Oxford-IIIT Pet / EfficientNet-B0 / 100ep / BS64 — all 9 methods | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~10.00 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C06"
TARGET="table_13"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C07 — EuroSAT / MobileNetV3-Small / 25ep / BS32 — all 9 methods

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C07 | EuroSAT / MobileNetV3-Small / 25ep / BS32 — all 9 methods | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~11.40 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C07"
TARGET="table_18_19"
METHODS="linear,bitfit,ssf,dt1d,bam,lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C08 — SVHN / ResNet-50 / 10ep / BS128 — split 1/3 (would exceed 12h as one session)

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C08 | SVHN / ResNet-50 / 10ep / BS128 — split 1/3 (would exceed 12h as one session) | compact <=12h session | UPDATED WHC proposal
# Methods/variants: full,linear,bitfit
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~11.80 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C08"
TARGET="table_08"
METHODS="full,linear,bitfit"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C09 — SVHN / ResNet-50 / 10ep / BS128 — split 2/3

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C09 | SVHN / ResNet-50 / 10ep / BS128 — split 2/3 | compact <=12h session | UPDATED WHC proposal
# Methods/variants: bam,lora_conv,residual
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~11.70 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C09"
TARGET="table_08"
METHODS="bam,lora_conv,residual"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C10 — SVHN / ResNet-50 / 10ep / BS128 — split 3/3

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C10 | SVHN / ResNet-50 / 10ep / BS128 — split 3/3 | compact <=12h session | UPDATED WHC proposal
# Methods/variants: ssf,dt1d,conv_r4
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~11.50 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C10"
TARGET="table_08"
METHODS="ssf,dt1d,conv_r4"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C11 — Food-101 / EfficientNet-B0 / 10ep / BS64 — split 1/2 (all methods would exceed 12h)

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C11 | Food-101 / EfficientNet-B0 / 10ep / BS64 — split 1/2 (all methods would exceed 12h) | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~11.73 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C11"
TARGET="table_11"
METHODS="linear,bitfit,ssf,dt1d,bam"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C12 — Food-101 / EfficientNet-B0 / 10ep / BS64 — split 2/2

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C12 | Food-101 / EfficientNet-B0 / 10ep / BS64 — split 2/2 | compact <=12h session | UPDATED WHC proposal
# Methods/variants: lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~9.87 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C12"
TARGET="table_11"
METHODS="lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C13 — Food-101 / ResNet-18 / 10ep / BS32 — split 1/2 (all methods would exceed 12h)

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C13 | Food-101 / ResNet-18 / 10ep / BS32 — split 1/2 (all methods would exceed 12h) | compact <=12h session | UPDATED WHC proposal
# Methods/variants: linear,bitfit,ssf,dt1d,bam
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~10.33 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C13"
TARGET="table_10"
METHODS="linear,bitfit,ssf,dt1d,bam"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## C14 — Food-101 / ResNet-18 / 10ep / BS32 — split 2/2

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: C14 | Food-101 / ResNet-18 / 10ep / BS32 — split 2/2 | compact <=12h session | UPDATED WHC proposal
# Methods/variants: lora_conv,residual,conv_r4,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~8.67 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> source/proposal validation -> install/tests -> restore -> data/weights -> plan -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="C14"
TARGET="table_10"
METHODS="lora_conv,residual,conv_r4,full"
SEEDS="0,1,2"
MANIFEST_REL=""
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"
REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS MANIFEST_REL REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

# 2) Kaggle dependencies + focused source tests; keep Kaggle CUDA torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_cnn_paper.py tools/run_from_config.py tools/aggregate_cnn_paper.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_reviewer_ablation_manifest.py tests/test_cnn_runner.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"


# 4) Download official dataset once and cache pretrained backbone before parallel workers.
python - <<'PYPRELOAD'
import os,yaml
from pathlib import Path
from torchvision import datasets,models
repo=Path(os.environ['REPO_DIR']);root=os.environ['DATA_ROOT'];target=os.environ['TARGET']
mp=os.environ.get('MANIFEST_REL','')
manifest=Path(mp) if mp else repo/'configs/paper/cnn_three_seed_manifest.yaml'
if not manifest.is_absolute(): manifest=repo/manifest
m=yaml.safe_load(manifest.read_text());s=m['targets'][target];ds=s['dataset'];bb=s['backbone'];print('PRELOAD',ds,bb)
if ds=='flowers102':[datasets.Flowers102(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='dtd':[datasets.DTD(root=root,split=x,download=True) for x in ('train','val','test')]
elif ds=='svhn':[datasets.SVHN(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='food101':[datasets.Food101(root=root,split=x,download=True) for x in ('train','test')]
elif ds=='oxford_iiit_pet':[datasets.OxfordIIITPet(root=root,split=x,download=True) for x in ('trainval','test')]
elif ds=='caltech101':datasets.Caltech101(root=root,download=True)
elif ds=='eurosat':datasets.EuroSAT(root=root,download=True)
elif ds=='fgvc_aircraft':[datasets.FGVCAircraft(root=root,split=x,download=True) for x in ('train','val','test')]
else:raise RuntimeError('No preloader for '+ds)
if bb=='resnet18':models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
elif bb=='resnet50':models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
elif bb=='efficientnet_b0':models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
elif bb=='mobilenet_v3_small':models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
else:raise RuntimeError('No backbone preloader for '+bb)
print('PRELOAD PASS')
PYPRELOAD

# 5) Generate exact execution plan.
RUNNER=(python tools/run_cnn_paper.py --target "$TARGET" --seeds "$SEEDS" --methods "$METHODS" --data-path "$DATA_ROOT" --device cuda --output-root "$OUTPUT_ROOT")
if [[ -n "$MANIFEST_REL" ]]; then RUNNER+=(--manifest "$MANIFEST_REL"); fi
"${RUNNER[@]}" --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json"; export PLAN
python - <<'PYPLAN'
import json,os,yaml
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
mp=os.environ.get('MANIFEST_REL','')
if mp:
    m=yaml.safe_load(Path(mp).read_text());expected=len(m['targets'][os.environ['TARGET']]['variants'])*len(ss)
else:
    ms=[x for x in os.environ['METHODS'].split(',') if x];expected=len(ms)*len(ss)
assert p['run_count']==expected,(p['run_count'],expected);assert p['seeds']==ss,p['seeds']
print('PLAN PASS:',p['run_count'],'runs')
PYPLAN

# 6) Dry-run one config for every unique method/variant.
python - <<'PYDRY'
import json,os,subprocess,sys,shutil
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);seen=set()
for j in plan['runs']:
    run_key=Path(j['output_dir']).parts[-2]
    if run_key in seen:continue
    seen.add(run_key);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+run_key.replace('/','_'));shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(tmp),'--data-path',os.environ['DATA_ROOT'],'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS:',len(seen),'unique method/variant configs')
PYDRY

# 7) Execute plan on up to two GPUs. Completed seeds restored from prior ZIP are skipped.
set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH'])
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'test_summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=j['experiment_id'];log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_from_config.py'),j['config'],'--output-dir',str(od),'--data-path',os.environ['DATA_ROOT'],'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'cnn','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

# 8) Aggregate when this standalone session contains all three seeds.
if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_cnn_paper.py --root "$OUTPUT_ROOT" --target "$TARGET" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=cnn";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell; completed runs are skipped.";fi
exit 0
```

## D01 — Oxford-IIIT Pet semantic / DeepLabV3-MobileNetV3 / 5ep / BS8

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: D01 | Oxford-IIIT Pet semantic / DeepLabV3-MobileNetV3 / 5ep / BS8 | all 9 methods | UPDATED WHC proposal
# Methods: linear,bitfit,ssf,whc_dt,bam,lora_conv,residual_adapter,conv_adapter,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~7.00 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> final-WHC validation -> install/tests -> restore -> dataset/weights -> runtime manifest -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="D01"; TARGET="semantic_pet_deeplab"; METHODS="linear,bitfit,ssf,whc_dt,bam,lora_conv,residual_adapter,conv_adapter,full"; SEEDS="0,1,2"
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; RUNTIME_MANIFEST="$OUTPUT_ROOT/runtime_dense_manifest.yaml"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP RUNTIME_MANIFEST DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_dense_paper.py tools/run_dense_from_config.py tools/aggregate_dense_results.py dense_main.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_dense_models.py tests/test_dense_manifest.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"

# Runtime manifest permits exactly the requested method subset while keeping every other target setting unchanged.
python - <<'PYMAN'
import os,copy,yaml
from pathlib import Path
src=Path('configs/dense/dense_prediction_manifest.yaml');d=yaml.safe_load(src.read_text());t=os.environ['TARGET'];methods=[x for x in os.environ['METHODS'].split(',') if x]
assert t in d['targets'];unknown=[m for m in methods if m not in d['methods']];assert not unknown,unknown
d['targets']={t:copy.deepcopy(d['targets'][t])};d['targets'][t]['methods']=methods;d['target_order']=[t];d['official_seeds']=[int(x) for x in os.environ['SEEDS'].split(',') if x]
Path(os.environ['RUNTIME_MANIFEST']).write_text(yaml.safe_dump(d,sort_keys=False));print('RUNTIME DENSE METHODS=',methods)
PYMAN

# Official dataset + pretrained-backbone pre-cache.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
m=yaml.safe_load(Path(os.environ['RUNTIME_MANIFEST']).read_text());s=m['targets'][os.environ['TARGET']];root=Path(os.environ['DATA_ROOT']);root.mkdir(parents=True,exist_ok=True);ds=s['dataset'];pipe=s['pipeline'];print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE';req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input');found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent;r=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in r):found=b;break
        if found is None:raise RuntimeError('DRIVE requires an authorized Kaggle Input or source path. This target normally does not use DRIVE.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip';urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc);shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else:raise RuntimeError(ds)
if pipe=='vit_b16_dense':models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}:models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --manifest "$RUNTIME_MANIFEST" --target "$TARGET" --methods target --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cuda --download --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json";export PLAN
python - <<'PYPLAN'
import json,os
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
assert p['run_count']==len(ms)*len(ss),(p['run_count'],ms,ss);assert p['seeds']==ss;print('PLAN PASS',p['run_count'])
PYPLAN

python - <<'PYDRY'
import json,os,subprocess,sys,shutil,yaml
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);dm=yaml.safe_load(Path(os.environ['RUNTIME_MANIFEST']).read_text());sub=dm['targets'][os.environ['TARGET']].get('data_subdir','');seen=set()
for j in plan['runs']:
    m=j['method']
    if m in seen:continue
    seen.add(m);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+m);shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_dense_from_config.py'),j['config'],'--data-path',str(Path(os.environ['DATA_ROOT'])/sub),'--output-dir',str(tmp),'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS',sorted(seen))
PYDRY

set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time,yaml
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH']);dm=yaml.safe_load(Path(os.environ['RUNTIME_MANIFEST']).read_text());sub=dm['targets'][os.environ['TARGET']].get('data_subdir','')
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=f"{j['target']}_{j['method']}_seed{j['seed']}";log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_dense_from_config.py'),j['config'],'--data-path',str(Path(os.environ['DATA_ROOT'])/sub),'--output-dir',str(od),'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'dense','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=dense";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell.";fi
exit 0
```

## D02 — Oxford-IIIT Pet detection / Faster R-CNN MobileNetV3-FPN / 1ep / BS1

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: D02 | Oxford-IIIT Pet detection / Faster R-CNN MobileNetV3-FPN / 1ep / BS1 | all 9 methods | UPDATED WHC proposal
# Methods: linear,bitfit,ssf,whc_dt,bam,lora_conv,residual_adapter,conv_adapter,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~7.50 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> final-WHC validation -> install/tests -> restore -> dataset/weights -> runtime manifest -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="D02"; TARGET="detection_pet_fasterrcnn"; METHODS="linear,bitfit,ssf,whc_dt,bam,lora_conv,residual_adapter,conv_adapter,full"; SEEDS="0,1,2"
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/data_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/run_$SESSION_ID"; RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"; RUNTIME_MANIFEST="$OUTPUT_ROOT/runtime_dense_manifest.yaml"
CELL_START_EPOCH="$(date +%s)"; DEADLINE_EPOCH="$((CELL_START_EPOCH + 715*60))"
export SESSION_ID TARGET METHODS SEEDS REPO_DIR DATA_ROOT OUTPUT_ROOT RESULT_ZIP RUNTIME_MANIFEST DEADLINE_EPOCH

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

# 1) Fresh clone of the UPDATED repository. Set DT1D_CNN_COMMIT=<sha> to pin an exact Git commit.
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
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
python -m py_compile tools/run_dense_paper.py tools/run_dense_from_config.py tools/aggregate_dense_results.py dense_main.py
python -m pytest -q tests/test_whc_compact_dt1d_adapter.py tests/test_dense_models.py tests/test_dense_manifest.py tests/test_run_dense_paper_selection.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(),'Enable Kaggle GPU accelerator.'
print('GPU count=',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
PYGPU

# 3) Restore the best previous partial/completed ZIP if attached as Kaggle Input.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json,os,shutil,zipfile
from pathlib import Path
sid=os.environ['SESSION_ID'];out=Path(os.environ['OUTPUT_ROOT']);inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in names)
            sts=[n for n in names if n.endswith('SESSION_STATUS.json')]
            for n in sts:
                try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                except Exception: pass
    except Exception: pass
    return (complete,summaries,z.stat().st_mtime)
if zs:
    z=max(zs,key=score);tmp=Path('/kaggle/working')/f'_restore_{sid}';shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir()
    with zipfile.ZipFile(z) as f:f.extractall(tmp)
    found=list(tmp.rglob(f'run_{sid}'))
    if len(found)==1:shutil.move(str(found[0]),str(out));print('RESTORED',z,'score=',score(z))
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"
echo "DATASET BOOTSTRAP ROOT=$DATA_ROOT"

# Runtime manifest permits exactly the requested method subset while keeping every other target setting unchanged.
python - <<'PYMAN'
import os,copy,yaml
from pathlib import Path
src=Path('configs/dense/dense_prediction_manifest.yaml');d=yaml.safe_load(src.read_text());t=os.environ['TARGET'];methods=[x for x in os.environ['METHODS'].split(',') if x]
assert t in d['targets'];unknown=[m for m in methods if m not in d['methods']];assert not unknown,unknown
d['targets']={t:copy.deepcopy(d['targets'][t])};d['targets'][t]['methods']=methods;d['target_order']=[t];d['official_seeds']=[int(x) for x in os.environ['SEEDS'].split(',') if x]
Path(os.environ['RUNTIME_MANIFEST']).write_text(yaml.safe_dump(d,sort_keys=False));print('RUNTIME DENSE METHODS=',methods)
PYMAN

# Official dataset + pretrained-backbone pre-cache.
python - <<'PYPRELOAD'
import os,shutil,urllib.request,yaml
from pathlib import Path
from torchvision import datasets,models
m=yaml.safe_load(Path(os.environ['RUNTIME_MANIFEST']).read_text());s=m['targets'][os.environ['TARGET']];root=Path(os.environ['DATA_ROOT']);root.mkdir(parents=True,exist_ok=True);ds=s['dataset'];pipe=s['pipeline'];print('PRELOAD',ds,pipe)
if ds=='drive':
    dst=root/'DRIVE';req=[dst/'training/images',dst/'training/1st_manual',dst/'test/images',dst/'test/1st_manual']
    if not all(p.is_dir() for p in req):
        inp=Path('/kaggle/input');found=None
        if inp.exists():
            for tr in inp.rglob('training'):
                b=tr.parent;r=[b/'training/images',b/'training/1st_manual',b/'test/images',b/'test/1st_manual']
                if all(p.is_dir() for p in r):found=b;break
        if found is None:raise RuntimeError('DRIVE requires an authorized Kaggle Input or source path. This target normally does not use DRIVE.')
        shutil.copytree(found,dst,dirs_exist_ok=True)
elif ds=='pennfudan':
    dst=root/'PennFudanPed'
    if not ((dst/'PNGImages').is_dir() and (dst/'PedMasks').is_dir()):
        arc=root/'PennFudanPed.zip';urllib.request.urlretrieve('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',arc);shutil.unpack_archive(str(arc),str(root))
elif ds in {'oxford_pet_segmentation','oxford_pet_detection'}:
    [datasets.OxfordIIITPet(root=str(root),split=x,target_types='segmentation',download=True) for x in ('trainval','test')]
else:raise RuntimeError(ds)
if pipe=='vit_b16_dense':models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
elif pipe in {'deeplab_mobilenet_v3','fasterrcnn_mobilenet_v3_fpn'}:models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
print('PRELOAD PASS')
PYPRELOAD

python tools/run_dense_paper.py --manifest "$RUNTIME_MANIFEST" --target "$TARGET" --methods target --seeds "$SEEDS" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --device cuda --download --plan-only
PLAN="$OUTPUT_ROOT/execution_plan.json";export PLAN
python - <<'PYPLAN'
import json,os
from pathlib import Path
p=json.loads(Path(os.environ['PLAN']).read_text());ms=[x for x in os.environ['METHODS'].split(',') if x];ss=[int(x) for x in os.environ['SEEDS'].split(',') if x]
assert p['run_count']==len(ms)*len(ss),(p['run_count'],ms,ss);assert p['seeds']==ss;print('PLAN PASS',p['run_count'])
PYPLAN

python - <<'PYDRY'
import json,os,subprocess,sys,shutil,yaml
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);dm=yaml.safe_load(Path(os.environ['RUNTIME_MANIFEST']).read_text());sub=dm['targets'][os.environ['TARGET']].get('data_subdir','');seen=set()
for j in plan['runs']:
    m=j['method']
    if m in seen:continue
    seen.add(m);tmp=Path('/kaggle/working')/('_dry_'+os.environ['SESSION_ID']+'_'+m);shutil.rmtree(tmp,ignore_errors=True)
    subprocess.run([sys.executable,str(repo/'tools/run_dense_from_config.py'),j['config'],'--data-path',str(Path(os.environ['DATA_ROOT'])/sub),'--output-dir',str(tmp),'--device','cuda','--dry-run'],cwd=repo,check=True)
    shutil.rmtree(tmp,ignore_errors=True)
print('DRY-RUN PASS',sorted(seen))
PYDRY

set +e
python - <<'PYRUN'
import json,os,queue,shutil,signal,subprocess,sys,threading,time,yaml
from pathlib import Path
plan=json.loads(Path(os.environ['PLAN']).read_text());repo=Path(os.environ['REPO_DIR']);root=Path(os.environ['OUTPUT_ROOT']);deadline=float(os.environ['DEADLINE_EPOCH']);dm=yaml.safe_load(Path(os.environ['RUNTIME_MANIFEST']).read_text());sub=dm['targets'][os.environ['TARGET']].get('data_subdir','')
import torch
ng=torch.cuda.device_count();assert ng>=1
q=queue.Queue();[q.put(x) for x in plan['runs']];errors=[];active={};lock=threading.Lock();stop=threading.Event()
def killpg(p):
    if p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except Exception:pass
    try:p.wait(timeout=30)
    except Exception:
        try:os.killpg(p.pid,signal.SIGKILL)
        except Exception:pass
def worker(gpu):
    while not stop.is_set():
        if time.time()>=deadline:stop.set();return
        try:j=q.get_nowait()
        except queue.Empty:return
        od=Path(j['output_dir']);summary=od/'summary.json'
        if summary.is_file():print('[SKIP COMPLETE]',summary,flush=True);q.task_done();continue
        shutil.rmtree(od,ignore_errors=True);od.mkdir(parents=True,exist_ok=True)
        exp=f"{j['target']}_{j['method']}_seed{j['seed']}";log=root/(exp+'.log');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,str(repo/'tools/run_dense_from_config.py'),j['config'],'--data-path',str(Path(os.environ['DATA_ROOT'])/sub),'--output-dir',str(od),'--device','cuda']
        with log.open('w',encoding='utf-8') as lf:
            p=subprocess.Popen(cmd,cwd=repo,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=p
            while p.poll() is None:
                if time.time()>=deadline:stop.set();killpg(p);break
                time.sleep(5)
            rc=p.poll()
            with lock:active.pop(gpu,None)
        if rc!=0 or not summary.is_file():errors.append({'experiment':exp,'rc':rc,'log':str(log)});stop.set() if time.time()<deadline else None
        else:print('[DONE]',exp,flush=True)
        q.task_done()
threads=[threading.Thread(target=worker,args=(i,),daemon=False) for i in range(min(2,ng))];[t.start() for t in threads];[t.join() for t in threads]
with lock:[killpg(p) for p in list(active.values())]
status={'session':os.environ['SESSION_ID'],'family':'dense','target':os.environ['TARGET'],'methods':os.environ['METHODS'].split(','),'seeds':[int(x) for x in os.environ['SEEDS'].split(',') if x],'complete':(not errors and q.qsize()==0),'errors':errors,'remaining_jobs':q.qsize(),'source_commit':os.environ.get('SOURCE_COMMIT')}
(root/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2));sys.exit(0 if status['complete'] else 20)
PYRUN
TRAIN_RC=$?
set -e

if [[ "$SEEDS" == "0,1,2" && "$TRAIN_RC" -eq 0 ]]; then
  python tools/aggregate_dense_results.py --input-root "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/aggregated" --require-seeds "$SEEDS" | tee "$OUTPUT_ROOT/aggregation_stdout.txt"
fi
{ echo "session=$SESSION_ID";echo "family=dense";echo "target=$TARGET";echo "methods=$METHODS";echo "seeds=$SEEDS";echo "commit=$SOURCE_COMMIT";nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results;trap - EXIT
if [[ "$TRAIN_RC" -eq 0 ]]; then echo "SESSION COMPLETE: $SESSION_ID";else echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun THIS SAME cell.";fi
exit 0
```

## V01 — VTAB-DTD / ViT-B/16 / BS32

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# FRESH-STANDALONE GUARANTEE: this file can be pasted into a new Kaggle GPU session by itself.
# It clones the current GitHub repository, validates the uploaded-code snapshot, installs dependencies,
# prepares/downloads its own dataset + pretrained weights, generates/dry-runs the plan, trains, aggregates, and zips results.
# No repository, dataset cache, model cache, or output from another training cell is required (DRIVE access exception documented below).

# Standalone Kaggle cell: V01 | VTAB-DTD / ViT-B/16 / BS32 — all 5 methods | all methods in one <=12h session
# Methods: whc_dt1d,vpt,pfeiffer,full,linear
# Fair protocol: 10 LR candidates/method on tune seed 42 -> final seeds 0/1/2 -> test once at best-validation checkpoint.
# Estimated 2xT4 wall time: ~11.50 h conservative (shared setup should reduce this); hard cutoff is 11 h 55 min with resumable ZIP.

SESSION_ID="V01"; DATASET="vtab-dtd"; BATCH_SIZE="32"; METHOD="whc_dt1d,vpt,pfeiffer,full,linear"; VPT_TOKENS="10"
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
```

## V02 — VTAB-EuroSAT / ViT-B/16 / BS32

```bash
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
```

## FINAL MERGE

Aggregation only; attach the result ZIPs first.

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
# Attach result ZIPs as Kaggle Inputs. Default STRICT_MERGE=1 requires all 23 compact sessions complete.
WORK=/kaggle/working;STAGE="$WORK/dt1d_merge_stage";CNNROOT="$WORK/combined_cnn";DENSEROOT="$WORK/combined_dense";OUT="$WORK/DT1D_WHC_FINAL_MERGED_RESULTS"
STRICT_MERGE="${STRICT_MERGE:-1}";rm -rf "$STAGE" "$CNNROOT" "$DENSEROOT" "$OUT";mkdir -p "$STAGE" "$CNNROOT" "$DENSEROOT" "$OUT/cnn" "$OUT/vit"
REPO="$WORK/dt1d-merge";rm -rf "$REPO";git clone --depth 1 "https://github.com/tydeptrai21042004/dt1d.git" "$REPO";cd "$REPO"
if [[ -n "${DT1D_CNN_COMMIT:-}" ]];then git fetch --depth 1 origin "$DT1D_CNN_COMMIT";git checkout --detach "$DT1D_CNN_COMMIT";fi
SOURCE_COMMIT="$(git rev-parse HEAD)";export SOURCE_COMMIT
python - <<'PYSOURCE'
import hashlib, os
from pathlib import Path
expected={
    "configs/ablations/whc_p2_fixed_gate_reviewer_ablation.yaml": "498e3b8cced23f3b4fffda4d9ca99b6b9e7180087cbdb2aabf034bd019af9819",
    "configs/dense/dense_prediction_manifest.yaml": "61f09777d631c844f5389412a84cb1cd84613475363c24711e3e7b3eff46f078",
    "configs/paper/cnn_three_seed_manifest.yaml": "87bb4464e1d324c63de15d108a7321dd16706382bcea457a719c2c025747bec9",
    "models/whc_compact_dt1d_adapter.py": "db6f55928a0d913b1653d3370bf38a88408af65425031e94915923c16caa76db",
    "tools/run_cnn_paper.py": "07488718f2646be54787ff93160c16bee19c0e565737df78e46e417919006b2b",
    "tools/run_dense_paper.py": "4f576833b022ce3a9774082330fd9613af105d0e46064cf28f75f9817da060bb"
}
root=Path('.')
bad=[]
for rel,want in expected.items():
    p=root/rel
    got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'
    if got != want: bad.append((rel,want,got))
if bad:
    print('SOURCE SNAPSHOT MISMATCH:')
    for row in bad: print(row)
    if os.environ.get('DT1D_ALLOW_SOURCE_MISMATCH','0') != '1':
        raise SystemExit('GitHub dt1d source does not match the uploaded final WHC snapshot. Push the uploaded source or set DT1D_ALLOW_SOURCE_MISMATCH=1 intentionally.')
print('CNN SOURCE SNAPSHOT PASS')
PYSOURCE

python -m pip install -q --upgrade-strategy only-if-needed -r requirements-kaggle.txt
python tools/validate_whc_compact.py
export EXPECTED_SESSIONS='A01,A02,P01,P02,P03,C01,C02,C03,C04,C05,C06,C07,C08,C09,C10,C11,C12,C13,C14,D01,D02,V01,V02'
python - <<'PYMERGE'
from pathlib import Path
import json,shutil,zipfile,os,pandas as pd
stage=Path('/kaggle/working/dt1d_merge_stage');cnn=Path('/kaggle/working/combined_cnn');dense=Path('/kaggle/working/combined_dense');out=Path('/kaggle/working/DT1D_WHC_FINAL_MERGED_RESULTS');inp=Path('/kaggle/input')
expected=set(os.environ['EXPECTED_SESSIONS'].split(','));strict=os.environ.get('STRICT_MERGE','1')=='1'
allz=list(inp.rglob('*_results.zip')) if inp.exists() else []
by={}
for z in allz:
 sid=z.name.split('_results.zip')[0]
 if sid not in expected: continue
 def score(z):
  complete=0;summaries=0
  try:
   with zipfile.ZipFile(z) as f:
    ns=f.namelist();summaries=sum(n.endswith(('test_summary.json','summary.json','run_summary.json')) or n.endswith('_fair_three_seed.csv') for n in ns)
    for n in ns:
     if n.endswith('SESSION_STATUS.json'):
      try:complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
      except Exception:pass
  except Exception:pass
  return (complete,summaries,z.stat().st_mtime)
 if sid not in by or score(z)>score(by[sid]):by[sid]=z
statuses=[];vit=[]
for sid,z in sorted(by.items()):
 d=stage/sid;d.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(z) as f:f.extractall(d)
 sts=list(d.rglob('SESSION_STATUS.json'))
 if len(sts)!=1: statuses.append({'session':sid,'complete':False,'zip':str(z),'error':f'status_count={len(sts)}'})
 else:
  st=json.loads(sts[0].read_text());st['zip']=str(z);statuses.append(st)
 for sm in d.rglob('test_summary.json'):
  rr=next((p for p in sm.parents if p.name==f'run_{sid}'),None)
  if rr:
   rel=sm.parent.relative_to(rr);dest=cnn/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(sm.parent,dest,dirs_exist_ok=True)
 for sm in d.rglob('summary.json'):
  rr=next((p for p in sm.parents if p.name==f'run_{sid}'),None)
  if rr:
   rel=sm.parent.relative_to(rr);dest=dense/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(sm.parent,dest,dirs_exist_ok=True)
 for cp in d.rglob('*fair_three_seed.csv'):
  try:
   df=pd.read_csv(cp);df['source_session']=sid;df['source_zip']=z.name;df['source_aggregate']=cp.name;vit.append(df)
  except Exception as e:print('VIT CSV READ WARN',cp,e)
sdf=pd.DataFrame(statuses);sdf.to_csv(out/'SESSION_STATUS_ALL.csv',index=False)
seen=set(sdf['session'].astype(str)) if not sdf.empty else set();complete=set(sdf.loc[sdf.get('complete',False).fillna(False).astype(bool),'session'].astype(str)) if not sdf.empty and 'complete' in sdf else set()
missing=sorted(expected-seen);incomplete=sorted(expected-complete)
audit={'expected_session_count':len(expected),'sessions_seen':len(seen),'sessions_complete':len(complete),'missing_sessions':missing,'incomplete_sessions':incomplete,'strict':strict}
(out/'PRE_AGGREGATION_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if strict and (missing or incomplete):raise SystemExit('Strict merge refused: missing/incomplete sessions. Set STRICT_MERGE=0 only for a deliberate partial debug merge.')
if vit:
 v=pd.concat(vit,ignore_index=True)
 keys=[k for k in ['batch_size','method_key'] if k in v.columns]
 if 'source_session' in v.columns and keys:
  # dataset is not a CSV column in current runner, infer from source sessions only by preserving all rows; identical key across different dataset files are distinguished by source file name below.
  pass
 v.to_csv(out/'VIT_ALL_3SEED_RAW.csv',index=False)
 # Infer dataset from each aggregate filename; this supports bundled ViT sessions.
 def infer_dataset(name):
  n=str(name)
  if n.startswith('flowers102_'): return 'flowers102'
  if n.startswith('vtab_caltech101_'): return 'vtab-caltech101'
  if n.startswith('vtab_dtd_'): return 'vtab-dtd'
  if n.startswith('vtab_eurosat_'): return 'vtab-eurosat'
  return 'unknown'
 v['dataset']=[infer_dataset(x) for x in v['source_aggregate']]
 dkeys=['dataset','batch_size','method_key']
 dup=v.duplicated(dkeys,keep=False)
 if dup.any():
  # Multiple independent sources for the same final key should not occur in this package.
  raise SystemExit('Duplicate ViT aggregate keys found:\n'+v.loc[dup,dkeys+['source_session']].to_string(index=False))
 v.sort_values(dkeys).to_csv(out/'VIT_ALL_3SEED.csv',index=False)
PYMERGE

# Strictly aggregate every reconstructed CNN/ablation target with three seeds.
if find "$CNNROOT" -name test_summary.json -print -quit | grep -q .;then
  mapfile -t TARGETS < <(find "$CNNROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f
' | sort)
  for TARGET in "${TARGETS[@]}";do
    python tools/aggregate_cnn_paper.py --root "$CNNROOT" --target "$TARGET" --output-dir "$OUT/cnn/$TARGET" --require-seeds 0,1,2
  done
fi
if find "$DENSEROOT" -name summary.json -print -quit | grep -q .;then
  python tools/aggregate_dense_results.py --input-root "$DENSEROOT" --output-dir "$OUT/dense_aggregated" --require-seeds 0,1,2
fi
python - <<'PYAUDIT'
from pathlib import Path
import json,pandas as pd,os
out=Path('/kaggle/working/DT1D_WHC_FINAL_MERGED_RESULTS');pre=json.loads((out/'PRE_AGGREGATION_AUDIT.json').read_text());pre['cnn_targets_aggregated']=len(list((out/'cnn').glob('*/aggregation_summary.json')));pre['vit_rows']=len(pd.read_csv(out/'VIT_ALL_3SEED.csv')) if (out/'VIT_ALL_3SEED.csv').exists() else 0;pre['dense_aggregate_present']=(out/'dense_aggregated').is_dir();pre['source_commit']=os.environ.get('SOURCE_COMMIT');(out/'AUDIT.json').write_text(json.dumps(pre,indent=2));print(json.dumps(pre,indent=2))
PYAUDIT
cd "$WORK";rm -f DT1D_WHC_FINAL_MERGED_RESULTS.zip;zip -qr DT1D_WHC_FINAL_MERGED_RESULTS.zip DT1D_WHC_FINAL_MERGED_RESULTS;ls -lh DT1D_WHC_FINAL_MERGED_RESULTS.zip
```

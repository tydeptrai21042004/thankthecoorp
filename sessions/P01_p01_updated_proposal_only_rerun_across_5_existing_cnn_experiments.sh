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

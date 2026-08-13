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

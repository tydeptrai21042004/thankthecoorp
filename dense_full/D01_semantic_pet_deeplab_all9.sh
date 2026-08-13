#!/usr/bin/env bash
set -Eeuo pipefail

# Standalone Kaggle cell: D01 | Oxford-IIIT Pet semantic / DeepLabV3-MobileNetV3 / 5ep / BS8 | all 9 methods | UPDATED WHC proposal
# Methods: linear,bitfit,ssf,whc_dt,bam,lora_conv,residual_adapter,conv_adapter,full
# Seeds: 0,1,2
# Estimated 2xT4 wall time: ~7.00 h; hard cutoff is 11 h 55 min with resumable ZIP.
# SELF-CONTAINED: clone -> final-WHC validation -> install/tests -> restore -> dataset/weights -> runtime manifest -> dry-run -> 2-GPU run -> aggregate/zip.

SESSION_ID="D01"; TARGET="semantic_pet_deeplab"; METHODS="linear,bitfit,ssf,whc_dt,bam,lora_conv,residual_adapter,conv_adapter,full"; SEEDS="0,1,2"
REPO_URL="https://github.com/tydeptrai21042004/dt1d.git"; REPO_COMMIT="${DT1D_CNN_COMMIT:-}"
WORKDIR="/kaggle/working"; REPO_DIR="$WORKDIR/dt1d-$SESSION_ID"; DATA_ROOT="$WORKDIR/dt1d_shared_data"
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
rm -rf "$REPO_DIR"
git clone --depth 1 "$REPO_URL" "$REPO_DIR"
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
        if found is None:raise RuntimeError('Official DRIVE not found. Add DRIVE as Kaggle Input with training/images, training/1st_manual, test/images, test/1st_manual.')
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

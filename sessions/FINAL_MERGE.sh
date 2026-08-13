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

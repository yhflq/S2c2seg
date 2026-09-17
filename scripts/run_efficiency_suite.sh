#!/usr/bin/env bash
# Efficiency suite: vocabulary-scaling sweep + FLOPs + end-to-end baseline comparison
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUT=results/efficiency
mkdir -p "$OUT/bench" "$OUT/logs"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- Experiment 1: vocabulary-scaling cost sweep (two-stage reference schedule, CSS vs. no-CSS) ----------
VOCABS=(
  "city|configs/cls_city_scapes.txt"
  "voc20|configs/cls_voc20.txt"
  "voc21|configs/cls_voc21.txt"
  "ctx59|configs/cls_context59.txt"
  "ctx60|configs/cls_context60.txt"
  "cocoobj|configs/cls_coco_object.txt"
  "ade150|configs/cls_ade20k.txt"
  "stuff171|configs/cls_coco_stuff.txt"
  "union303|configs/cls_union.txt"
)

run_sweep() {
  local csv=$OUT/vocab_scaling.csv
  rm -f "$csv"
  local entry tag vocab mode
  for entry in "${VOCABS[@]}"; do
    IFS='|' read -r tag vocab <<<"$entry"
    for mode in twostage nocss; do
      log "sweep ${tag}/${mode}"
      "$PY" scripts/bench_efficiency.py --vocab "$vocab" --mode "$mode" \
        --tag "$tag" --csv "$csv" --out "$OUT/bench/${tag}_${mode}.md" \
        > "$OUT/logs/sweep_${tag}_${mode}.log" 2>&1 \
        || log "warning: ${tag}/${mode} failed, see the log"
    done
  done
}

# ---------- Experiment 2: FLOPs ----------
run_flops() {
  log "FLOPs accounting"
  "$PY" scripts/bench_flops.py --out "$OUT/flops.md" --csv "$OUT/flops_units.csv" \
    > "$OUT/logs/flops.log" 2>&1 || log "warning: FLOPs failed, see the log"
}

# ---------- Experiment 3: end-to-end evaluation ----------
# These options match scripts/run_journal.sh for the corresponding protocols.
opts_for() {
  case "$1" in
    *_baseline) echo "model.x_options.X_PAPER_PIPELINE=false model.x_options.X_PAPER_T_MAX=0 model.x_options.X_PAPER_GLOBAL_SCORE=patch_mean model.x_options.X_PAPER_RESIDUAL_BETA=0 model.x_options.X_PAPER_ADAPTIVE_ADMIT=false" ;;
    voc21_r1)   echo "model.x_options.X_ADAPTIVE_BG=otsu model.x_options.X_ADAPTIVE_BG_C=0.5 model.x_options.X_SPATIAL_REFINE=3 model.x_options.X_SPATIAL_REFINE_TEMP=100" ;;
    ctx59_r1)   echo "model.x_options.X_SPATIAL_REFINE=3 model.x_options.X_SPATIAL_REFINE_TEMP=100" ;;
    stuff_r1)   echo "model.x_options.X_PAPER_LAM=0.12 model.x_options.X_SPATIAL_REFINE=3 model.x_options.X_SPATIAL_REFINE_TEMP=100" ;;
  esac
}

e2e() {
  local name=$1 cfg=$2 opts=$3
  local logf=$OUT/logs/e2e_${name}.log
  local t0 t1
  log "end-to-end ${name} start (opts: ${opts:-none})"
  t0=$(date +%s)
  if [ -n "$opts" ]; then
    timeout 6h "$PY" eval.py --config "$cfg" --cfg-options $opts > "$logf" 2>&1
  else
    timeout 6h "$PY" eval.py --config "$cfg" > "$logf" 2>&1
  fi
  local rc=$?
  t1=$(date +%s)
  "$PY" scripts/analyze_efficiency.py --record "$name" "$cfg" "$((t1 - t0))" "$rc" --log "$logf"
  if [ "$rc" -ne 0 ]; then
    log "warning: ${name} failed; its row records the exit code"
  fi
  log "end-to-end ${name} done (rc=${rc}, $(( (t1 - t0) / 60 )) min)"
}

run_e2e() {
  echo "name,config,wall_s,rc,images,ms_per_img,mIoU,peak_alloc_mb" > "$OUT/e2e_runs.csv"
  e2e voc21_baseline configs/cfg_voc21.py "$(opts_for voc21_baseline)"
  e2e voc21_r1 configs/cfg_voc21.py "$(opts_for voc21_r1)"
  e2e ctx59_baseline configs/cfg_context59.py "$(opts_for ctx59_baseline)"
  e2e ctx59_r1 configs/cfg_context59.py "$(opts_for ctx59_r1)"
  e2e stuff_baseline configs/cfg_coco_stuff164k.py "$(opts_for stuff_baseline)"
  e2e stuff_r1 configs/cfg_coco_stuff164k.py "$(opts_for stuff_r1)"
}

# STAGES=sweep,flops,e2e (default all); e.g. STAGES=e2e re-measures only the end-to-end runs.
STAGES=${STAGES:-sweep,flops,e2e}
log "===== efficiency suite start (stages: ${STAGES}) ====="
case ",${STAGES}," in *,sweep,*) run_sweep ;; esac
case ",${STAGES}," in *,flops,*) run_flops ;; esac
case ",${STAGES}," in *,e2e,*) run_e2e ;; esac
log "===== data collection finished, generating the report ====="
"$PY" scripts/analyze_efficiency.py > "$OUT/logs/analyze.log" 2>&1 \
  || log "warning: analysis script failed, see $OUT/logs/analyze.log"
log "===== all done ====="

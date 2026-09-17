#!/usr/bin/env bash

set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PROTOCOLS=(
  "voc20|configs/cfg_voc20.py"
  "voc21|configs/cfg_voc21.py"
  "city|configs/cfg_city_scapes.py"
  "ade150|configs/cfg_ade20k.py"
  "context59|configs/cfg_context59.py"
  "context60|configs/cfg_context60.py"
  "coco_object|configs/cfg_coco_object.py"
  "coco_stuff|configs/cfg_coco_stuff164k.py"
)

run_all() {
  mkdir -p results/logs
  local summary=results/${VERSION}_summary.md
  local csv=results/${VERSION}.csv
  {
    echo "# ${VERSION}: results on the eight protocols"
    echo
    echo "- start: $(date '+%F %T')"
    echo
    echo "| protocol | mIoU | aAcc | mAcc | minutes |"
    echo "|---|---:|---:|---:|---:|"
  } > "$summary"
  echo "name,miou,aacc,macc,minutes" > "$csv"

  local mious=() failed=0
  local entry name cfg opts log t0 mins line miou aacc macc
  for entry in "${PROTOCOLS[@]}"; do
    IFS='|' read -r name cfg <<<"$entry"
    log=results/logs/${VERSION}_${name}.log
    opts=$(opts_for "$name")
    t0=$(date +%s)
    echo "[$(date '+%F %T')] ${VERSION}/${name} start (opts: ${opts:-none})"
    if [ -n "$opts" ]; then
      timeout 6h "$PY" eval.py --config "$cfg" --cfg-options $opts > "$log" 2>&1
    else
      timeout 6h "$PY" eval.py --config "$cfg" > "$log" 2>&1
    fi
    mins=$(( ($(date +%s) - t0) / 60 ))
    line=$(grep -h '^RESULT ' "$log" | tail -1)
    if [ -z "$line" ]; then
      echo "[$(date '+%F %T')] ${VERSION}/${name} failed (no RESULT line, see $log)"
      echo "| ${name} | FAILED | - | - | ${mins} |" >> "$summary"
      echo "${name},FAILED,,,${mins}" >> "$csv"
      failed=1
      continue
    fi
    miou=$(sed -n 's/.* mIoU=\([0-9.]*\).*/\1/p' <<<"$line")
    aacc=$(sed -n 's/.* aAcc=\([0-9.]*\).*/\1/p' <<<"$line")
    macc=$(sed -n 's/.* mAcc=\([0-9.]*\).*/\1/p' <<<"$line")
    mious+=("$miou")
    echo "[$(date '+%F %T')] ${VERSION}/${name} done: mIoU=${miou} aAcc=${aacc} (${mins} min)"
    echo "| ${name} | ${miou} | ${aacc} | ${macc} | ${mins} |" >> "$summary"
    echo "${name},${miou},${aacc},${macc},${mins}" >> "$csv"
  done

  if [ "$failed" -eq 0 ] && [ "${#mious[@]}" -eq 8 ]; then
    local mean
    mean=$(printf '%s\n' "${mious[@]}" | awk '{s+=$1} END{printf "%.2f", s/NR}')
    {
      echo "| **mean** | **${mean}** | | | |"
      echo
      echo "- end: $(date '+%F %T')"
    } >> "$summary"
    echo "mean,${mean},,," >> "$csv"
    echo "[$(date '+%F %T')] ${VERSION} finished, eight-protocol mean mIoU = ${mean}"
  else
    {
      echo
      echo "**A protocol failed; the mean was not computed.** end: $(date '+%F %T')"
    } >> "$summary"
    echo "[$(date '+%F %T')] ${VERSION} finished with failures; the mean was not computed"
  fi
}

VERSION=conference

opts_for() {
  # Single pass; keep the conference settings explicit when journal defaults differ.
  echo "model.x_options.X_PAPER_T_MAX=0 model.x_options.X_PAPER_GLOBAL_SCORE=patch_mean model.x_options.X_PAPER_RESIDUAL_BETA=0 model.x_options.X_PAPER_ADAPTIVE_ADMIT=false model.x_options.X_SPATIAL_REFINE=0 model.x_options.X_SPATIAL_REFINE_TEMP=None model.x_options.X_ADAPTIVE_BG=none"
}

run_all

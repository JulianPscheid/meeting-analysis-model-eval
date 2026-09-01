#!/usr/bin/env bash
# Copyright 2026 Julian Pscheid
# Speed or KV allocation probes for one or more label=/path/model.gguf pairs.
set -euo pipefail

usage() {
  echo "usage: $0 speed|kv /path/to/llama/bin output-dir label=model.gguf [...]" >&2
  exit 2
}

[[ $# -ge 4 ]] || usage
mode=$1
bin_dir=$2
out_dir=$3
shift 3
mkdir -p "$out_dir"

active_pid=""
cleanup() {
  if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
    kill "$active_pid" 2>/dev/null || true
    wait "$active_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

require_metal() {
  local output=$1
  # Backend init logs use "GPU name"; --list-devices uses the MTL device table.
  if grep -Eq 'failed to create command queue|GPU name: \(null\)' "$output" ||
     ! grep -Eiq 'GPU name:.*Apple|^[[:space:]]*MTL[0-9]+:.*Apple' "$output"; then
    echo "Metal GPU not positively identified in $output; refusing CPU/BLAS fallback" >&2
    exit 1
  fi
}

device_log="$out_dir/devices.log"
"$bin_dir/llama-bench" --list-devices >"$device_log" 2>&1
require_metal "$device_log"

case "$mode" in
  speed)
    prompt_tokens=${PROMPT_TOKENS:?set PROMPT_TOKENS from the production-shaped workload}
    generated_tokens=${GENERATED_TOKENS:-128}
    batch=${BATCH:?set BATCH from the current native configuration}
    repeats=${REPEATS:-3}
    for spec in "$@"; do
      label=${spec%%=*}
      model=${spec#*=}
      [[ "$label" != "$model" && "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || usage
      "$bin_dir/llama-bench" -m "$model" -p "$prompt_tokens" -n "$generated_tokens" \
        -b "$batch" -fa on -r "$repeats" --progress -o jsonl \
        >"$out_dir/speed_${label}.jsonl" 2>"$out_dir/speed_${label}.stderr"
      require_metal "$out_dir/speed_${label}.stderr"
    done
    ;;
  kv)
    contexts=${CONTEXTS:?set CONTEXTS, for example "8192 16384"}
    batch=${BATCH:?set BATCH from the current native configuration}
    swa_full=${SWA_FULL:?set SWA_FULL from the current native configuration}
    [[ "$swa_full" == on || "$swa_full" == off ]] || { echo "SWA_FULL must be on or off" >&2; exit 2; }
    port=${PORT:-8099}
    for spec in "$@"; do
      label=${spec%%=*}
      model=${spec#*=}
      [[ "$label" != "$model" && "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || usage
      for context in $contexts; do
        log="$out_dir/kv_${label}_ctx${context}_swa-${swa_full}.log"
        extra=()
        [[ "$swa_full" == on ]] && extra+=(--swa-full)
        "$bin_dir/llama-server" -m "$model" -c "$context" -ngl 999 -b "$batch" \
          -fa on "${extra[@]}" -lv 10 --port "$port" --host 127.0.0.1 -np 1 \
          >"$log" 2>&1 &
        active_pid=$!
        ready=0
        for _ in $(seq 1 300); do
          if grep -q 'llama_context: n_seq_max' "$log"; then
            ready=1
            break
          fi
          kill -0 "$active_pid" 2>/dev/null || break
          sleep 2
        done
        [[ $ready -eq 1 ]] || { echo "context init failed; inspect $log" >&2; exit 1; }
        require_metal "$log"
        kill "$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
        active_pid=""
      done
    done
    ;;
  *) usage ;;
esac

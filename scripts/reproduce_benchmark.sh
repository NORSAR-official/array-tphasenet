#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="config_1stat.yaml"
CONT_CONFIG=""
SKIP_TRAIN=0
SKIP_PREFLIGHT=0
DRY_RUN=0
AUTO_SKIP_TRAIN=0
MODEL_OVERRIDE=""
TEST_YEAR_COUNT=1

usage() {
  cat <<'EOF'
Usage:
  scripts/reproduce_benchmark.sh [options]

Options:
  --config <path>       Config file (default: config_1stat.yaml)
  --skip-train          Skip train.py (use existing model instead)
  --skip-preflight      Do not run scripts/check_inputs.py checks
  --dry-run             Print commands without executing
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

trim() {
  local s="$1"
  # shellcheck disable=SC2001
  s="$(echo "$s" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  echo "$s"
}

unquote() {
  local s="$1"
  s="${s%\"}"
  s="${s#\"}"
  s="${s%\'}"
  s="${s#\'}"
  echo "$s"
}

config_key_value() {
  local key="$1"
  local line
  line="$(
    awk -v key="$key" '
      /^[[:space:]]*#/ { next }
      $0 ~ "^[[:space:]]*" key "[[:space:]]*:" { print; exit }
    ' "${CONFIG}" 2>/dev/null || true
  )"
  if [[ -z "${line}" ]]; then
    echo ""
    return 0
  fi
  local value="${line#*:}"
  value="${value%%#*}"
  value="$(trim "${value}")"
  value="$(unquote "${value}")"
  echo "${value}"
}

config_bool() {
  local key="$1"
  local default_val="${2:-0}"
  local raw
  raw="$(config_key_value "${key}")"
  if [[ -z "${raw}" ]]; then
    echo "${default_val}"
    return 0
  fi
  case "${raw,,}" in
    1|true|yes|y|on) echo "1" ;;
    *) echo "0" ;;
  esac
}

config_model_override() {
  local raw
  raw="$(config_key_value "predict_with_other_model")"
  case "${raw,,}" in
    ""|false|0|none|null|no|off) echo "" ;;
    *) echo "${raw}" ;;
  esac
}

config_test_year_count() {
  local raw
  raw="$(config_key_value "test_years")"
  if [[ -z "${raw}" ]]; then
    echo "1"
    return 0
  fi
  raw="${raw// /}"
  raw="${raw#[}"
  raw="${raw%]}"
  if [[ -z "${raw}" ]]; then
    echo "0"
    return 0
  fi
  awk -F',' '{ print NF }' <<< "${raw}"
}

run_cmd() {
  local cmd="$1"
  echo "+ ${cmd}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    eval "${cmd}"
  fi
}

preflight() {
  local stage="$1"
  local stage_config="${2:-${CONFIG}}"
  if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
    run_cmd "python scripts/check_inputs.py --config \"${stage_config}\" --stage \"${stage}\""
  fi
}

echo "Reproducibility pipeline config: ${CONFIG}"
echo "Repo root: ${ROOT_DIR}"

CONT_CONFIG="${CONFIG}"
CONFIG_BASE="$(basename "${CONFIG}")"
CONFIG_DIR="$(dirname "${CONFIG}")"
if [[ "${CONFIG_BASE}" == "config_1statfullarray.yaml" ]]; then
  CONT_CANDIDATE="${CONFIG_DIR}/config_1statfullarray_cont.yaml"
  if [[ -f "${CONT_CANDIDATE}" ]]; then
    CONT_CONFIG="${CONT_CANDIDATE}"
    echo "[INFO] Using ${CONT_CONFIG} for continuous stages."
  else
    echo "[WARN] Expected ${CONT_CANDIDATE} for continuous stages, but it was not found. Using ${CONFIG}."
  fi
fi

if [[ "${SKIP_TRAIN}" -eq 0 && -f "${CONFIG}" ]]; then
  MODEL_OVERRIDE="$(config_model_override)"
  TEST_YEAR_COUNT="$(config_test_year_count)"
  if [[ "$(config_bool "only_predict" "0")" -eq 1 ]]; then
    SKIP_TRAIN=1
    AUTO_SKIP_TRAIN=1
  fi
elif [[ -f "${CONFIG}" ]]; then
  MODEL_OVERRIDE="$(config_model_override)"
  TEST_YEAR_COUNT="$(config_test_year_count)"
fi

if [[ "${AUTO_SKIP_TRAIN}" -eq 1 ]]; then
  echo "[INFO] Detected run.only_predict=True in ${CONFIG}; skipping training stage."
  if [[ -n "${MODEL_OVERRIDE}" ]]; then
    echo "[INFO] predict_with_other_model=${MODEL_OVERRIDE}; predict/evaluate stages will use that model."
  fi
fi

# Ensure standard output folders exist
run_cmd "mkdir -p output/models output/predictions output/continuous"

TRAIN_EXECUTED=0
if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
  preflight "train"
  run_cmd "python train.py --config \"${CONFIG}\""
  TRAIN_EXECUTED=1
fi

RUN_PREDICT_TEST=1
if [[ "${TRAIN_EXECUTED}" -eq 1 ]]; then
  # train.py already writes predictions_*.npz for the test set.
  # Keep explicit predict-test only when a model override is requested
  # or when multiple test years need per-year prediction outputs.
  if [[ -z "${MODEL_OVERRIDE}" && "${TEST_YEAR_COUNT}" -le 1 ]]; then
    RUN_PREDICT_TEST=0
    echo "[INFO] Skipping predict-test: train.py already produced test predictions for this config."
  fi
fi

if [[ "${RUN_PREDICT_TEST}" -eq 1 ]]; then
  preflight "predict-test"
  run_cmd "python predict_on_testdata.py --config \"${CONFIG}\""
fi

preflight "evaluate-test"
run_cmd "python evaluate_on_testdata.py --config \"${CONFIG}\""

preflight "predict-continuous" "${CONT_CONFIG}"
run_cmd "python predict_continuous.py -c \"${CONT_CONFIG}\""

preflight "evaluate-continuous" "${CONT_CONFIG}"
run_cmd "python evaluate_continuous.py --config \"${CONT_CONFIG}\""

echo "Reproducibility pipeline completed."

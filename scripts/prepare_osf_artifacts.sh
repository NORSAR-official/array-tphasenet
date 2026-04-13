#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PROJECT_ID="27FPK"
DOWNLOAD_DIR="${ROOT_DIR}/downloads/osf/artifacts"
MODELS_DIR="${ROOT_DIR}/output/models"
PREDICTIONS_DIR="${ROOT_DIR}/output/predictions"
SKIP_DOWNLOAD=0
DRY_RUN=0
PATTERNS=(
  "saved_model_*.tf"
  "predictions_*.npz"
  "saved_model_*.zip"
  "predictions_*.zip"
)
CUSTOM_PATTERNS_SET=0

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_osf_artifacts.sh [options]

Downloads pre-trained model and prediction artifacts from OSF into
output/models and output/predictions.

Options:
  --project <id>             OSF project ID (default: 27FPK)
  --downloads-dir <dir>      Directory for downloaded files
  --models-dir <dir>         Models output directory (default: ./output/models)
  --predictions-dir <dir>    Predictions output directory
                             (default: ./output/predictions)
  --pattern <glob>           Basename glob to select remote files (repeatable).
                             If provided, custom patterns replace defaults
                             (saved_model_*.tf, predictions_*.npz,
                             saved_model_*.zip, predictions_*.zip).
  --skip-download            Skip OSF fetch calls and only copy local matches
  --dry-run                  Print commands without executing
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT_ID="$2"
      shift 2
      ;;
    --downloads-dir)
      DOWNLOAD_DIR="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --predictions-dir)
      PREDICTIONS_DIR="$2"
      shift 2
      ;;
    --pattern)
      if [[ "${CUSTOM_PATTERNS_SET}" -eq 0 ]]; then
        PATTERNS=()
        CUSTOM_PATTERNS_SET=1
      fi
      PATTERNS+=("$2")
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
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

run_cmd() {
  local cmd="$1"
  echo "+ ${cmd}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    eval "${cmd}"
  fi
}

require_osfclient() {
  if command -v osf >/dev/null 2>&1; then
    return 0
  fi
  echo "[FAIL] 'osf' command not found. Install osfclient first:" >&2
  echo "       pip install osfclient" >&2
  return 1
}

osf_list() {
  if osf -p "${PROJECT_ID}" ls >/dev/null 2>&1; then
    osf -p "${PROJECT_ID}" ls
  else
    osf -p "${PROJECT_ID}" list
  fi
}

discover_remote_files() {
  local -a lines=()
  local -A seen=()
  local -a matches=()
  local line trimmed base pattern

  mapfile -t lines < <(osf_list)
  for line in "${lines[@]}"; do
    trimmed="$(echo "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "${trimmed}" ]] && continue
    [[ "${trimmed}" == */ ]] && continue
    base="${trimmed##*/}"
    for pattern in "${PATTERNS[@]}"; do
      if [[ "${base}" == ${pattern} ]]; then
        if [[ -z "${seen[${trimmed}]+x}" ]]; then
          matches+=("${trimmed}")
          seen["${trimmed}"]=1
        fi
        break
      fi
    done
  done

  printf "%s\n" "${matches[@]}"
}

destination_for() {
  local base="$1"
  if [[ "${base}" == saved_model_*.tf ]]; then
    printf "%s\n" "${MODELS_DIR}"
  elif [[ "${base}" == predictions_*.npz ]]; then
    printf "%s\n" "${PREDICTIONS_DIR}"
  else
    printf "%s\n" "${DOWNLOAD_DIR}"
  fi
}

# Only the top level of --downloads-dir (no subdirectory search).
find_matching_artifact_paths() {
  for pat in "${PATTERNS[@]}"; do
    find "${DOWNLOAD_DIR}" -maxdepth 1 \( -type f -o -type d \) \
      -name "${pat}" -print 2>/dev/null
  done | sort -u
}

sync_artifact_to_output() {
  local src="$1"
  local base="${src##*/}"
  local dest_dir
  dest_dir="$(destination_for "${base}")"
  if [[ "${dest_dir}" == "${DOWNLOAD_DIR}" ]]; then
    return 0
  fi
  if [[ -d "${src}" ]]; then
    run_cmd "rm -rf \"${dest_dir}/${base}\""
    run_cmd "cp -a \"${src}\" \"${dest_dir}/${base}\""
  elif [[ -f "${src}" ]]; then
    run_cmd "cp -f \"${src}\" \"${dest_dir}/${base}\""
  fi
}

extract_zip_archive() {
  local archive="$1"
  local target_dir="$2"
  echo "+ extract zip \"${archive}\" -> \"${target_dir}\""
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  python - "${archive}" "${target_dir}" <<'PY'
import os
import sys
import zipfile

archive = sys.argv[1]
target = sys.argv[2]
os.makedirs(target, exist_ok=True)
with zipfile.ZipFile(archive, "r") as zf:
    zf.extractall(target)
PY
}

echo "Repo root: ${ROOT_DIR}"
echo "OSF project: ${PROJECT_ID}"
echo "Downloads dir: ${DOWNLOAD_DIR}"
echo "Models dir: ${MODELS_DIR}"
echo "Predictions dir: ${PREDICTIONS_DIR}"
echo "Patterns:"
for pattern in "${PATTERNS[@]}"; do
  echo "  - ${pattern}"
done

run_cmd "mkdir -p \"${DOWNLOAD_DIR}\" \"${MODELS_DIR}\" \"${PREDICTIONS_DIR}\""

if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  require_osfclient
  echo "Discovering matching OSF files..."
  mapfile -t REMOTE_FILES < <(discover_remote_files)
  if [[ "${#REMOTE_FILES[@]}" -eq 0 ]]; then
    echo "[FAIL] No remote OSF files matched the configured patterns." >&2
    echo "       You can inspect project files with: osf -p ${PROJECT_ID} ls" >&2
    exit 1
  fi
  echo "Matched remote files:"
  for remote in "${REMOTE_FILES[@]}"; do
    echo "  - ${remote}"
  done
  echo "Starting downloads with osfclient..."
  for remote in "${REMOTE_FILES[@]}"; do
    run_cmd "osf -p \"${PROJECT_ID}\" fetch \"${remote}\" \"${DOWNLOAD_DIR}/${remote##*/}\""
  done
fi

echo "Extracting zip artifacts (if any)..."
while IFS= read -r archive; do
  [[ -z "${archive}" ]] && continue
  base="${archive##*/}"
  [[ "${base}" == *.zip ]] || continue
  extract_zip_archive "${archive}" "${DOWNLOAD_DIR}"
done < <(
  for pat in "${PATTERNS[@]}"; do
    find "${DOWNLOAD_DIR}" -maxdepth 1 -type f -name "${pat}" -print 2>/dev/null
  done | sort -u
)

echo "Copying downloaded artifacts to output directories..."
mapfile -t artifacts < <(find_matching_artifact_paths)
if [[ "${#artifacts[@]}" -eq 0 ]]; then
  echo "[WARN] No paths directly under ${DOWNLOAD_DIR} matched patterns."
  echo "       Place model/prediction files (or matching zips) in that directory." >&2
else
  for src in "${artifacts[@]}"; do
    [[ -e "${src}" ]] || continue
    base="${src##*/}"
    if [[ "${base}" == *.zip ]]; then
      continue
    fi
    sync_artifact_to_output "${src}"
  done
fi

echo "OSF artifact preparation complete."

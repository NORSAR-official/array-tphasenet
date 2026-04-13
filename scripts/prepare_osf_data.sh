#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PROJECT_ID="27FPK"
DOWNLOAD_DIR="${ROOT_DIR}/downloads/osf"
DATA_DIR="${ROOT_DIR}/data"
SKIP_DOWNLOAD=0
SKIP_MERGE=0
DRY_RUN=0
PATTERNS=(
  "*.hdf5"
)
CUSTOM_PATTERNS_SET=0

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_osf_data.sh [options]

Downloads selected OSF files via osfclient and merges known split HDF5 chunks
into the local data folder.

Options:
  --project <id>           OSF project ID (default: 27FPK)
  --downloads-dir <dir>    Directory for downloaded chunk files
  --data-dir <dir>         Data output directory (default: ./data)
  --pattern <glob>         Basename glob to select remote files (repeatable).
                           If provided, custom patterns replace the default
                           '*.hdf5' selection.
                           Example: --pattern "1statfullarray_2022_*_000*.hdf5"
  --skip-download          Skip download step
  --merge-only             Alias for --skip-download
  --skip-merge             Skip merge step
  --dry-run                Print commands without executing
  -h, --help               Show this help
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
    --data-dir)
      DATA_DIR="$2"
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
    --skip-download|--merge-only)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-merge)
      SKIP_MERGE=1
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

merge_pattern() {
  local pattern="$1"
  local output="$2"
  shopt -s nullglob
  local matches=( ${pattern} )
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "[WARN] No files match pattern: ${pattern}"
    return 0
  fi
  run_cmd "python split_data_for_repo.py merge --pattern \"${pattern}\" --output \"${output}\""
}

is_split_chunk() {
  local base="$1"
  [[ "${base}" == *_000*.hdf5 ]]
}

echo "Repo root: ${ROOT_DIR}"
echo "OSF project: ${PROJECT_ID}"
echo "Downloads dir: ${DOWNLOAD_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Patterns:"
for pattern in "${PATTERNS[@]}"; do
  echo "  - ${pattern}"
done

run_cmd "mkdir -p \"${DOWNLOAD_DIR}\" \"${DATA_DIR}\""

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

if [[ "${SKIP_MERGE}" -eq 0 ]]; then
  echo "Running merge step for known split files..."

  merge_pattern \
    "${DOWNLOAD_DIR}/1statfullarray_2022_single_station_waveforms_000*.hdf5" \
    "${DATA_DIR}/1statfullarray_2022_single_station_waveforms.hdf5"

  merge_pattern \
    "${DOWNLOAD_DIR}/array25arces_2022_array_waveforms_000*.hdf5" \
    "${DATA_DIR}/arces25_2022_array_waveforms.hdf5"

  # Alternative prefix seen in some user downloads.
  merge_pattern \
    "${DOWNLOAD_DIR}/arces25_2022_array_waveforms_000*.hdf5" \
    "${DATA_DIR}/arces25_2022_array_waveforms.hdf5"
fi

echo "Syncing non-split HDF5 files into data dir..."
shopt -s nullglob
downloaded_hdf5=( "${DOWNLOAD_DIR}"/*.hdf5 )
shopt -u nullglob
if [[ "${#downloaded_hdf5[@]}" -eq 0 ]]; then
  echo "[WARN] No HDF5 files found in ${DOWNLOAD_DIR}."
else
  for src in "${downloaded_hdf5[@]}"; do
    base="${src##*/}"
    if is_split_chunk "${base}"; then
      continue
    fi
    run_cmd "cp -f \"${src}\" \"${DATA_DIR}/${base}\""
  done
fi

echo "OSF data preparation complete."

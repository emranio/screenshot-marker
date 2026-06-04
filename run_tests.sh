#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

SCREENS_DIR="${ROOT_DIR}/tests/screens"
ANNOTATIONS_DIR="${ROOT_DIR}/tests/annotations"
OUTPUT_DIR="${ROOT_DIR}/tests/rendered"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

if [[ ! -d "${SCREENS_DIR}" ]]; then
  echo "Missing test screenshots directory: ${SCREENS_DIR}" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

shopt -s nullglob
for image_path in "${SCREENS_DIR}"/*.{png,jpg,jpeg,webp}; do
  stem="$(basename "${image_path}")"
  stem="${stem%.*}"
  source_json="${ANNOTATIONS_DIR}/${stem}.json"

  if [[ ! -f "${source_json}" ]]; then
    echo "Skipping ${stem}: no matching ${source_json}" >&2
    continue
  fi

  queries_file="${tmp_dir}/${stem}-queries.json"
  result_file="${tmp_dir}/${stem}-result.json"
  "${PYTHON_BIN}" - "${source_json}" "${queries_file}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
data = json.loads(source.read_text())
if isinstance(data, list):
    queries = data
else:
    queries = [ann["request_text"] for ann in data.get("annotations", [])]
dest.write_text(json.dumps(queries, indent=2))
PY

  if [[ ! -s "${queries_file}" ]]; then
    echo "Skipping ${stem}: no queries found in ${source_json}" >&2
    continue
  fi

  echo "Annotating ${stem}..."
  "${PYTHON_BIN}" "${ROOT_DIR}/annotate.py" \
    --image "tests/screens/$(basename "${image_path}")" \
    --queries-file "${queries_file}" \
    --output "tests/rendered/${stem}.png" \
    "$@" \
    > "${result_file}"
  mv "${result_file}" "${source_json}"
done

echo "Done. Outputs written to ${OUTPUT_DIR}"

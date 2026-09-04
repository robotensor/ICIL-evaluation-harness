#!/usr/bin/env bash
# Build the ICIL image for a vla-eval simulator image: ./docker/build.sh libero [tag]
set -euo pipefail
name="${1:-libero}"
tag="${2:-icil-eval/${name}:dev}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
rm -rf dist && python -m pip wheel --no-deps -w dist . >/dev/null 2>&1 || uv build --wheel -o dist >/dev/null
wheel="$(ls dist/*.whl | head -1)"
docker build -f "docker/Dockerfile.${name}" --build-arg WHEEL="${wheel}" -t "${tag}" .
echo "built ${tag} from ${wheel}"

#!/usr/bin/env bash
set -euo pipefail
dev=0
role=draft
gpu=()
for argument in "$@"; do
  case "$argument" in
    --dev) dev=1 ;;
    --target) role=target ;;
    --gpu) gpu=(--gpu) ;;
    --help)
      echo 'Usage: bash tools/install.sh [--target] [--dev] [--gpu]'
      echo 'For an already activated isolated venv. New users: bash setup.sh'
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
bash tools/run_in_env.sh true
command -v git >/dev/null || { echo 'Install Git before setup.' >&2; exit 2; }
python - <<'CHECK'
import ctypes
import platform
name, version = platform.libc_ver()
if name != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 34):
    raise SystemExit('Pinned SGLang wheels require glibc >= 2.34; use a newer Linux base image.')
if platform.machine() != 'x86_64':
    raise SystemExit('The pinned runtime requires Linux x86-64.')
for name in ('libibverbs.so.1', 'libmlx5.so.1'):
    try:
        ctypes.CDLL(name)
    except OSError as error:
        raise SystemExit(f'{name}: {error}. Install rdma-core runtime libraries '
                         '(Ubuntu: libibverbs1 libmlx5-1), or use the container.') from error
CHECK
if ! printf '#include <bit>\nint main() { return std::bit_cast<unsigned>(1.0f) == 0; }\n' |
  "${CXX:-c++}" -std=c++20 -x c++ -fsyntax-only -; then
  echo 'SGLang JIT needs a C++20 compiler and matching standard library.' >&2
  echo 'Install a modern C++ toolchain (Ubuntu: g++), or set CXX to one; see installation-advanced.md.' >&2
  exit 2
fi
manifest=requirements/draft.txt
check=()
if [[ $role == target ]]; then
  manifest=requirements/sglang-v0.5.21.txt
  check=(--target)
fi
extras="."
(( dev == 0 )) || extras=".[dev]"
python -m pip install -r "$manifest" -e "$extras"
printf '%s\n' "$role" > "$VIRTUAL_ENV/.aspec-role"
if [[ $role == target ]]; then
  aspec patch-sglang
  aspec patch-sglang --check
fi
aspec doctor "${check[@]}" "${gpu[@]}"

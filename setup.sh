#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv="$repo_root/.venv"
base_python=""
options=()
role=draft
explicit_venv=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv|--python)
      [[ $# -ge 2 && -n $2 && $2 != --* ]] || { echo "$1 requires a value" >&2; exit 2; }
      if [[ $1 == --venv ]]; then venv=$2; explicit_venv=1; else base_python=$2; fi
      shift 2 ;;
    --target) role=target; options+=("$1"); shift ;;
    --dev|--gpu) options+=("$1"); shift ;;
    --help)
      echo 'Usage: bash setup.sh [--venv PATH] [--python EXECUTABLE] [--target] [--dev] [--gpu]'
      echo 'Creates an isolated Linux Python >=3.11 venv and installs pinned dependencies.'
      echo 'Defaults to .venv. Existing venvs require this checkout owner.'
      echo 'Does not install host packages/drivers, download models, start training or clean clusters.'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ $role == target && $explicit_venv == 0 ]]; then venv="$repo_root/.venv-target"; fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo 'Use Linux x86-64; see docs/user-guide/installation.md.' >&2; exit 2;
}
[[ -z ${PYTHONPATH:-} && -z ${PYTHONHOME:-} ]] || {
  echo 'Unset PYTHONPATH and PYTHONHOME before setup.' >&2; exit 2;
}
if [[ -z $base_python ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] < (3,11))'; then
      base_python=$candidate; break
    fi
  done
fi
[[ -n $base_python ]] || { echo 'Install Python >=3.11 or pass --python EXECUTABLE.' >&2; exit 2; }
"$base_python" - <<'CHECK'
import sys
if sys.version_info[:2] < (3, 11):
    raise SystemExit("Python >=3.11 is required")
CHECK
venv=$("$base_python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$venv")
if [[ -e $venv || -L $venv ]]; then
  [[ -f $venv/.aspec-source && $(cat "$venv/.aspec-source") == "$repo_root" && -x $venv/bin/python ]] || {
    echo "Refusing an unowned environment: $venv. Choose a new --venv path." >&2; exit 2;
  }
else
  "$base_python" -m venv "$venv"
  printf '%s\n' "$repo_root" > "$venv/.aspec-source"
fi
source "$venv/bin/activate"
cd "$repo_root"
bash tools/install.sh "${options[@]}"
printf '\nSetup complete. Activate with:\n  source %q\n' "$venv/bin/activate"
if [[ $role == draft ]]; then
  printf 'Prepare Target separately: bash setup.sh --target\nActivate its environment and start aspec target on the allocated Target GPUs.\nStart training from the Draft environment on its allocated GPUs.\n'
fi

#!/bin/bash
# Quick 100-question eval on a LoRA adapter (80 answerable + 20 unanswerable).
# Sets LD_LIBRARY_PATH before Python starts so bitsandbytes finds libnvJitLink.so.13.
#
# Usage:
#   bash scripts/run_quick_eval.sh --adapter PATH [--output PATH] [--save-predictions PATH]

set -euo pipefail

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

_NV_CU13=$(python3 -c "
import sys, os
sp = next(p for p in sys.path if 'site-packages' in p)
lib = os.path.join(sp, 'nvidia', 'cu13', 'lib')
print(lib)
" 2>/dev/null || echo "")
if [ -n "$_NV_CU13" ] && [ -d "$_NV_CU13" ]; then
    export LD_LIBRARY_PATH="$_NV_CU13:${LD_LIBRARY_PATH:-}"
    echo "LD_LIBRARY_PATH set: $_NV_CU13"
fi

python3 scripts/quick_eval_adapter.py "$@"

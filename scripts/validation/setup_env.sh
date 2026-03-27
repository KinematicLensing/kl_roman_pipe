#!/bin/bash
# Create a validation environment with both kl_pipe and geko installed.
# geko (astro-geko) is not part of the main klpipe conda env.
#
# Idempotent: skips creation if env already exists with geko importable.
# Pass -q/--quiet to suppress output when env already exists (used by Makefile deps).
set -e

QUIET=false
for arg in "$@"; do
    case "$arg" in -q|--quiet) QUIET=true ;; esac
done

ENV_NAME="klpipe_validation"

if conda env list | grep -q "$ENV_NAME"; then
    # env exists — verify geko is importable
    if conda run -n "$ENV_NAME" python -c "import geko" 2>/dev/null; then
        $QUIET || echo "Environment '$ENV_NAME' already exists with geko installed. Nothing to do."
        exit 0
    else
        echo "Environment '$ENV_NAME' exists but geko not importable. Installing..."
        conda run -n "$ENV_NAME" pip install astro-geko
        exit 0
    fi
fi

echo "Cloning klpipe -> $ENV_NAME ..."
conda create --name "$ENV_NAME" --clone klpipe -y

echo "Installing astro-geko ..."
conda run -n "$ENV_NAME" pip install astro-geko

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
echo "Then run: make render-validation-geko"

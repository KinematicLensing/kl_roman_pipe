#!/bin/bash
# Create a validation environment with both kl_pipe and geko installed.
# geko (astro-geko) is not part of the main klpipe conda env.
set -e

ENV_NAME="klpipe_validation"

if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Remove it first with:"
    echo "  conda env remove -n $ENV_NAME"
    exit 1
fi

echo "Cloning klpipe -> $ENV_NAME ..."
conda create --name "$ENV_NAME" --clone klpipe -y

echo "Installing astro-geko ..."
conda run -n "$ENV_NAME" pip install astro-geko

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
echo "Then run: make render-validation-geko"

# exit as soon as a command fails:
set -e

REPO_DIR="$(realpath "$(dirname "$0")")"
CWD=$(pwd)

# not all users can control their conda base environment (e.g. at HPCs), so we
# allow the user to specify a different conda environment to use for the base
BASE_ENV=${BASE_ENV:-base}

ENVFILE="$(dirname "$0")"/environment.yaml
ENVNAME="$(grep '^name:' "$ENVFILE" | cut -d' ' -f2)"

echo ENVFILE="$ENVFILE"
echo ENVNAME="$ENVNAME"

# Always execute this script with bash, so that conda shell.hook works.
# Relevant conda bug: https://github.com/conda/conda/issues/7980
if test "$BASH_VERSION" = ""
then
    exec bash "$0" "$@"
fi

eval "$(conda shell.bash hook)"

# an existing environment is replaced only on explicit confirmation
# (interactive prompt, or KLPIPE_REINSTALL=1 when there is no terminal)
if conda env list | awk '{print $1}' | grep -qx "$ENVNAME"; then
    if [ -t 0 ]; then
        read -r -p "Environment '$ENVNAME' exists and will be removed and reinstalled. Continue? [y/N] " reply
        if [ "$reply" != "y" ] && [ "$reply" != "Y" ]; then
            echo "Aborted; '$ENVNAME' left unchanged."
            exit 1
        fi
    elif [ "${KLPIPE_REINSTALL:-0}" != "1" ]; then
        echo "ERROR: environment '$ENVNAME' exists. Set KLPIPE_REINSTALL=1 to replace it non-interactively." >&2
        exit 1
    fi
    echo "Removing existing environment '$ENVNAME'..."
    conda deactivate || true
    conda env remove -n "$ENVNAME" --yes
fi

# install environment fresh
echo "Installing '$ENVNAME' from reproducible conda-lock.yml..."
conda run -n ${BASE_ENV} conda-lock install --name "$ENVNAME" "$REPO_DIR/conda-lock.yml"

# activate conda environment
conda activate "$ENVNAME"

echo "cd $REPO_DIR"
cd $REPO_DIR

echo "Pip installing kl_roman_test..."
pip install --no-build-isolation --no-deps --editable "$REPO_DIR/."

#echo "Pip installing my special repo..."
#pip install --no-build-isolation --no-deps --editable "$PATH_TO_REPO/."

echo "Installing pre-commit hooks..."
pre-commit install

echo "cd $CWD"
cd "$CWD"

echo "conda deactivate"
conda deactivate

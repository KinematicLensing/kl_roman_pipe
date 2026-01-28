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

# check if the environment already exists
if conda info --envs | grep -q "$ENVNAME"; then
    echo "Environment '$ENVNAME' already exists. Removing it first..."
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

echo "cd $CWD"
cd "$CWD"

echo "conda deactivate"
conda deactivate

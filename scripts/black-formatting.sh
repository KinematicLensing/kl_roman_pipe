#!/bin/bash

# This script executes the `black` formatter on all the python files in the
# git repository. Command line arguments get passed to `black`, e.g.,
#
#	$ ./scripts/black-formatting.sh --check
#
# will only check and not modify the files.

set -e

# Always execute this script with bash, so that conda shell.hook works.
# Relevant conda bug: https://github.com/conda/conda/issues/7980
if test "$BASH_VERSION" = ""
then
	exec bash "$0" "$@"
fi

. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate klpipe


# Only include python files in the current repo (excluding submodules):
PYFILES="`git ls-files '*.py'`"

## add jupyter notebooks:
#PYFILES="$PYFILES `git ls-files '*.ipynb'`"

echo PYFILES=$PYFILES

if test "$PYFILES" != ""; then
	black --skip-string-normalization "$@" $PYFILES
fi

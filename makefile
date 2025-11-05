# Makefile

DATA_DIR = data
TEST_DIR = tests
TEST_DATA_DIR = $(TEST_DIR)/data

# TODO: eventually incorporate conda-lock when it is warranted
GENERATE_CONDA_LOCK = cd "$(shell dirname "$(1)")"; conda-lock -f "$(shell basename "$(2)")" -p osx-64 -p osx-arm64 -p linux-64

# NOTE: we can add any required test data files here that need to be
# generated or downloaded before running unit tests
UNIT_TEST_FILES = 
# UNIT_TEST_FILES = .../unit_test_file1 .../unit_test_file2

.PHONY: install
install:
	@echo "Installing kl_roman_test repository..."
	@bash install.sh
	@echo "kl_roman_test environment installed."


# Run tests
.PHONY: test
test: test-data
	@echo "Running tests..."
	@conda run -n kltest python -m unittest discover -s $(TEST_DIR) -p "test_*.py"

# Download or generate test data
.PHONY: test-data
test-data: $(UNIT_TEST_FILES)

# Regenerate the conda-lock.yml file
conda-lock.yml:
	@echo "Regenerating $@..."
	@$(call GENERATE_CONDA_LOCK,$@,environment.yaml)

#-------------------------------------------------------------------------------
# NOTE: These may be useful in the future if we use git submodules

# update-submodules:
# 	git submodule update --init --recursive --remote  # use `branch` in .gitmodules to update

# ...

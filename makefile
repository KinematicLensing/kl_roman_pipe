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

TNG50_DATA_DIR = $(DATA_DIR)/tng50

# NOTE: I cannot currently get this to automatically download due to
# issues with sharepoint; easiest solution is to move this somewhere
# easier to access, such as a UA server
TNG50_DRIVE_URL = https://emailarizona-my.sharepoint.com/:f:/g/personal/ylai2_arizona_edu/ElDBfFY6hGpFgCEYc4DugfEBd4DPxGf2z6v60PISgH4RLA?e=0t3RhO

FORMATTER = ./scripts/black-formatting.sh

WGET ?= wget

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

# Format code
.PHONY: format
format:
	@$(FORMATTER)

# Check the format of the code; **does not reformat the code**
.PHONY: check-format
check-format:
	@$(FORMATTER) --check

#-------------------------------------------------------------------------------
# data file downloads

.PHONY: download-tng50
download-tng50:
	@echo "Downloading TNG50 data files..."
	@mkdir -p $(TNG50_DATA_DIR)
#	@$(WGET) -r -np -nd -N --tries=5 --timeout=15 -R 'index.html*' \
	    -P '$(TNG50_DATA_DIR)' '$(TNG50_DRIVE_URL)'
#	@echo "TNG50 data files downloaded to $(TNG50_DATA_DIR)"
	@echo "NOTE: For now, please manually download the TNG50 data files from the provided SharePoint link."
	@echo "URL: $(TNG50_DRIVE_URL)"
	@echo "Destination: $(TNG50_DATA_DIR)"

#-------------------------------------------------------------------------------
# NOTE: These may be useful in the future if we use git submodules

# update-submodules:
# 	git submodule update --init --recursive --remote  # use `branch` in .gitmodules to update

# ...

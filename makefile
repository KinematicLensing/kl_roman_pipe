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

# CyVerse data configuration
CYVERSE_DATA_DIR = $(DATA_DIR)/cyverse
CYVERSE_CONFIG = $(CYVERSE_DATA_DIR)/cyverse_data.conf
CYVERSE_DOWNLOAD_SCRIPT = scripts/download_cyverse_data.sh
CYVERSE_DATA_MARKER = $(CYVERSE_DATA_DIR)/.cyverse_data_downloaded

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

.PHONY: download-cyverse-data
download-cyverse-data:
	@echo "Downloading CyVerse data files..."
	@mkdir -p $(CYVERSE_DATA_DIR)
	@bash $(CYVERSE_DOWNLOAD_SCRIPT) $(CYVERSE_CONFIG)
	@touch $(CYVERSE_DATA_MARKER)
	@echo "CyVerse data download complete."

# Target for checking if CyVerse data has been downloaded
$(CYVERSE_DATA_MARKER): $(CYVERSE_CONFIG)
	@$(MAKE) download-cyverse-data

.PHONY: clean-cyverse-data
clean-cyverse-data:
	@echo "Removing downloaded CyVerse data files..."
	@if [ -f "$(CYVERSE_CONFIG)" ]; then \
		while IFS='|' read -r public_path local_path || [ -n "$$public_path" ]; do \
			[ -z "$$public_path" ] || echo "$$public_path" | grep -q "^[[:space:]]*#" && continue; \
			local_path=$$(echo "$$local_path" | xargs); \
			local_file="$(DATA_DIR)/$$local_path"; \
			if [ -f "$$local_file" ]; then \
				echo "Removing: $$local_file"; \
				rm -f "$$local_file"; \
			fi; \
		done < "$(CYVERSE_CONFIG)"; \
	fi
	@rm -f $(CYVERSE_DATA_MARKER)
	@echo "CyVerse data removed."

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
# test related targets

# Download or generate test data
.PHONY: test-data
test-data: $(UNIT_TEST_FILES)


.PHONY: test
test:
	@conda run -n klpipe pytest tests/ -v

.PHONY: test-tng50
test-tng50: $(CYVERSE_DATA_MARKER)
	@echo "Running TNG50 tests with CyVerse data..."
	@conda run -n klpipe pytest tests/ -v -m tng50

.PHONY: test-all
test-all: $(CYVERSE_DATA_MARKER)
	@echo "Running all tests (basic + TNG50)..."
	@conda run -n klpipe pytest tests/ -v

.PHONY: test-coverage
test-coverage:
	@conda run -n klpipe pytest tests/ -v --cov=kl_pipe --cov-report=html --cov-report=term-missing

.PHONY: test-fast
test-fast:
	@conda run -n klpipe && pytest tests/ -v -x

.PHONY: test-verbose
test-verbose:
	@conda run -n klpipe && pytest tests/ -v -s

.PHONY: test-clean
clean-test:
	rm -rf tests/out/
	#rm -rf .pytest_cache/
	#rm -rf htmlcov/
	rm -rf .coverage

#-------------------------------------------------------------------------------
# NOTE: These may be useful in the future if we use git submodules

# update-submodules:
# 	git submodule update --init --recursive --remote  # use `branch` in .gitmodules to update

# ...

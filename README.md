# kl_roman_test
Test repo to try out a new pipeline approach for Roman KL

NOTE: currently under significant construction. New users should check out the latest branch until things have settled (currently `se/basic_models`)

## Installation Instructions

1) Get `conda` (I recommend [miniforge](https://github.com/conda-forge/miniforge))
2) Get `conda-lock` in your base env: `conda install conda-lock`
3) `make install`
4) `make test`

## CyVerse Data Dow

More advanced pipline tests require the TNG50 files hosted on CyVerse. To grab them, simply do the following:

1. `make download-cyverse-data`
2. Follow instructions for setting up authentification with a `.netrc` file if you are downloading restricted files. This should only have to be done once.

NOTE: Until Spencer is added to the KL space on CyVerse, it currently just downloads a file called `success.txt` to confirm CyVerse data downloading success. Will be updated soon!

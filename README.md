# Embedding benchmarking

## What is this?

Collection of scripts to analyze the different factor impact on embedding model results in benchmarking.

Done as a part of Master's Thesis at "University of Latvia".

## Required libraries

See `/requirements.txt`.

## Structure

- `/computed` directory contains the raw evaluations for the model x task pairs.
- `/common` directory contains helper scripts which are used in multiple places.
- `.env.example` - file which shows example required environment variables to run scripts
- `/requirements.txt` - python dependencies
- `/analysis/domains` - all scripts to analyze factors of single language tasks
- `/analysis/languages` - all scripts to analyze factors of multi language tasks
- `/analysis/lv` - all scripts to analyze embedding performance on Latvian tasks
- `/analysis/code` - all scripts to evaluate code finetuned models
- `/buildThesisContent.sh` - runs all the scripts to output content for the thesis

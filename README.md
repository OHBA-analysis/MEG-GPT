# EphysGPT

Foundation models for electrophysiological data.

## Installation

First install EphysGPT:
```
git clone git@github.com:OHBA-analysis/EphysGPT.git
cd EphysGPT
mamba env create -f envs/egpt.yml
conda activate egpt
pip install -e .
```

Then install the data loaders:
```
git clone git@github.com:OHBA-analysis/EphysDataLoaders.git
cd EphysDataLoaders
pip install -e .
```

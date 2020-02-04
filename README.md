# conv_qsar_fast
QSAR/QSPR using descriptor-free molecular embedding, updated for python 3

In the release `python3minimalchange` you will find the version of code which runs under python3 and closely resembles the original code (as opposed to the newest version). You can find it under releases or switch to tag `python3minimalchange` (same button as branch switching).

The newest version has been changed dramatically:
- splitting datasets has been removed, now each fold/subset should be kept in a separate file. As a result:
  - file `main/main_cv.py` has seen major changes
  - file `main/data.py` has been simplified a lot. No more splitting, and no more checking which set is used - every information is now kept in config files.
  - as a result config files have changed a lot. Unused fields have been removed, new fields appeared:
    - `cv` - cross-validation or train-val-test?
    - `fold1`, ..., `foldk` - cross validation splits if `cv: true`
    - `train`, `val` -  train and valid if `cv: false`
    - `test` - the test set, always
  - `scripts/split_datasets.py` includes the code for splitting Abraham, Bradley and Delany datasets (you will find them already calculated and included in `data`)
  - TOX21 is not cross-valed anymore, instead the split from the challenge is used
- `main/test.py` has seen major changes, most importantly you get to choose if rocauc and parity plots should be calculated, the plots and jsons with the score are saved. New section `TEST` and new fields (`calculate_parity`, `calculate_rocauc`) appeared in the config files
- each config file has a full information about the dataset (so that you don't need all those ifs when loading the data), new fields appeared:
  - `y_index` - label index
  - `y_label` - label name used for titles of plots
  - `delimiter` - delimiter used in the data format, used by csv.reader
  - `skip_line` - some datasets have column names in the first line (set `true`), others don't (set `false`)
- in the original code TOX21 was not averaged, now it is. New field (`averaging`) in config files appeared and can have the following values:
  - `mean` - mean value is calculated and set as label (this was originally used for Abraham, Bradley and Delaney)
  - `max` - max value is set as label (new stuff, for TOX)
- changes in mutiple files to make the code more readable, removing files that were not used anymore

## Requirements
This code relies on [Keras](http://keras.io/) for the machine learning framework, [Theano](http://deeplearning.net/software/theano/) for computations as its back-end, and [RDKit](http://www.rdkit.org/) for parsing molecules from SMILES strings. Plotting is done in [matplotlib](http://matplotlib.org/). All other required packages should be dependencies of Keras, Theano, or RDKit.

- python 3.7.6
- tensorflow 1.14.0
- keras 2.2.4
- theano 1.0.4
- rdkit 2019.09.3.0

Full copy of environment is listed in directory `environment`


## Issues

Even though result on TOX21 are similar to those in the paper, the model does not learn anything on Abraham, Bradley nor Delaney.
Learning on fingerprints (ECFP) does not work.


## Basic use
This code implements the tensor-based convolutional embedding strategy described in __placeholder__ for QSAR/QSPR tasks. The model architecture, training schedule, and data source are defined in a configuration file and trained using a cross-validation (CV). The basic architecture is as follows:

- Pre-processing to convert a SMILES string into an attributed graph, then into an attributed adjacency tensor
- Convolutional embedding layer, which takings a molecular tensor and produces a learned feature vector
- Optional dropout layer
- Optional hidden densely-connected neural network layer
- Optional dropout layer
- Optional second hidden densely-connected neural network layer
- Linear dense output layer

Models are built, trained, and tested with the command
```
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/<input_file>.cfg
```

Numerous example input files, corresponding the models described in __placeholder__ are included in `inputs`. These include models to be trained on full datasets, 5-fold CVs with internal validation and early stopping, 5-fold CVs without internal validation, models initialized with weights from other trained models, and multi-task models predicting on multiple data sets. Note that when using multi-task models, the `output_size` must be increased and the `loss` function must be `custom` to ensure `NaN` values are filtered out if not all inputs x have the full set of outputs y.

## Data sets
There are four available data sets in this version of the code contained in `data`:

1. Abraham octanol solubility data, from Abraham and Admire's 2014 paper.
2. Delaney aqueous solubility data, from Delaney's 2004 paper.
3. Bradley double plus good melting point data, from Bradley's open science notebook initiative.
4. Tox21 data from the Tox21 Data Challenge 2014, describing toxicity against 12 targets.

Because certain entries could not be unambiguously resolved into chemical structures, or because duplicates in the data sets were found, the effective data sets after processing are exported using `scripts/save_data.py` as `coley_abraham.tdf`, `coley_delaney.tdf`, `coley_bradley.tdf`, `coley_tox21.tdf`, and `coley_tox21-test.tdf`.

## Model interpretation
This version of the code contains the general method of non-linear model interpretation of assigning individual atom and bond attributes to their average value in the molecular tensor representation. The extent to which this hurts performance is indicative of how dependent a trained model has become on that atom/bond feature. As long as the configuration file defines a model which loads previously-trained weights, the testing routine is performed by
```
python conv_qsar_fast/main/test_index_removal.py conv_qsar_fast/inputs/<input_file>.cfg
```
It is assumed that the trained model used molecular_attributes, as the indices for removal are hard-coded into this script.

## Suggestions for modification
#### Overall architecture
The range of possible architectures (beyond what is enabled with the current configuration file style) can be extended by modifying `build_model` in `main/core.py`. See the Keras documentation for ideas.

#### Data sets
Additional `.csv` data sets can be incorporated by adding an additional `elif` statement to `main/data.py`. As long as one column corresponds to SMILES strings and another to the property target, the existing code can be used with minimal modification.

#### Atom-level or bond-level attributes
Additional atom- or bond-level attributes can be included by modifying `utils/neural_fp.py`, specifically the `bondAttributes` and `atomAttributes` functions. Because molecules are already stored as RDKit molecule objects, any property calculable in RDKit can easily be added.

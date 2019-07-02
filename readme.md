# Skoltech EEG research repository

Includes:

* Preprocessing scripts for datasets we use
* Grid search pipeline for resting state EEG classification
* Experiment log
* Selected notebooks with experiment results in `master` branch
* Branches with experiments

## Tasks

* Classification of Major Depression Disorder patients from healthy controls
* Classification of children with Autism from healthy controls and Organic Mental Disorder children


## Requirements to data for classification grid search pipeline

Pipeline takes preprocessed data. You should provide a directory with

* Return a path-file with at least these columns
    + `fn`, string, filename
    + `target`, string, class of the 
* CSV-files in same directory with the path-file with only necessary channels, without index column

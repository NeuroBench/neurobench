# Motor Prediction for non-human primate

This branch contains the models used for predicting cursor location based on the dataset from: https://zenodo.org/record/3854034#.ZEufx-xBwfj

## Dataset
The following files were used to test our network:
* loco_20170301_05.mat
* loco_20170217_02.mat
* loco_20170210_03.mat
* indy_20170131_02.mat
* indy_20160630_01.mat
* indy_20160407_02.mat

The .mat files originally downloaded from https://zenodo.org/record/3854034#.ZEufx-xBwfj is incompatible with Scipy's loadmat() method, as file's format version is not supported. We have converted the files mentioned above to a loadmat() compatible version, which can be downloaded from:

https://drive.google.com/drive/folders/170i_kEprZ6pwF1RmAwLBdShjqUD6W_8o


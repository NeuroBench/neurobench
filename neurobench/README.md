# Object Detection with Spiking Neural Networks on Automotive Event Data

*This work is supported by the French technological research agency (ANRT) through a CIFRE thesis in collaboration between Renault and Université Côte d'Azur.*

This repository is based on the codes for the paper [Object Detection with Spiking Neural Networks on Automotive Event Data](https://arxiv.org/abs/2205.04339), accepted to the IJCNN 2022, presenting the first SNNs capable of doing object detection on the complex [Prophesee GEN1 event dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/).

Train a DenseNet121-16 on Prophesee NCARS with 5 timesteps and 2 tbins:

    python classification.py -dataset ncars -path path/to/NCARS_dataset -model densenet121-16 -T 5 -tbin 2

To measure test accuracy and sparsity on a pretrained model:

    python object_detection.py -dataset ncars -path path/to/NCARS_dataset -model densenet121-16 -T 5 -tbin 2 -pretrained path/to/pretrained_model -no_train -test

Other parameters are available in `classification.py`.

## Task list

- [x] Convert the code for Prophesee gen 2 mini dataset.
- [ ] Training and validation source code for reproducing Prophesee's RED network results (ongoing with some bug -fixing).
- [ ] Train with full dataset 
- [ ] Integrate [snnmetrics](https://github.com/open-neuromorphic/snnmetrics) for SNN benchmark. (Ongoing)
- [ ] Clean code and marge with the main codes


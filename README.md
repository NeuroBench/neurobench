# algorithms_benchmarks
Benchmark harness and baseline results for the NeuroBench algorithms track.

## 1 Megapixel Event Dataset Object Detection (imec version)

| Contact | Name         | Email               |
|:--------| :------------|:--------------------|
|    All  | Guangzhi Tang|guangzhi.tang@imec.nl|
|    Code | Shenqi Wang  |wang69@imec.be       |

## Task list

- [x] Source code for raw dataset preprocessing.
- [x] Training and validation source code for reproducing Prophesee's RED network results.
- [x] Hybrid Spiking RED with LIF replacing LSTM.
- [ ] Full Spiking RED with full LIF activation. (Ongoing)
- [ ] Integrate [snnmetrics](https://github.com/open-neuromorphic/snnmetrics) for SNN benchmark. (Ongoing)

## Dependencies

### Prophesee 1 Megapixel Event Dataset

Download dataset from [HERE](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/).

The dowloaded zip DAT-format dataset will be 1.2TB and will cost 3.5TB disk space after unzip.

**If the disk space is not big enough, one option is to iteratively dowload each file, perform preprocessing and delete the raw DAT data. This approach can reduce the disk requirement to 1T**

### Prophesee Metavision SDK

We are using Metavision [3.1.2](https://docs.prophesee.ai/3.1.2/index.html).

Install Prophesee Metavision SDK using this [LINK](https://docs.prophesee.ai/3.1.2/installation/linux.html)

### SpikingJelly

We are using SpikingJelly [0.0.0.0.14](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/#index-en) for Spiking LIF implementation.

### Other Dependencies

* Ubuntu 20.04
* Nvidia RTX A6000

## Example Usages

### Data Proprocessing

`algorithms_benchmarks/Neuronbench/utils/data_preprocess.py` is to pre-process .dat files.

The pre-process function is multi_channel_timesurface.


### RED network training and testing

`algorithms_benchmarks/Neuronbench/train_RED/run.py` is to train the original RED model.

The training needs about 25 epochs. The final mAP on the test set is about 0.42.

### Hybrid Spiking RED training and testing

`algorithms_benchmarks/Neuronbench/train_hybrid_spikingRED/run.py` is to train the hybrid spikingRED model.

The training needs about 25 epochs. The final mAP on the test set is about 0.2.

# Training Prophesee's RED model based on Perot et al. 2019
This is a training script mainly based on Prophesee's tutorials, wrapped in a PyTorch Lightning model. It trains a Recurrent Event Detector (RED) model, which is based on a convLSTM Single Shot Detector architecture. To get started quickly, do the following:

1. Download one of the pre-processed datasets from https://docs.prophesee.ai/stable/datasets.html#precomputed-datasets and extract them. The preprocessed histograms are just 86GB in download size! You can run `wget https://dataset.prophesee.ai/index.php/s/1owjuv5jt2CLQRM -O Gen4Hist.zip` for the histograms. 
2. Install Metavision SDK and the related Python dependencies https://docs.prophesee.ai/stable/installation/index.html In order to use their Python bindings, I had to use the system Python and install all packages that are needed in there, rather than in my local conda/pip environment. If you figure out a way to combine the global metavision packages with a local env, please let me know. 
3. Run the training command and adjust batch size and number of time bins in your mini batch to whatever your GPU can fit!

```
python3 train.py --batch_size 16 --num_tbins 12 --accelerator gpu --devices 1 --precision 16 --dataset_path "data/Gen 4 Histograms"
```

All Lightning trainer flags available here: https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags

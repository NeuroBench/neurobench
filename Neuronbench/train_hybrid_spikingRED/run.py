import sys
sys.path.append('/home/shenqi/Master_thesis')
from trainer import Trainer
from Neuronbench.utils.models import  Vanilla_lif
from Neuronbench.utils.dataset import seq_dataloader

from metavision_ml.detection.anchors import Anchors
from metavision_ml.detection.rpn import BoxHead



dataloader = seq_dataloader()
model = Vanilla_lif(cin = dataloader.in_channels, cout = 256, base = 16)
box_coder = Anchors(num_levels=model.levels, anchor_list="PSEE_ANCHORS", variances=[0.1, 0.2])
head = BoxHead(model.cout, box_coder.num_anchors, len(dataloader.wanted_keys) + 1, 0)

trainer = Trainer(model, head, box_coder,dataloader,log_dir='./log')
trainer.fit(epochs=20)


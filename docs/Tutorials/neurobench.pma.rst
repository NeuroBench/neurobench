.. _pma-object-detection-benchmark-tutorial:

=========================================
**PMA Object Detection Benchmark Tutorial**
=========================================

This tutorial aims to provide an insight on how the NeuroBench framework is organized and how you can use it to benchmark your own models!

.. _about-dvs-object-detection:

**About PMA Object Detection**
------------------------------

Real-time object detection is a widely used computer vision task with applications in several domains, including robotics, autonomous driving, and surveillance. Its applications include event cameras for smart home and surveillance systems, drones that monitor and track objects of interest, and self-driving cars that detect obstacles to ensure safe operation. Efficient energy consumption and real-time performance are crucial in such scenarios, particularly when deployed on low-power or always-on edge devices.

.. _dataset:

**Dataset**
------------

The object detection benchmark utilizes the Prophesee 1 Megapixel Automotive Detection Dataset. This dataset was recorded with a high-resolution event camera with a 110-degree field of view mounted on a car windshield. The car was driven in various areas under different daytime weather conditions over several months. The dataset was labeled using the video stream of an additional RGB camera in a semi-automated way, resulting in over 25 million bounding boxes for seven different object classes: pedestrian, two-wheeler, car, truck, bus, traffic sign, and traffic light. The labels are provided at a rate of 60Hz, and the recording of 14.65 hours is split into 11.19, 2.21, and 2.25 hours for training, validation, and testing, respectively.

.. _benchmark-task:

**Benchmark Task**
-------------------

The task of object detection in event-based spatio-temporal data involves identifying bounding boxes of objects belonging to multiple predetermined classes in an event stream. Training for this task is performed offline based on the data splits provided by the original dataset.

.. _code-imports:

**Code Imports**
----------------

First, we will import the relevant libraries. These include the datasets, preprocessors, and accumulators. To ensure your model is compatible with the NeuroBench framework, we will import the wrapper for snnTorch models, which will not change your model. Finally, we import the Benchmark class, which will run the benchmark and calculate your metrics.

.. code:: python

   import torch
   from neurobench.datasets import Gen4DetectionDataLoader
   from neurobench.models import NeuroBenchModel
   from neurobench.benchmarks import Benchmark
   from metavision_ml.detection.anchors import Anchors
   from metavision_ml.detection.rpn import BoxHead

.. _model-imports:

**Model Imports**
------------------

For this tutorial, we will make use of the example architecture that is included in the NeuroBench framework.

.. code:: python

   from obj_det_model import Vanilla, Vanilla_lif

.. _data-loading:

**Data Loading**
----------------

To get started, we will load our desired dataset in a dataloader:

.. code:: python

   # Data loading may take around 7 minutes for loading and approximately 20 minutes for model evaluation and score calculation on a system with an i9-12900KF and an RTX3080.
   test_set_dataloader = Gen4DetectionDataLoader(dataset_path="data/Gen 4 Multi channel",
              split="testing",
              label_map_path="neurobench/datasets/label_map_dictionary.json",
              batch_size=12,
              num_tbins=12,
              preprocess_function_name="multi_channel_timesurface",
              delta_t=50000,
              channels=6,  # multichannel with six channels
              height=360,
              width=640,
              max_incr_per_pixel=5,
              class_selection=["pedestrian", "two-wheeler", "car"],
              num_workers=2)

.. _model-loading:

**Model Loading**
-----------------

For the models we want to benchmark, we need a wrapper:

.. code:: python

   # Evaluation pipeline and models trained by Shenqi Wang (wang69@imec.be) and Guangzhi Tang (guangzhi.tang@imec.nl) at imec.

   class ObjDetectionModel(NeuroBenchModel):
       def __init__(self, net, box_coder, head):
           self.net = net
           self.box_coder = box_coder
           self.head = head

       def __call__(self, batch):
           self.net.eval()
           inputs = batch.permute(1, 0, 2, 3, 4).to(device='cuda')  # dataloader supplies batch, timestep, *; model expects timestep, batch, *
           with torch.no_grad():
               feature = self.net(inputs)
               loc_preds_val, cls_preds_val = self.head(feature)
               scores = self.head.get_scores(cls_preds_val)
               scores = scores.to('cpu')
               for i, feat in enumerate(feature):
                   feature[i] = feature[i].to('cpu')
               inputs = inputs.to('cpu')
               loc_preds_val = loc_preds_val.to('cpu')
               preds = box_coder.decode(feature, inputs, loc_preds_val, scores, batch_size=inputs.shape[1], score_thresh=0.05,
                                        nms_thresh=0.5, max_boxes_per_input=500)
           return preds

       def __net__(self):
           # returns only the network, not the box_coder and head
           return self.net

.. _model-configuration:

**Model Configuration**
-----------------------

Next, we load our model. This example includes two possibilities, a hybrid model that uses artificial neurons and spiking neurons or a fully artificial neural network without spiking neurons.

.. code:: python

   # Loading the model
   mode = "hybrid"  # "ann" or "hybrid"
   if mode == "ann":
       # Baseline ANN RED architecture
       model = Vanilla(cin=6, cout=256, base=16)
       box_coder = Anchors(num_levels=model.levels, anchor_list="PSEE_ANCHORS", variances=[0.1, 0.2])
       head = BoxHead(model.cout, box_coder.num_anchors, 3 + 1, 0)
       model = model.to('cuda')
       head = head to('cuda')
       model.load_state_dict(torch.load('neurobench

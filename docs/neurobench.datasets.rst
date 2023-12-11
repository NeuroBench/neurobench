neurobench.datasets
-------------------

Google Speech Commands
^^^^^^^^^^^^^^^^^^^^^^

The Google Speech Commands dataset (V2) is a commonly used dataset in assessing the performance of keyword spotting algorithms. 
The dataset consists of 105,829 1 second utterances of 35 different words from 2,618 distinct speakers. The data is encoded 
as linear 16-bit, single-channel, pulse code modulated values, at a 16 kHz sampling frequency.

More information about the dataset and the benchmark task in the Google Speech Commands tutorial.

.. automodule:: neurobench.datasets.speech_commands
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:


DVS Gestures
^^^^^^^^^^^^

The IBM Dynamic Vision Sensor (DVS) Gesture dataset is composed of recordings of 29 distinct individuals executing 10 different
types of gestures, including but not limited to clapping, waving, etc. Additionally, an 11th gesture class is included that comprises 
gestures that cannot be categorized within the first 10 classes. The gestures are recorded under four distinct lighting conditions, 
and each gesture is associated with a label that indicates the corresponding lighting condition under which it was performed.

More information about the dataset and the benchmark task in the DVS Gestures tutorial.

.. automodule:: neurobench.datasets.DVSGesture_loader
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:


Prophesee Megapixel Automotive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Prophesee 1 Megapixel Automotive Detection Dataset was recorded with a high-resolution event camera with a 110 degree field
of view mounted on a car windshield. The car was driven in various areas under different daytime weather conditions over several 
months. The dataset was labeled using the video stream of an additional RGB camera in a semi-automated way, resulting in over 25 
million bounding boxes for seven different object classes: pedestrian, two-wheeler, car, truck, bus, traffic sign, and traffic light.
The labels are provided at a rate of 60Hz, and the recording of 14.65 hours is split into 11.19, 2.21, and 2.25 hours for training, 
validation, and testing, respectively.

More information about the dataset and the benchmark task in the Prophesee Megapixel Automotive tutorial.

.. automodule:: neurobench.datasets.megapixel_automotive
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:


Nonhuman Primate Reaching
^^^^^^^^^^^^^^^^^^^^^^^^^

The Nonhuman Primate reaching Dataset consists of multi-channel recordings obtained from the sensorimotor cortex of two non-human primates
(NHP) during self-paced reaching movements towards a grid of targets. The variable x is represented by threshold crossing times
(or spike times) and sorted units for each of the recording channels. The target y is represented by 2-dimensional position coordinates
of the fingertip of the reaching hand, sampled at a frequency of 250 Hz. The complete dataset contains 37 sessions spanning 10 months 
for NHP-1 and 10 sessions from NHP-2 spanning one month. For this study, three sessions from each NHP were selected to include the
entire recording duration, resulting in a total of 6774 seconds of data.

More information about the dataset and the benchmark task in the Nonhuman Primate Reaching tutorial.

.. automodule:: neurobench.datasets.primate_reaching
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:


Mackey-Glass
^^^^^^^^^^^^

The Mackey Glass dataset is synthetic and consists of a one-dimensional non-linear time delay differential equation, where 
the evolution of the signal can be altered by a number of different parameters. These parameters are defined in NeuroBench.

More information about the dataset and the benchmark task in the Mackey-Glass tutorial.

.. automodule:: neurobench.datasets.mackey_glass
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:


Multi-Lingual Spoken Word Corpus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MLCommons Multilingual Spoken Words Corpus is a large and growing audio dataset of spoken words in 50 languages for academic
research and commercial applications in keyword spotting and spoken term search, licensed under CC-BY 4.0. The dataset contains
more than 340,000 keywords, totaling 23.4 million 1-second spoken examples (over 6,000 hours).

The NeuroBench harness does not use the full MSWC dataset. For more information on the subset used, see the NeuroBench paper.

More information about the dataset and the benchmark task in the MSWC tutorial.

.. automodule:: neurobench.datasets.MSWC
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:


Wireless Sensor Data Mining
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The "WISDM Smartphone and Smartwatch Activity and Biometrics Dataset" includes data
collected from 51 subjects, each of whom were asked to perform 18 tasks for 3 minutes
each. Each subject had a smartwatch placed on his/her dominant hand and a smartphone
in their pocket. The data collection was controlled by a custom-made app that ran
on the smartphone and smartwatch. The sensor data that was collected was from the
accelerometer and gyrocope on both the smartphone and smartwatch, yielding four 
total sensors. The sensor data was collected at a rate of 20 Hz (i.e., every 50ms).

More information about the dataset and the benchmark task in the NEHAR tutorial.

.. automodule:: neurobench.datasets.WISDM_loader
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:
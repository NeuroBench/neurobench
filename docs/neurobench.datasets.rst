neurobench.datasets
-------------------

Google Speech Commands
^^^^^^^^^^^^^^^^^^^^^^

The Google Speech Commands dataset (V2) is a commonly used dataset in assessing the performance of keyword spotting algorithms. 
The dataset consists of 105,829 1 second utterances of 35 different words from 2,618 distinct speakers. The data is encoded 
as linear 16-bit, single-channel, pulse code modulated values, at a 16 kHz sampling frequency.

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

.. automodule:: neurobench.datasets.DVSGesture_loader
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:


Prophesee Megapixel Automotive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: neurobench.datasets.megapixel_automotive
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:

Nonhuman Primate Reaching
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: neurobench.datasets.primate_reaching
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:

Mackey-Glass
^^^^^^^^^^^^

The Mackey Glass dataset is a one-dimensional non-linear time delay differential equation, where 
the evolution of the signal can be altered by a number of different parameters. These parameters are defined in NeuroBench.

.. math::

   \\frac{ \\sum_{t=0}^{N}f(t,k) }{N}

.. automodule:: neurobench.datasets.mackey_glass
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:
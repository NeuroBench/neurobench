neurobench.datasets
-------------------

Google Speech Commands
^^^^^^^^^^^^^^^^^^^^^^
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
The task is a classification task. The benchmark task is to use samples from the 23 initial subjects as training and generalize
to samples from the remaining 6 subjects.


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
.. automodule:: neurobench.datasets.mackey_glass
    :special-members: __init__, __getitem__
    :members:
    :undoc-members:
    :show-inheritance:
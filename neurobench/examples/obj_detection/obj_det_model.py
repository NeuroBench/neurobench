from neurobench.models.model import NeuroBenchModel

## Define model ##
class ObjDetectionModel(NeuroBenchModel):
    def __init__(self, net, box_coder, head):
        ...

    def __call__(self, x):
    	...

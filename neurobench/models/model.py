"""
"""

class NeuroBenchModel:
    """
    Abstract class for NeuroBench models. Individual model frameworks are
    responsible for defining model inference. Data should always have timesteps
    as the final dimension.
    """

    def __init__(self):
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __init__")

    def __call__(self):
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __call__")

    def parameters(self):
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement parameters")

    def buffers(self):
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement buffers")
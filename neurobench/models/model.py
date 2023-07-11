"""
"""

class NeuroBenchModel:
    """
    Abstract class for NeuroBench models. Individual model frameworks are
    responsible for defining model inference.
    """

    def __init__(self):
        """
        Init using a trained network.
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __init__")

    def __call__(self):
        """
        Includes the whole pipeline from data to inference (output should be same format as targets).
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __call__")


    # Other functions, e.g. for tracking model size, still under consideration

    def track_run(self):
        """
        Returns dictionary of results from a single benchmark run.
        """
        ...

    def track_batch(self):
        """
        Returns dictionary of results for each test batch.
        """
        ...
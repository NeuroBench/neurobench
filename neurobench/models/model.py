"""
"""

class NeuroBenchModel:
    """
    Abstract class for NeuroBench models. Individual model frameworks are
    responsible for defining model inference.
    """

    def __init__(self):
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __init__")

    def __call__(self):
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement __call__")

    def track_run(self):
        """
        Returns dictionary of results from a single benchmark run.
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement track_run")

    def track_batch(self):
        """
        Returns dictionary of results for each test batch.
        """
        raise NotImplementedError("Subclasses of NeuroBenchModel should implement track_batch")
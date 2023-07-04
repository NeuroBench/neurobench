"""
"""

from .model import NeuroBenchModel

class FSCILModel(NeuroBenchModel):
	"""
	Abstract class for FSCIL models.
	Adds session training functionality to NeuroBenchModel.
	"""

	def __init__(self):
		raise NotImplementedError("Subclasses of FSCILModel should implement __init__")

	def train(self, session, data):
		"""
		Train the model using certain session and data.
		"""
		raise NotImplementedError("Subclasses of FSCILModel should implement train")
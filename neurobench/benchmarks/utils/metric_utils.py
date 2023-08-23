import torch

def check_shape(preds, labels):
	""" Checks that the shape of the predictions and labels are the same.
	"""
	if preds.shape != labels.shape:
		raise ValueError("preds and labels must have the same shape")

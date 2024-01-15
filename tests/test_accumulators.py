import torch

from neurobench.postprocessing import choose_max_count

def test_choose_max_count():
    # Create a tensor of all 0's except for one class
    a = torch.zeros((256, 1000, 35))
    a[:, :, 5] = 1
    assert choose_max_count(a).shape == (256, )
    assert torch.equal(choose_max_count(a), torch.tensor([5] * 256))


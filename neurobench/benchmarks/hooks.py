class SpikeHook():
    """ Hook class for .
    https://www.kaggle.com/code/sironghuang/understanding-pytorch-hooks
    """

    def __init__(self, module):
        """ Initializes the SpikeHook class.

        Args:
            module: A PyTorch nn.Module.
        """
        self.spike_outputs = []
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        """

        Args:
            module:
            input:
            output:
        """
        self.spike_outputs.append(output[0])
    
    def close(self):
        """
        """
        self.hook.remove()
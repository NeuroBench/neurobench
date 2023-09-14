class ActivationHook():
    """ Hook class for an activation layer in a NeuroBenchModel.

    Output of the activation layer in each forward pass will be stored.
    """

    def __init__(self, layer):
        """ Initializes the class.
        
        A forward hook is registered for the activation layer.

        Args:
            layer: The activation layer which is a PyTorch nn.Module.
        """
        self.activation_outputs = []
        self.hook = layer.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, layer, input, output):
        """Hook function that will be called after each forward pass of 
        the activation layer.

        Each output of the activation layer will be stored. 

        Args:
            layer: The registered layer 
            input: Input of the registered layer
            output: Output of the registered layer
        """
        self.activation_outputs.append(output[0])

    def close(self):
        """ Remove the registered hook.
        """
        self.hook.remove()
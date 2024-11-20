from abc import ABC


class LayerHook(ABC):
    def __init__(self, layer) -> None:
        self.layer = layer
        self.inputs = []
        if layer is not None:
            self.hook = layer.register_forward_pre_hook(self.hook_fn)
        else:
            self.hook = None

    def hook_fn(self, module, input):
        self.inputs.append(input)

    def register_hook(self):
        self.hook = self.layer.register_forward_pre_hook(self.hook_fn)

    def reset(self):
        self.inputs.clear()

    def close(self):
        if self.hook:
            self.hook.remove()

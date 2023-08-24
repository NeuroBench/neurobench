"""
Speech2Spikes License
Copyright © 2023 Accenture. 

Speech2Spikes is made available under a proprietary license that permits using, 
copying, sharing, and making derivative works from Speech2Spikes and its source 
code for academics/non-commercial purposes only, as long as the above copyright 
notice and this permission notice are included in all copies of the software.

All distribution of Speech2Spikes in any form (source or executable), including 
any derivative works that you create or to which you contribute, must be under 
the terms of this license. You must inform recipients that any form of 
Speech2Spikes and its derivatives is governed by the terms of this license, and 
how they can obtain a copy of this license and a copy of the source code of 
Speech2Spikes. You may not attempt to alter or restrict the recipients’ rights 
in any form. If you are interested to use Speech2Spikes and/or develop 
derivatives for commercial purposes, licenses can be purchased from Accenture, 
please contact neuromorphic_inquiries@accenture.com for more information.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You agree to indemnify and hold Accenture harmless from and against all 
liabilities, claims and suits and to pay all costs and expenses thereby 
incurred, including reasonable legal fees and court courts, arising out of, 
caused by or in any way connected with your use of Speech2Spikes. 

The original code can be found at:
https://github.com/Accenture/speech2spikes
"""

from .preprocessor import NeuroBenchProcessor

import torch
import torchaudio


def tensor_to_events(batch, threshold=1, device=None):
    """ Converts a batch of continuous signals to binary spikes via delta modulation
    (https://en.wikipedia.org/wiki/Delta_modulation).

    Args:
        batch (Tensor): PyTorch tensor of shape (..., timesteps).
        threshold (float): The difference between the residual and signal that
            will be considered an increase or decrease. Defaults to 1.
        device (torch.device, optional): A torch.Device used by PyTorch for the
            computation. Defaults to None.

    Returns:
        Tensor: A PyTorch int8 tensor of events of shape (..., timesteps).

    TODO:
        Add support for using multiple channels for polarity instead of signs.
    """
    events = torch.zeros(batch.shape)
    levels = torch.round(batch[..., 0])
    if device:
        events = events.to(device)

    for t in range(batch.shape[-1]):
        events[..., t] = (batch[..., t] - levels > threshold).to(torch.int8) - (
            batch[..., t] - levels < -threshold
        ).to(torch.int8)
        levels += events[..., t] * threshold
    return events


class S2SProcessor(NeuroBenchProcessor):
    """ The SpikeEncoder class manages the conversion from raw audio into spikes
    and stores the required conversion parameters.
    """
    def __init__(self, device=None):
        """
        Args:
            device (torch.device, optional): A torch.Device used by PyTorch for the
                computation. Defaults to None.
        """
        self.device = device
        self._default_spec_kwargs = {
            "sample_rate": 16000,
            "n_mels": 20,
            "n_fft": 512,
            "f_min": 20,
            "f_max": 4000,
            "hop_length": 80,
        }
        self.transform = torchaudio.transforms.MelSpectrogram(
            **self._default_spec_kwargs
        )

    def __call__(self, batch):
        """ Converts raw audio data to spikes using Speech2Spikes algorithm
        (https://doi.org/10.1145/3584954.3584995)

        Args:
            batch: A tuple of data and corresponding targets (data_tensor, targets)

        Returns:
            tensors: PyTorch int8 tensor of shape (batch, timesteps, ...)
            targets: A tensor of corresponding targets.

        TODO:
            Add support for cumulative sum of features
        """
        tensors, targets = batch

        # Tensors will be batch, timestep, channels and need to be transposed
        tensors = self.transform(tensors.transpose(1, 2))
        tensors = torch.log(tensors)
        tensors = tensor_to_events(tensors, device=self.device)
        tensors = tensors.transpose(1, 3).squeeze() # Transpose back to timestep last
        return tensors, targets

    def configure(self, threshold=1, **spec_kwargs):
        """ Allows the user to configure parameters of the S2S class and the
        MelSpectrogram transform from torchaudio.

        Go to (https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html)
        for more information on the available transform parameters.

        Args:
            threshold (float): The difference between the residual and signal that
                will be considered an increase or decrease. Defaults to 1.
            **spec_kwargs: Keyword arguments passed to torchaudio's MelSpectrogram.
        """
        self.threshold = threshold

        spec_kwargs = {**self._default_spec_kwargs, **spec_kwargs}
        self.transform = torchaudio.transforms.MelSpectrogram(spec_kwargs)

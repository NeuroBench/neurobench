import torch
from torch.utils.data import DataLoader

from tonic.datasets import DVSGesture as tonic_DVSGesture

# from glob import glob

from neurobench.datasets.dataset import NeuroBenchDataset

import os
import numpy as np
import matplotlib.pyplot as plt

# make animation
from matplotlib.animation import FuncAnimation


class DVSGesture(NeuroBenchDataset):
    """
    Installs DVSGesture Dataset with individual events in each file,
    if not yet installed, else pass the path of the tonic DVSGesture install.

    Data information:
    - Event rate: 1MHz -> dt 1e-6
    - Sample length: 1.7 seconds
    - Default timestep for frames: 5 ms

    For possible preprocessing functions, see:
    https://docs.prophesee.ai/stable/tutorials/ml/data_processing/event_preprocessing.html?highlight=metavision_ml%20preprocessing
    """
    def __init__(
        self, path, split="testing", data_type="frames", preprocessing="stack"
    ):
        """ Initialization will load in data from path if possible, else will download dataset into path. 
        
        Args:
            path (str): Path of DVS Gesture dataset folder if applicable, else the destination of DVS Gesture dataset.
            split (str): Return testing or training data.
            data_type (str): If 'frames', returns frames with preprocessing applied; else returns raw events.
            preprocessing (str): Preprocessing to get frames from raw events.
        """
        # download or load data
        if split == "training":
            self.dataset = tonic_DVSGesture(save_to=path)
        else:
            self.dataset = tonic_DVSGesture(save_to=path, train=False)

        self.filenames = self.dataset.data
        self.path = path
        self.prepr = preprocessing
        self.data_type = data_type

        # sample parameters:
        self._deltat = 5000  # DVS is in microseconds -> deltat = 5ms
        self._T = 1700  # in ms, sample time is 1.7 sec
        self.random_window = False

    def __len__(self):
        """ Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """ Getter method for test data in the DataLoader.

        Args:
            idx (int): Index of the sample.

        Returns:
            sample (tensor): Individual data sample, which can be a sequence of frames or raw data.
            target (tensor): Corresponding gesture label.
        """
        structured_array = self.dataset[idx][0]

        # label = torch.nn.functional.one_hot(torch.tensor(self.dataset[idx][1]), num_classes=11)
        label = torch.tensor(self.dataset[idx][1])

        # get data
        x_data = np.array(structured_array["x"], dtype=np.int16)
        y_data = np.array(structured_array["y"], dtype=np.int16)
        p_data = np.array(structured_array["p"], dtype=bool)
        t_data = np.array(
            structured_array["t"], dtype=np.int64
        )  # time is in microseconds

        xypt = torch.stack(
            (
                torch.tensor(x_data),
                torch.tensor(y_data),
                torch.tensor(p_data),
                torch.tensor(t_data),
            ),
            dim=1,
        )

        # create sample
        t_end = (
            t_data[-1] - self._T * 1000
        )  # find latest time at which we can sample including buffer of factor 1.5 (*1000 to convert to microseconds)
        start_time = np.random.randint(0, t_end) if self.random_window else 0
        sample = xypt[
            (start_time <= xypt[:, 3]) & (xypt[:, 3] <= (start_time + self._T * 1000))
        ]
        sample[:, 3] = sample[:, 3] - sample[0, 3]  # shift timestamps
        tbins = self._T * 1000 // self._deltat
        if self.data_type == "frames":
            # add own preprocessing functions
            if self.prepr == "histo_diff":
                events = histogram_difference_preprocessing(
                    sample,
                    tbins=tbins,
                    delta_t=self._deltat,
                    h_og=128,
                    w_og=128,
                    display_frame=False,
                )
                return events, label

            elif self.prepr == "stack":
                events = stack_preprocessing(
                    sample,
                    delta_t=self._deltat,
                    tbins=tbins,
                    h_og=128,
                    w_og=128,
                    display_frame=False,
                )
                return events, label

        return sample, label

    def set_sample_params(self, delta_t=5, length=1700, random_window=False):
        """
        Sets sample parameters used if frames are created from events.

        Args:
            delta_t (int): Time steps to stack events into frames (in milliseconds).
            length (int): Length in milliseconds of each sample.
            random_window (bool): If True, the sample will be a random time window of length within the gesture.
        """
        self._deltat = delta_t * 1000  # convert to microseconds
        self._T = length
        self.random_window = random_window


def stack_preprocessing(
    xypt, delta_t=5000, tbins=200, h_og=128, w_og=128, channels=3, display_frame=False
):
    """
    Applies stack preprocessing to events. If at least one event has occurred at (x,y) in delta_t corresponding channel 
    (pos or neg) will be 1, else zero.

    Args:
        delta_t (int): Time steps to stack events into frames (in milliseconds).
        tbins (int): Number of frames required.
        h_og (int): Number of pixels in height.
        w_og (int): Number of pixels in width.
        channels (int): Number of channels in each frame (default 3 for plotting purposes).
        display_frame (bool): If True, will create an animation to visualize event frames.
    """
    frames = np.zeros((tbins, channels, h_og, w_og))
    for frame in frames:
        # delete prev neg times
        xypt_new = xypt[xypt[:, 3] >= 0]
        xypt = xypt_new

        # change timestamps
        xypt[:, 3] = xypt[:, 3] - delta_t

        xypt_sub = xypt[xypt[:, 3] <= 0]  # events for the current frame
        pos_pol = np.unique(xypt_sub[xypt_sub[:, 2] == True][:, :2], axis=0)
        neg_pol = np.unique(xypt_sub[xypt_sub[:, 2] == False][:, :2], axis=0)

        frame[0, :, :][pos_pol[:, 0], pos_pol[:, 1]] = 1
        frame[1, :, :][neg_pol[:, 0], neg_pol[:, 1]] = 1

    if display_frame:
        frame = frame.astype(float) / np.max(frame) 

        animation = FuncAnimation(
            fig, update, frames=tbins, fargs=(frames,), interval=delta_t/1000
        )  
        animation.save("test.gif")
        plt.suptitle('Stack preprocessing')
        plt.show()

    return frames


def histogram_difference_preprocessing(xypt, delta_t=5000, tbins=200, h_og=128, w_og=128, channels=3, display_frame=False):
    """
    Applies histogram preprocessing to events. For every positive (pos) or negative (neg) event that has occurred 
    at (x,y) in delta_t, 1 will be added to (x,y) in the corresponding channel (pos or neg).

    Args:
        delta_t (int): Time steps to stack events into frames (in milliseconds).
        tbins (int): Number of frames required.
        h_og (int): Number of pixels in height.
        w_og (int): Number of pixels in width.
        channels (int): Number of channels in each frame (default 3 for plotting purposes).
        display_frame (bool): If True, will create an animation to visualize event frames.
    """
    histogram = np.zeros((tbins, channels, h_og, w_og))
    for frame in histogram:
        # delete prev neg times
        xypt_new = xypt[xypt[:, 3] >= 0]
        xypt = xypt_new

        # change timestamps
        xypt[:, 3] = xypt[:, 3] - delta_t

        xypt_sub = xypt[xypt[:, 3] <= 0]  # events for the current frame
        pos_pol, pos_count = np.unique(xypt_sub[xypt_sub[:, 2] == True][:, :2], axis=0, return_counts=True)
        neg_pol, neg_count = np.unique(xypt_sub[xypt_sub[:, 2] == False][:, :2], axis=0, return_counts=True)
        
        counts_dict = {}

        # Update counts from the positives
        for value, count in zip(pos_pol, pos_count):
            counts_dict[tuple(value)] = counts_dict.get(tuple(value), 0) + count

        # Update counts from the negatives
        for value, count in zip(neg_pol, neg_count):
            counts_dict[tuple(value)] = counts_dict.get(tuple(value), 0) - count

        # Convert the dictionary into a NumPy array
        array_data = [[*key, value] for key, value in counts_dict.items()]
        result_array = np.array(array_data)
        pos_pol = result_array[result_array[:,2]>0]
        neg_pol = result_array[result_array[:,2]<0]
        frame[0, :, :][pos_pol[:, 0], pos_pol[:, 1]] = pos_pol[:,2]
        frame[1, :, :][neg_pol[:, 0], neg_pol[:, 1]] = -neg_pol[:,2] # avoid clipping between [0,1]

    if display_frame:
        frame = frame.astype(float) / np.max(frame) 
        
        animation = FuncAnimation(
            fig, update, frames=tbins, fargs=(histogram,), interval=5
        )  
        animation.save("waving_hand.gif", fps=1 / (5e-3))
        
        plt.suptitle('Histogram difference method')
        plt.show()

    return histogram


fig, ax = plt.subplots()


def update(frame, frames):
    """
    Helper function for animation. 
    """
    ax.clear()
    image = frames[frame].transpose(1, 2, 0)

    ax.imshow(image, cmap="brg")  # You can adjust the colormap as needed
    ax.set_title(f"Frame {frame}")


if __name__ == "__main__":
    path = os.curdir
    dataset = DVSGesture(
        os.path.join(path, "data/dvs_gesture"),
        split="testing", preprocessing="histo_diff"
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for local_batch, local_labels in dataloader:
        print(local_batch[0].shape, local_labels.shape)

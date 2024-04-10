from neurobench.datasets.dataset import Dataset
import numpy as np
import torch
import os
from jitcdde import y, t, jitcdde_lyap
from .utils import download_url
from urllib.error import URLError
import tarfile

"""
The jitcdde package used to generate the MackeyGlass time-series can vary based 
on platform, due to lower level integration solvers. In order to ensure that you
are using the same data as the authors, please use the downloaded version, which
will be automatically downloaded.

https://huggingface.co/datasets/NeuroBench/mackey_glass
"""


class MackeyGlass(Dataset):
    """Dataset for the Mackey-Glass task."""

    def __init__(
        self,
        file_path=None,
        tau=17,
        lyaptime=197,
        constant_past=0.7206597,
        nmg=10,
        beta=0.2,
        gamma=0.1,
        pts_per_lyaptime=75,
        traintime=10.0,
        testtime=10.0,
        start_offset=0.0,
        seed_id=0,
        bin_window=1,
        download=True,
    ):
        """
        Initializes the Mackey-Glass dataset.

        Args:
            file_path (str): path to .npy file containing Mackey-Glass time-series. If this is provided, then tau, lyaptime, constant_past, nmg, beta, gamma are ignored.
            tau (float): parameter of the Mackey-Glass equation
            lyaptime (float): Lyapunov time of the time-series
            constant_past (float): initial condition for the solver
            nmg (float): parameter of the Mackey-Glass equation
            beta (float): parameter of the Mackey-Glass equation
            gamma (float): parameter of the Mackey-Glass equation
            pts_per_lyaptime (int): number of points to sample per one Lyapunov time
            traintime (float): number of Lyapunov times to be used for training a model
            testtime (float): number of Lyapunov times to be used for testing a model
            start_offset (int): added offset in number of points to shift the timeseries forward
            seed_id (int): seed for generating function solution
            bin_window (int): number of points forming lookback window for each prediction
            download (bool): If True, downloads the dataset from the internet and puts it in root
                                 directory. If dataset is already downloaded, it will not be downloaded again.

        """

        super().__init__()

        # Parameters
        self.tau = tau
        self.lyaptime = lyaptime
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.pts_per_lyaptime = pts_per_lyaptime

        # Time units for train (user should split out the warmup or validation)
        self.traintime = traintime * self.lyaptime
        # Time units to forecast
        self.testtime = testtime * self.lyaptime

        self.start_offset = start_offset
        self.seed_id = seed_id

        self.bin_window = bin_window

        # Total time to simulate the system
        self.maxtime = (
            self.traintime + self.testtime + (self.lyaptime / self.pts_per_lyaptime)
        )

        # Discrete-time versions of the continuous times specified above
        self.traintime_pts = round(traintime * self.pts_per_lyaptime)
        self.testtime_pts = round(testtime * self.pts_per_lyaptime)
        self.maxtime_pts = (
            self.traintime_pts + self.testtime_pts + 1
        )  # eval one past the end

        # Specify the system using the provided parameters
        self.mackeyglass_specification = [
            self.beta * y(0, t - self.tau) / (1 + y(0, t - self.tau) ** self.nmg)
            - self.gamma * y(0)
        ]

        self.file_path = file_path
        self.url = "https://huggingface.co/datasets/NeuroBench/mackey_glass/resolve/main/data.tar.gz"

        if download and not os.path.exists(self.file_path):
            print("downloading ....")
            self.download()
        # Load or generate time-series
        if os.path.exists(self.file_path) is not None:
            self.load_data(self.file_path)
        else:
            self.generate_data()

        # Generate train/test indices
        self.split_data()

    def download(self):
        """Download the Mackey Glass data if it doesn't exist already."""

        if os.path.exists(self.file_path):
            print("The dataset already exists!")
            return

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # download file
        file_path = f"{os.path.dirname(self.file_path)}/data.tar.gz"
        try:
            print(f"Downloading {self.url}")
            download_url(self.url, file_path)
        except URLError as error:
            print(f"Failed to download (trying next):\n{error}")
        finally:
            print("Unzipping file...")
            with tarfile.open(file_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile():  # Check if it's a file
                        # Remove the directory path from the member's name
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=os.path.dirname(file_path))
            print()

    def load_data(self, file):
        all_data = np.load(file)

        self.mackeyglass_soln = all_data[
            int(self.start_offset) : int(self.start_offset + self.maxtime_pts)
        ]

        self.mackeyglass_soln = torch.tensor(self.mackeyglass_soln, dtype=torch.float64)
        self.mackeyglass_soln = self.mackeyglass_soln.unsqueeze(dim=-1)

        # pad the soln with preceding zeroes for lookback window
        self.mackeyglass_soln = torch.cat(
            (
                torch.zeros((self.bin_window - 1, 1), dtype=torch.float64),
                self.mackeyglass_soln,
            ),
            0,
        )

    def generate_data(self):
        """Generate time-series using the provided parameters of the equation."""
        np.random.seed(self.seed_id)

        # Create the equation object based on the settings
        self.DDE = jitcdde_lyap(self.mackeyglass_specification)
        # self.DDE.set_integration_parameters(atol=1e-17, rtol=1e-17, min_step=1e-17) # TODO: comment this out later after testing
        self.DDE.constant_past([self.constant_past])
        self.DDE.step_on_discontinuities()

        ##
        # Generate data from the Mackey-Glass system
        ##
        self.mackeyglass_soln = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        lyaps = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        lyaps_weights = torch.zeros((self.maxtime_pts, 1), dtype=torch.float64)
        count = 0

        offset = self.start_offset * self.lyaptime / self.pts_per_lyaptime

        for time in torch.linspace(
            self.DDE.t + offset,
            self.DDE.t + offset + self.maxtime,
            steps=self.maxtime_pts,
            dtype=torch.float64,
        ):
            value, lyap, weight = self.DDE.integrate(time.item())
            self.mackeyglass_soln[count, 0] = value[0]
            lyaps[count, 0] = lyap[0]
            lyaps_weights[count, 0] = weight
            count += 1

        # Total variance of the generated Mackey-Glass time-series
        self.total_var = torch.var(self.mackeyglass_soln[:, 0], True)

        # Estimate Lyapunov exponent
        self.lyap_exp = ((lyaps.T @ lyaps_weights) / lyaps_weights.sum()).item()

        # pad the soln with preceding zeroes for lookback window
        self.mackeyglass_soln = torch.cat(
            (
                torch.zeros((self.bin_window - 1, 1), dtype=torch.float64),
                self.mackeyglass_soln,
            ),
            0,
        )

    def split_data(self):
        """Generate training and testing indices."""
        self.ind_train = torch.arange(0, self.traintime_pts)
        self.ind_test = torch.arange(self.traintime_pts, self.maxtime_pts - 1)

    def __len__(self):
        """
        Returns number of samples in dataset.

        Returns:
            int: number of samples in dataset

        """
        return len(self.mackeyglass_soln) - 1

    def __getitem__(self, idx):
        """
        Getter method for dataset.

        Args:
            idx (int or tensor): index(s) of sample(s) to return

        Returns:
            sample (tensor): individual data sample, shape=(timestamps, features)=(1,1)
            target (tensor): corresponding next state of the system, shape=(label,)=(1,)

        """
        # using Subset with list of indices
        if isinstance(idx, list) or (isinstance(idx, torch.Tensor) and idx.ndim > 0):
            # return in format (batch, bin_window, feature)
            data = [self.mackeyglass_soln[i : i + self.bin_window, :] for i in idx]
            sample = torch.stack(data)

        # idx is an integer
        else:
            sample = self.mackeyglass_soln[idx : idx + self.bin_window, :]

        target = self.mackeyglass_soln[
            idx + self.bin_window, :
        ]  # add to account for pre-padding

        return sample, target

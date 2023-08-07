from . import NeuroBenchProcessor
import torch
from tqdm import tqdm
import numpy as np

class PrimateReachingProcessor(NeuroBenchProcessor):
    def __init__(self, step=0.004, bin_width_time=0.004, spike_sorting=False):
        super(NeuroBenchProcessor).__init__()

        self.step = step
        self.bin_width_time = bin_width_time
        self.spike_sorting = spike_sorting

    def __call__(self, spikes, t, cursor_pos, dataset, d_type):
        no_of_units, no_of_probes = spikes.shape # Dimension is 5*96
        max_len = torch.zeros((1))

        tstart = t[0]
        tend = t[-1]

        # ref = spikes[0, 0]
        # print("Reference Example", dataset[ref].shape)  # This give us a shape of 1*19112
        # Based on this, we should be looking into 
        # print("Actual Reference Example: ", dataset[ref][:])

        # Spikes processing
        if not self.spike_sorting:
            # Sum units for every channel
            for r in range(no_of_probes):
                max_lens = torch.zeros((1))
                for c in range(no_of_units):
                    # max_lens += spikes[:, c][r].shape[0]
                    ref = spikes[c, r]
                    actual_obj = dataset[ref]
                    if actual_obj.shape[-1] == 2:
                        continue
                    # print("probes, units: ", r, c)
                    # print("Actual Obj Dim is: ", actual_obj.shape, actual_obj[:])
                    max_lens += actual_obj.shape[-1]
                if max_len < max_lens:
                    max_len = max_lens

            spike_times = torch.zeros((no_of_probes, 1, int(max_len.item())))

            zero_row = torch.zeros((1))
            for r in range(no_of_probes):
                tmp = []
                for c in range(no_of_units):
                    spikes_inner_shape = dataset[spikes[c, r]].shape[-1]
                    if spikes_inner_shape == 2:
                        continue
                    # spikes_inner_shape = spikes[r][c].shape[0]
                    local_spikes = dataset[spikes[c, r]]
                    for s in range(spikes_inner_shape):
                        tmp.append(local_spikes[:, s])
                tmp.sort()
                if len(tmp):
                    spike_times[r - int(zero_row.item())][0][0: len(tmp)] = torch.Tensor(np.stack(tmp))[:, 0]

        else:
            # No spike_sorting for units
            len_col = torch.zeros((1))
            for r in range(no_of_probes):
                for c in range(no_of_units):
                    # max_lens = spikes[r][c].shape[0]
                    max_lens = dataset[spikes[c, r]].shape[-1]
                    if max_lens <= 2 and np.max(dataset[spikes[c, r]][:]) < tstart:
                        continue
                    if max_lens:
                        len_col += 1
                    if max_len < max_lens:
                        max_len = max_lens
                    

            spike_times = torch.zeros((int(len_col.item()), 1, max_len))

            row_record = torch.zeros((1))
            for r in range(no_of_probes):
                tmp = []
                for c in range(no_of_units):
                    spikes_inner_shape = dataset[spikes[c, r]].shape[-1]
                    if spikes_inner_shape != 2:
                        local_spikes = dataset[spikes[c, r]]
                        spike_times[int(row_record.item())][0][0: spikes_inner_shape] = torch.Tensor(local_spikes[0, :])
                        row_record += 1

        # Extract data--time steps(Num. of spikes)
        print("Start and End Time is: ", tstart, tend)
        time_to_track = np.arange(tstart, tend+self.step, self.step)
        time_to_track = torch.Tensor(time_to_track)
        print("Time to Track: ", time_to_track.shape)
        print("Spike_times Dim is: ", spike_times.shape)

        testfr = torch.zeros((len(time_to_track), len(spike_times)))
        print("Total Time = {}, Channel Num. = {}".format(testfr.shape[0], testfr.shape[1]))

        for j in tqdm(range(len(spike_times))):
            a = spike_times[j][0]
            for i in range(len(time_to_track)):
                bin_edges = [time_to_track[i] - self.bin_width_time, time_to_track[i]]
                inter = a.gt(bin_edges[0]) & a.le(bin_edges[1])
                testfr[i, j] = torch.sum(inter)

        data = torch.swapaxes(testfr, 0, 1)

        ## Extract label
        skip_num = round(cursor_pos.shape[-1] / data.shape[1])
        # print("Skip Num is:", skip_num)

        cursor = torch.tensor(cursor_pos[:, ::skip_num], dtype=d_type)
        cursor_tmp = cursor[:, 1:] - cursor[:, 0:-1]
        # print(cursor.shape, cursor_tmp.shape)
        cursor_start = torch.Tensor(([0], [0]))
        label = torch.cat((cursor_start, cursor_tmp), dim=1)

        return (data, label)
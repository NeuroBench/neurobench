"""
"""
import torch
import numpy as np
from tqdm import tqdm

class PreProcessor():

    @staticmethod
    def preprocessing(spikes, t, cursor_pos, d_type, spike_sorting=False,
                      advance=0.036, bin_width=0.28, Np=None, mode="2D"):
        n, u = spikes.shape
        max_len = torch.zeros((1))

        # Spikes processing
        if not spike_sorting:
            # Sum units for every channel
            for r in range(n):
                max_lens = torch.zeros((1))
                for c in range(u):
                    max_lens += spikes[:, c][r].shape[0]
                if max_len < max_lens:
                    max_len = max_lens

            spike_times = torch.zeros((n, 1, int(max_len.item())))

            zero_row = torch.zeros((1))
            for r in range(n):
                tmp = []
                for c in range(u):
                    spikes_inner_shape = spikes[r][c].shape[0]
                    for s in range(spikes_inner_shape):
                        tmp.append(spikes[r][c][s])
                tmp.sort()
                if len(tmp):
                    spike_times[r - int(zero_row.item())][0][0: len(tmp)] = torch.Tensor(np.stack(tmp))[:, 0]

        else:
            # No spike_sorting for units
            len_col = torch.zeros((1))
            for r in range(n):
                for c in range(u):
                    max_lens = spikes[r][c].shape[0]
                    if max_lens:
                        len_col += 1
                    if max_len < max_lens:
                        max_len = max_lens

            spike_times = torch.zeros((int(len_col.item()), 1, max_len))

            row_record = torch.zeros((1))
            for r in range(n):
                tmp = []
                for c in range(u):
                    spikes_inner_shape = spikes[r][c].shape[0]
                    if spikes_inner_shape:
                        spike_times[int(row_record.item())][0][0: spikes_inner_shape] = torch.Tensor(spikes[r][c][:, 0])
                        row_record += 1

        # Extract data--time steps(Num. of spikes)
        tstart = t[0]
        tend = t[-1]
        time_to_track = np.arange(tstart, tend+advance, advance)
        time_to_track = torch.Tensor(time_to_track)
        print(time_to_track.shape)

        testfr = torch.zeros((len(time_to_track), len(spike_times)))
        print("Total Time = {}, Channel Num. = {}".format(testfr.shape[0], testfr.shape[1]))

        for j in tqdm(range(len(spike_times))):
            a = spike_times[j][0]
            for i in range(len(time_to_track)):
                bin_edges = [time_to_track[i] - bin_width, time_to_track[i]]
                inter = a.gt(bin_edges[0]) & a.le(bin_edges[1])
                testfr[i, j] = torch.sum(inter)

        data = torch.swapaxes(testfr, 0, 1)

        # elif mode == "3D":
        #     testfr_spike = torch.zeros((len(time_to_track), len(spike_times), Np))
        #     print("Total Time = {}, Channel Num. = {}, Time_step = {}".format(testfr_spike.shape[0],
        #                                                                       testfr_spike.shape[1],
        #                                                                       testfr_spike.shape[2]))
        #     st_dur = bin_width / Np
        #     for j in tqdm(range(len(spike_times))):
        #         a = spike_times[j][0]
        #         for i in range(len(time_to_track)):
        #             bin_edges = [time_to_track[i] - bin_width, time_to_track[i]]
        #             inter = a.gt(bin_edges[0]) & a.le(bin_edges[1])
        #             b = a[inter]
        #             if len(b):
        #                 for ts in range(Np):
        #                     st_inter = b.ge((b[0] + ts * st_dur)) & b.le((b[0] + ((ts + 1) * st_dur)))
        #                     testfr_spike[i, j, ts] = torch.sum(st_inter)
        #                     # testfr_spike[i, j, ts] = 1 if torch.sum(st_inter) > 0 else 0

        #     data = torch.swapaxes(testfr_spike, 0, 1)

        ## Extract label
        skip_num = round(cursor_pos.shape[0] / data.shape[1])

        cursor = torch.tensor(cursor_pos[::skip_num, :], dtype=d_type).t()
        cursor_tmp = cursor[:, 1:] - cursor[:, 0:-1]

        cursor_start = torch.Tensor(([0], [0]))
        label = torch.cat((cursor_start, cursor_tmp), dim=1)

        return data, label

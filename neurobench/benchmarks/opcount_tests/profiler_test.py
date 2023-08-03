import torch.nn as nn
import torch

from torch.profiler import profile, record_function, ProfilerActivity

from fvcore.nn import FlopCountAnalysis, flop_count_table

import sys
sys.path.append("../../..")

class TestModel(nn.Module):
    '''
    standard ANN, this works fine
    '''
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=1000, out_features=10)
        self.conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = x.flatten(1)
        out = self.fc(x)

        return out

class TestSNNModel(nn.Module):
    '''
    Default SNNTorch Leaky layer is not supported by the fvcore tool.

        File "/Users/jasonyik/.pyenv/versions/3.8.10/lib/python3.8/site-packages/snntorch/_neurons/leaky.py", line 197, in forward
            self.reset = self.mem_reset(self.mem)
        File "/Users/jasonyik/.pyenv/versions/3.8.10/lib/python3.8/site-packages/snntorch/_neurons/neurons.py", line 106, in mem_reset
            mem_shift = mem - self.threshold
        RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient
    '''

    def __init__(self):
        super().__init__()

        import snntorch as snn
        from snntorch import surrogate

        beta = 0.9
        spike_grad = surrogate.fast_sigmoid()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(256, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(256, 256),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(256, 35),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
        )

    def forward(self, x):
        return self.net(x)

class TestSJModel(nn.Module):
    '''
    Default SJ model also fails fvcore

        File "/Users/jasonyik/.pyenv/versions/3.8.10/lib/python3.8/site-packages/spikingjelly/activation_based/neuron.py", line 239, in single_step_forward
            self.neuronal_charge(x)
        File "/Users/jasonyik/.pyenv/versions/3.8.10/lib/python3.8/site-packages/spikingjelly/activation_based/neuron.py", line 729, in neuronal_charge
            self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
        RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient
    '''
    def __init__(self):
        super().__init__()

        from spikingjelly.activation_based import neuron, layer, surrogate

        self.net = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        return self.net(x)

def test(model, inputs):
    model.eval()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            y = model(inputs)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))

# TODO: models seem to work fine with the profiler but do we want to simply report the flop count generated here?
# other note: the flop count appears to count adds/mults separately, MACs is half of the FLOPs reported

test(TestModel(), torch.randn((1,3,10,10)))
'''
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:294] Completed Stage: Warm Up
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:300] Completed Stage: Collection
------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                          Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  Total KFLOPs  
------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
               model_inference        26.08%       1.082ms        99.59%       4.132ms       4.132ms             1            --  
                  aten::conv2d         4.68%     194.000us        71.90%       2.983ms       2.983ms             1         6.000  
             aten::convolution         4.17%     173.000us        67.22%       2.789ms       2.789ms             1            --  
            aten::_convolution        29.38%       1.219ms        63.05%       2.616ms       2.616ms             1            --  
             aten::thnn_conv2d         3.90%     162.000us        24.54%       1.018ms       1.018ms             1            --  
    aten::_slow_conv2d_forward         8.70%     361.000us        20.63%     856.000us     856.000us             1            --  
                 aten::reshape        11.54%     479.000us        11.57%     480.000us     480.000us             1            --  
       aten::_nnpack_available         9.13%     379.000us         9.13%     379.000us     379.000us             1            --  
                  aten::linear         0.17%       7.000us         1.04%      43.000us      43.000us             1            --  
                   aten::addmm         0.48%      20.000us         0.55%      23.000us      23.000us             1        20.000  
                    aten::relu         0.17%       7.000us         0.43%      18.000us      18.000us             1            --  
                   aten::zeros         0.34%      14.000us         0.41%      17.000us      17.000us             1            --  
                       aten::t         0.14%       6.000us         0.31%      13.000us      13.000us             1            --  
               aten::clamp_min         0.27%      11.000us         0.27%      11.000us      11.000us             1            --  
                   aten::copy_         0.19%       8.000us         0.19%       8.000us       4.000us             2            --  
               aten::transpose         0.12%       5.000us         0.17%       7.000us       7.000us             1            --  
                 aten::flatten         0.05%       2.000us         0.14%       6.000us       6.000us             1            --  
                    aten::view         0.12%       5.000us         0.12%       5.000us       2.500us             2            --  
          aten::_reshape_alias         0.12%       5.000us         0.12%       5.000us       2.500us             2            --  
                 aten::resize_         0.07%       3.000us         0.07%       3.000us       3.000us             1            --  
                   aten::zero_         0.05%       2.000us         0.05%       2.000us       2.000us             1            --  
              aten::as_strided         0.05%       2.000us         0.05%       2.000us       1.000us             2            --  
                  aten::expand         0.05%       2.000us         0.05%       2.000us       2.000us             1            --  
                   aten::empty         0.02%       1.000us         0.02%       1.000us       0.333us             3            --  
                  aten::detach         0.00%       0.000us         0.00%       0.000us       0.000us             1            --  
            aten::resolve_conj         0.00%       0.000us         0.00%       0.000us       0.000us             2            --  
------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.149ms
'''


test(TestSNNModel(), torch.randn((1,20)))
'''
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:294] Completed Stage: Warm Up
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:300] Completed Stage: Collection
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls   Total FLOPs  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
              model_inference        66.05%       1.428ms        99.72%       2.156ms       2.156ms             1            --  
                  FastSigmoid         8.74%     189.000us        19.47%     421.000us      52.625us             8            --  
                     aten::to         0.65%      14.000us         9.07%     196.000us       8.167us            24            --  
               aten::_to_copy         1.02%      22.000us         8.93%     193.000us      12.062us            16            --  
                  aten::copy_         7.86%     170.000us         7.86%     170.000us       7.083us            24            --  
                  aten::clamp         7.49%     162.000us         7.49%     162.000us      40.500us             4            --  
                     aten::gt         1.16%      25.000us         2.45%      53.000us       6.625us             8            --  
                 aten::linear         0.32%       7.000us         2.27%      49.000us      12.250us             4            --  
                  aten::addmm         1.39%      30.000us         1.48%      32.000us       8.000us             4    290304.000  
                    aten::sub         1.34%      29.000us         1.34%      29.000us       2.417us            12            --  
                    aten::mul         0.79%      17.000us         0.79%      17.000us       1.417us            12      1610.000  
             aten::zeros_like         0.14%       3.000us         0.51%      11.000us       2.750us             4            --  
                      aten::t         0.28%       6.000us         0.46%      10.000us       2.500us             4            --  
                    aten::add         0.46%      10.000us         0.46%      10.000us       2.500us             4       803.000  
                  aten::clone         0.42%       9.000us         0.42%       9.000us       2.250us             4            --  
             aten::empty_like         0.32%       7.000us         0.37%       8.000us       2.000us             4            --  
                     aten::eq         0.37%       8.000us         0.37%       8.000us       2.000us             4            --  
                  aten::zeros         0.28%       6.000us         0.28%       6.000us       6.000us             1            --  
             aten::is_nonzero         0.09%       2.000us         0.28%       6.000us       1.500us             4            --  
              aten::transpose         0.19%       4.000us         0.19%       4.000us       1.000us             4            --  
                  aten::alias         0.19%       4.000us         0.19%       4.000us       1.000us             4            --  
          aten::empty_strided         0.19%       4.000us         0.19%       4.000us       0.167us            24            --  
                   aten::item         0.14%       3.000us         0.19%       4.000us       1.000us             4            --  
                 aten::expand         0.05%       1.000us         0.05%       1.000us       0.250us             4            --  
                 aten::detach         0.05%       1.000us         0.05%       1.000us       0.250us             4            --  
    aten::_local_scalar_dense         0.05%       1.000us         0.05%       1.000us       0.250us             4            --  
                  aten::empty         0.00%       0.000us         0.00%       0.000us       0.000us             2            --  
                  aten::zero_         0.00%       0.000us         0.00%       0.000us       0.000us             5            --  
                aten::flatten         0.00%       0.000us         0.00%       0.000us       0.000us             1            --  
             aten::as_strided         0.00%       0.000us         0.00%       0.000us       0.000us             8            --  
           aten::resolve_conj         0.00%       0.000us         0.00%       0.000us       0.000us             8            --  
                       detach         0.00%       0.000us         0.00%       0.000us       0.000us             4            --  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.162ms
'''

test(TestSJModel(), torch.randn((1,28,28)))
'''
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:294] Completed Stage: Warm Up
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:300] Completed Stage: Collection
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls   Total FLOPs  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                        model_inference        92.99%      18.851ms        99.96%      20.264ms      20.264ms             1            --  
    jit_eval_single_step_forward_hard_reset_decay_input         0.19%      38.000us         5.37%       1.089ms       1.089ms             1            --  
                                              aten::neg         2.94%     597.000us         2.94%     597.000us     597.000us             1            --  
                                        aten::full_like         0.03%       6.000us         1.34%     272.000us     272.000us             1            --  
                                            aten::fill_         1.30%     263.000us         1.30%     263.000us     263.000us             1            --  
                                              aten::sub         1.03%     208.000us         1.07%     217.000us     108.500us             2            --  
                                               aten::ge         0.97%     196.000us         0.97%     197.000us     197.000us             1            --  
                                           aten::linear         0.05%      11.000us         0.21%      43.000us      43.000us             1            --  
                                           aten::matmul         0.02%       5.000us         0.10%      21.000us      21.000us             1            --  
                                               aten::to         0.02%       4.000us         0.10%      20.000us       3.333us             6            --  
                                               aten::mm         0.08%      16.000us         0.08%      16.000us      16.000us             1     15680.000  
                                         aten::_to_copy         0.05%      10.000us         0.08%      16.000us       2.667us             6            --  
                                              aten::div         0.06%      12.000us         0.06%      13.000us      13.000us             1            --  
                                                aten::t         0.04%       8.000us         0.05%      11.000us      11.000us             1            --  
                                              aten::add         0.04%       9.000us         0.05%      11.000us       3.667us             3        30.000  
                                              aten::mul         0.04%       9.000us         0.05%      11.000us       5.500us             2        20.000  
                                          aten::flatten         0.03%       7.000us         0.04%       9.000us       9.000us             1            --  
                                            aten::zeros         0.03%       7.000us         0.04%       8.000us       8.000us             1            --  
                                            aten::copy_         0.02%       4.000us         0.02%       4.000us       0.667us             6            --  
                                        aten::transpose         0.01%       2.000us         0.01%       3.000us       3.000us             1            --  
                                       aten::empty_like         0.01%       2.000us         0.01%       3.000us       3.000us             1            --  
                                    aten::empty_strided         0.01%       3.000us         0.01%       3.000us       0.429us             7            --  
                                   aten::_reshape_alias         0.01%       2.000us         0.01%       2.000us       2.000us             1            --  
                                            aten::empty         0.00%       1.000us         0.00%       1.000us       0.500us             2            --  
                                       aten::as_strided         0.00%       1.000us         0.00%       1.000us       1.000us             1            --  
                                            aten::zero_         0.00%       0.000us         0.00%       0.000us       0.000us             1            --  
                                     aten::resolve_conj         0.00%       0.000us         0.00%       0.000us       0.000us             2            --  
                                               defaults         0.00%       0.000us         0.00%       0.000us       0.000us             6            --  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 20.272ms
'''

test(torch.load("../esn/esn.pth"), torch.randn((1,1), dtype=torch.float64))
'''
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:294] Completed Stage: Warm Up
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:300] Completed Stage: Collection
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls   Total FLOPs  
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            model_inference        45.71%     128.000us        96.43%     270.000us     270.000us             1            --  
                               aten::linear        -6.79%     -19.000us        16.43%      46.000us      15.333us             3            --  
                                  aten::cat         6.07%      17.000us        11.79%      33.000us      16.500us             2            --  
                          aten::concatenate         1.07%       3.000us        11.43%      32.000us      32.000us             1            --  
                               aten::matmul        10.00%      28.000us        11.07%      31.000us      10.333us             3            --  
                                   aten::mm        10.36%      29.000us        10.36%      29.000us       9.667us             3     81204.000  
                                  aten::mul         3.93%      11.000us         6.79%      19.000us       6.333us             3       401.000  
                              aten::numpy_T         2.14%       6.000us         5.71%      16.000us       4.000us             4            --  
                               aten::narrow         2.50%       7.000us         4.64%      13.000us       6.500us             2            --  
                              aten::permute         4.29%      12.000us         4.29%      12.000us       3.000us             4            --  
                                 aten::tanh         4.29%      12.000us         4.29%      12.000us      12.000us             1            --  
                                aten::zeros         3.57%      10.000us         3.57%      10.000us      10.000us             1            --  
                                   aten::to         0.36%       1.000us         2.86%       8.000us       8.000us             1            --  
                                    aten::t         2.50%       7.000us         2.86%       8.000us       2.667us             3            --  
                             aten::_to_copy         1.07%       3.000us         2.50%       7.000us       7.000us             1            --  
                                aten::slice         1.79%       5.000us         2.14%       6.000us       3.000us             2            --  
                                  aten::add         2.14%       6.000us         2.14%       6.000us       3.000us             2       400.000  
                                aten::copy_         1.79%       5.000us         1.79%       5.000us       2.500us             2            --  
                                 aten::ones         1.43%       4.000us         1.43%       4.000us       4.000us             1            --  
                                aten::empty         0.36%       1.000us         0.36%       1.000us       0.250us             4            --  
                        aten::empty_strided         0.36%       1.000us         0.36%       1.000us       1.000us             1            --  
                           aten::as_strided         0.36%       1.000us         0.36%       1.000us       0.111us             9            --  
                           aten::empty_like         0.36%       1.000us         0.36%       1.000us       1.000us             1            --  
                            aten::transpose         0.36%       1.000us         0.36%       1.000us       0.333us             3            --  
                                aten::zero_         0.00%       0.000us         0.00%       0.000us       0.000us             1            --  
                                aten::fill_         0.00%       0.000us         0.00%       0.000us       0.000us             1            --  
                         aten::resolve_conj         0.00%       0.000us         0.00%       0.000us       0.000us             5            --  
    aten::_has_compatible_shallow_copy_type         0.00%       0.000us         0.00%       0.000us       0.000us             1            --  
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 280.000us
'''

test(torch.load("../lstm/lstm.pth"), torch.randn((1,1), dtype=torch.float64))
'''
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:294] Completed Stage: Warm Up
STAGE:2023-08-03 16:52:33 14067:2948126 ActivityProfilerController.cpp:300] Completed Stage: Collection
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls   Total FLOPs  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
              model_inference        30.43%      91.000us        99.00%     296.000us     296.000us             1            --  
                   aten::lstm        18.39%      55.000us        61.20%     183.000us     183.000us             1            --  
                 aten::linear         4.68%      14.000us        18.06%      54.000us      10.800us             5            --  
                  aten::addmm         8.70%      26.000us         9.36%      28.000us       5.600us             5     60500.000  
           aten::unsafe_chunk         0.67%       2.000us         5.35%      16.000us       8.000us             2            --  
                 aten::unbind         4.35%      13.000us         5.02%      15.000us       3.750us             4            --  
           aten::unsafe_split         4.01%      12.000us         4.68%      14.000us       7.000us             2            --  
                  aten::stack         4.01%      12.000us         4.35%      13.000us       3.250us             4            --  
               aten::sigmoid_         4.01%      12.000us         4.01%      12.000us       2.000us             6            --  
                   aten::add_         3.34%      10.000us         3.34%      10.000us       2.500us             4            --  
              aten::transpose         2.01%       6.000us         2.01%       6.000us       0.857us             7            --  
                   aten::view         2.01%       6.000us         2.01%       6.000us       0.750us             8            --  
                      aten::t         2.01%       6.000us         2.01%       6.000us       1.200us             5            --  
                aten::squeeze         1.67%       5.000us         1.67%       5.000us       1.667us             3            --  
                   aten::relu         1.00%       3.000us         1.67%       5.000us       5.000us             1            --  
                  aten::zeros         1.34%       4.000us         1.34%       4.000us       1.333us             3            --  
              aten::unsqueeze         1.34%       4.000us         1.34%       4.000us       4.000us             1            --  
                    aten::mul         1.34%       4.000us         1.34%       4.000us       0.667us             6       300.000  
                  aten::tanh_         1.00%       3.000us         1.00%       3.000us       1.500us             2            --  
                 aten::select         0.67%       2.000us         0.67%       2.000us       0.333us             6            --  
                 aten::narrow         0.67%       2.000us         0.67%       2.000us       0.250us             8            --  
                   aten::tanh         0.67%       2.000us         0.67%       2.000us       1.000us             2            --  
              aten::clamp_min         0.67%       2.000us         0.67%       2.000us       2.000us             1            --  
                 aten::expand         0.33%       1.000us         0.33%       1.000us       0.200us             5            --  
                  aten::copy_         0.33%       1.000us         0.33%       1.000us       0.200us             5            --  
                    aten::cat         0.33%       1.000us         0.33%       1.000us       0.250us             4            --  
                  aten::empty         0.00%       0.000us         0.00%       0.000us       0.000us             4            --  
                  aten::zero_         0.00%       0.000us         0.00%       0.000us       0.000us             3            --  
             aten::as_strided         0.00%       0.000us         0.00%       0.000us       0.000us            30            --  
    aten::cudnn_is_acceptable         0.00%       0.000us         0.00%       0.000us       0.000us             1            --  
           aten::resolve_conj         0.00%       0.000us         0.00%       0.000us       0.000us            10            --  
                  aten::slice         0.00%       0.000us         0.00%       0.000us       0.000us             8            --  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 299.000us
'''
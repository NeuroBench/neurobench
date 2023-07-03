from neurobench.datasets.primate_reaching import PrimateReaching
from neurobench.models import ANNModel, LSTMModel
from neurobench.benchmarks.benchmark_ann import BenchmarkANN
from neurobench.benchmarks.benchmark_lstm import BenchmarkLSTM
import yaml
import torch
from neurobench import utils

if __name__ == "__main__":
    study_name = "benchmark"

    with open('./neurobench/benchmarks/hyperparams.yaml') as f:
        hyperparams = yaml.load(f, Loader=yaml.loader.SafeLoader)

    if torch.cuda.is_available():
        print("using cuda")
        hyperparams['device'] = torch.device("cuda")
    else:
        print("using cpu")
        hyperparams['device'] = torch.device("cpu")

    MODEL_TYPE = hyperparams["model_type"]

    for file_no in range(1):
        if hyperparams['filename'].split('_')[0] == "indy":
            input_dim = hyperparams['ANN_input_dimension'][0]
        else:
            input_dim = hyperparams['ANN_input_dimension'][1]
            
        train_ratio_num = hyperparams['Train_data_ratio']
        print('Filename: {}, Time Shift: {}, Train Ratio: {}'.format(hyperparams['filename'], hyperparams['delay'], train_ratio_num))
        final_results = torch.zeros((hyperparams['max_run'], 6), device=hyperparams['device'])
                
        advance_time = 0.004
        biological_delay = int(hyperparams['delay'] / advance_time)
        ds = PrimateReaching(hyperparams, path=hyperparams['dataset_file'], filename=hyperparams['filename'],
                            postpr_data_path=hyperparams['postpr_data_path'], regenerate=hyperparams['regenerate'],
                            biological_delay=biological_delay, spike_sorting=False, Np=hyperparams['num_steps'],
                            mode="3D", advance=advance_time, bin_width=advance_time*50, train_ratio=train_ratio_num)

        for run_num in range(hyperparams['max_run']):
            torch.manual_seed(hyperparams['seed'][run_num])

            layer1_num, layer2_num = 32, 48
            net = ANNModel(hyperparams, input_dim=input_dim * hyperparams['num_steps'], layer1=layer1_num, layer2=layer2_num, layer3=32, output_dim=2, dropout_rate=0.5)
            # net = LSTMModel(input_dim, 16, 2, hyperparams['batch_size'], hyperparams['num_steps'])

            if MODEL_TYPE == "ANN":
                benchmark = BenchmarkANN(dataset=ds, net=net, hyperparams=hyperparams, model_type=MODEL_TYPE, 
                                        train_ratio=train_ratio_num, delay=hyperparams['delay'])
            if MODEL_TYPE == "LSTM":
                benchmark = BenchmarkLSTM(dataset=ds, net=net, hyperparams=hyperparams, model_type=MODEL_TYPE, 
                                        train_ratio=train_ratio_num, delay=hyperparams['delay'])

            final_results = benchmark.run(run_num, final_results)

        if hyperparams['save_results']:
            final_mean = torch.mean(final_results, dim=0)
            print(final_mean)
            utils.save_results(final_mean, train_ratio_num, hyperparams['delay'], hyperparams['filename'], 
                                hyperparams['results_save_path'], layer1=hyperparams['layer1_neuron'], layer2=hyperparams['layer2_neuron'])




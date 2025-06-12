import torch
import torch.nn as nn

class ANNModel(nn.Module):
    def __init__(
        self, 
        input_dim, 
        layer1=16, 
        layer2=32, 
        output_dim=2,
        bin_window=0.2, 
        sampling_rate=0.004, 
        drop_rate=0.5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = layer1
        self.layer2 = layer2
        self.drop_rate = drop_rate

        self.bin_window_time = bin_window
        self.sampling_rate = sampling_rate
        self.bin_window_size = int(self.bin_window_time / self.sampling_rate)

        self.fc1 = nn.Linear(self.input_dim, self.layer1)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.fc3 = nn.Linear(self.layer2, self.output_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2)
        self.activation = nn.ReLU()
        self.batch_size = 1
        self.weight_update = False
        self.lr = 5e-3

        self.register_buffer("data_buffer", torch.zeros(1, 1, input_dim).type(torch.float32), persistent=False)

    def relu_derivative(self, x):
        return (x > 0).float()

    def update(self, input_, y1, y1_activated, y2, y2_activated, pred, label_):
        batch_num = input_.shape[0]
        ### update 3rd layer
        delta3 = label_ - pred
        # print("delta3: ", delta3)
        dW = self.lr*(y2.t() @ delta3)/batch_num
        db3 = self.lr*delta3.sum(dim=0, keepdim=True)/batch_num

        ### update 2nd layer
        delta2 = (delta3 @ self.fc3.weight.data) * self.relu_derivative(y2_activated)
        dV = self.lr*(y1.t() @ delta2)/batch_num
        db2 = self.lr*delta2.sum(dim=0, keepdim=True)/batch_num
       
        ### update 1st layer
        delta1 = (delta2 @ self.fc2.weight.data) * self.relu_derivative(y1_activated)
        dU = self.lr*(input_.t() @ delta1)/batch_num
        db1 = self.lr*delta1.sum(dim=0, keepdim=True)/batch_num
       
        self.fc3.weight.data = self.fc3.weight.data + dW.t()
        self.fc2.weight.data = self.fc2.weight.data + dV.t()
        self.fc1.weight.data = self.fc1.weight.data + dU.t()
       
        self.fc3.bias.data = self.fc3.bias.data + db3.squeeze(dim=0)
        self.fc2.bias.data = self.fc2.bias.data + db2.squeeze(dim=0)
        self.fc1.bias.data = self.fc1.bias.data + db1.squeeze(dim=0)
    
    def forward(self, x, y):
        self.data_buffer = torch.cat((self.data_buffer, x), dim=0)
        self.data_buffer = self.data_buffer[1:, :, :]

        if self.weight_update:
            y1 = self.fc1(x.view(self.batch_size, -1))
            y1_activated = self.activation(y1)
            y2 = self.fc2(y1_activated)
            y2_activated = self.activation(y2)
            pred = self.fc3(y2_activated)
            self.update(input_=x.view(self.batch_size, -1), y1=y1, y1_activated=y1_activated, y2=y2, y2_activated=y2_activated,  pred=pred, label_=y)
           
            loss = nn.MSELoss()(pred, y)
            pred = pred.squeeze(dim=0)
        else:
            x = self.activation(self.fc1(x.view(self.batch_size, -1)))
            x = self.activation(self.dropout(self.fc2(x)))
            x = self.fc3(x)
    
            pred = x.squeeze(dim=0)

        return pred

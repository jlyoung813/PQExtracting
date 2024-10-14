import torch
from torch import nn


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, input_dim2, linear_dim=16384, device="cpu"):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, bidirectional=True, num_layers=1)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = nn.Linear(in_features=linear_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.output = nn.Sigmoid()
        self.input_dim2 = input_dim2
        self.relu = nn.ReLU()
        self.device = device
        filter_x = [[47, 0, -47], [162, 0, -162], [47, 0, -47]]
        filter_y = [[47, 162, 47], [0, 0, 0], [-47, -162, -47]]
        self.kernel_x = torch.FloatTensor(filter_x).unsqueeze(0).unsqueeze(0).to(self.device)
        self.kernel_y = torch.FloatTensor(filter_y).unsqueeze(0).unsqueeze(0).to(self.device)

    def forward(self, input):
        h0 = torch.zeros(2, self.input_dim2, 128)
        h0 = h0.to(self.device)
        c0 = torch.zeros(2, self.input_dim2, 128)
        c0 = c0.to(self.device)
        G_x = torch.nn.functional.conv2d(input, self.kernel_x, stride=1, padding=1)
        G_y = torch.nn.functional.conv2d(input, self.kernel_y, stride=1, padding=1)
        x = torch.sqrt(torch.pow(G_y, 2) + torch.pow(G_x, 2))
        x = torch.nn.functional.normalize(x)
        x = self.lstm(x, (h0, c0))[0]
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        output = self.output(logits)
        return output

class CNNLSTMNetwork(nn.Module):

    def __init__(self, input_size, linear_dim, input_dim2=9, device="cpu"):
        super().__init__()
        self.input_dim2 = input_dim2
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, bidirectional=True, num_layers=1)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = nn.Linear(in_features=linear_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.output = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        h0 = torch.zeros(2, self.input_dim2, 128)
        h0 = h0.to(self.device)
        c0 = torch.zeros(2, self.input_dim2, 128)
        c0 = c0.to(self.device)
        x = self.lstm(x, (h0, c0))[0]
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        output = self.output(logits)
        return output

class CNNNetwork(nn.Module):
    def __init__(self, linear_dim, device="CPU", use_grads=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = nn.Linear(in_features=linear_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.output = nn.Sigmoid()
        self.device = device
        self.use_grads = use_grads
        filter_x = [[47, 0, -47], [162, 0, -162], [47, 0, -47]]
        filter_y = [[47, 162, 47], [0, 0, 0], [-47, -162, -47]]
        self.kernel_x = torch.FloatTensor(filter_x).unsqueeze(0).unsqueeze(0).to(self.device)
        self.kernel_y = torch.FloatTensor(filter_y).unsqueeze(0).unsqueeze(0).to(self.device)

    def forward(self, input_data):
        x = input_data
        if self.use_grads:
            G_x = torch.nn.functional.conv2d(input_data, self.kernel_x, stride=1, padding=1)
            G_y = torch.nn.functional.conv2d(input_data, self.kernel_y, stride=1, padding=1)
            x = G_y + G_x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        output = self.output(logits)
        return output

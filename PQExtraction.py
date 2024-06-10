import torch
from torch import nn
import torchaudio
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class CNNNetwork(nn.Module):

    def __init__(self, linear_dim):
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

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        output = self.output(logits)
        return output


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, linear_dim=8192):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, bidirectional=True)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = nn.Linear(in_features=linear_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.output = nn.Sigmoid()

    def forward(self, input):
        h0 = torch.zeros(2, 32, 128)
        h0 = h0.to(device)
        c0 = torch.zeros(2, 32, 128)
        c0 = c0.to(device)
        x = self.lstm(input, (h0, c0))[0]
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        output = self.output(logits)
        return output


class CNNLSTMNetwork(nn.Module):

    def __init__(self, input_size, linear_dim):
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
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, bidirectional=True)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = nn.Linear(in_features=linear_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.output = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        h0 = torch.zeros(2, 3, 128)
        h0 = h0.to(device)
        c0 = torch.zeros(2, 3, 128)
        c0 = c0.to(device)
        x = self.lstm(x, (h0, c0))[0]
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        output = self.output(logits)
        return output

path = 'Audio Files'
postfix = 'ENSS.wav'
SAMPLE_RATE = 16000
NUM_SAMPLES = 392000
name_set = []

for file in os.listdir(path):
    name_set.append(file[:len(file)-8])


data_file = open('PQDB.json', 'r')
labels = json.loads(data_file.read())

data =[[labels[name]["CAPE-V Roughness"],labels[name]["CAPE-V Breathiness"],labels[name]["CAPE-V Strain"],labels[name]["CAPE-V Pitch"],labels[name]["CAPE-V Loudness"]] for name in name_set]
data_np = np.array(data, dtype="float")

data_np = data_np/data_np.max(axis=0)


def load_item(idx):
    waveform, sr = torchaudio.load(os.path.join(path, name_set[idx] + postfix))
    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=32, normalized=False)
    waveform = resampler(waveform)
    waveform = torch.mean(waveform,dim=0,keepdim=True)
    if waveform.shape[1] > NUM_SAMPLES:
        waveform = waveform[:, :NUM_SAMPLES]
    if waveform.shape[1] < NUM_SAMPLES:
        num_padding = NUM_SAMPLES - waveform.shape[1]
        last_dim_padding = (0, num_padding)
        waveform = torch.nn.functional.pad(waveform, last_dim_padding)
    spectrogram = mel_spec(waveform)
    return spectrogram


def split_data(fts, lbls, factor=5):
    x = []
    y = []
    for i in range(len(fts)):
        feat = torch.split(fts[i], fts[i].size()[2] // factor, dim=2)
        for j in range(factor):
            x.append(feat[j])
            y.append(lbls[i])
    return x, y


def train(feats, ls, mod, loss_func, adj, target, train_size, output=[]):
    idxs = [i for i in range(train_size)]
    for j in range(50):
        random.shuffle(idxs)
        epoch_loss = 0
        for idx in idxs:
            spec = feats[idx]
            spec = spec.to(device)
            label = torch.tensor([ls[idx][target]])
            label = label.to(device)
            logits = mod(spec)
            loss = loss_func(logits.float(), label.float())
            epoch_loss += loss.item()
            adj.zero_grad()
            loss.backward()
            adj.step()
        output.append(epoch_loss)


features = [load_item(i) for i in range(len(name_set))]
x_split, y_split = split_data(features, data_np, factor=10)
x = features
y = data_np

test_size = len(x)//10
train_size = len(x) - test_size

test_size_split = len(x_split)//10
train_size_split = len(x_split) - test_size_split

feat_names = ["Roughness", "Breathiness", "Strain", "Pitch", "Loudness"]
print("CNNLSTMNetwork-No Split:")
for j in range(5):
    model = CNNLSTMNetwork(input_size=49, linear_dim=98304).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters())
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    plt.plot([i for i in range(len(train_loss))], train_loss)
    losses = []
    for i in range(train_size, len(x)):
        spec2 = x[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y[i][j]])
        label = label.to(device)
        logits = model(spec2)
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j] + ":", sum(losses)/len(losses))
    torch.save(model.state_dict(), f"CNNLSTM_{feat_names[j]}.pt")
"""
plt.legend(feat_names)
plt.title("Per Epoch Loss Graph of CNNLSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()"""

"""print("CNNLSTMNetwork-Factor 5 Split:")
for j in range(5):
    model = CNNLSTMNetwork(input_size=6, linear_dim=98304).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters())
    train_loss = []
    train(x_split, y_split, model, torch.nn.MSELoss(), adjuster, j, train_size_split, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
    losses = []
    for i in range(train_size_split, len(x_split)):
        spec2 = x_split[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y_split[i][j]])
        label = label.to(device)
        logits = model(spec2)
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j] + ":", sum(losses)/len(losses))
    torch.save(model.state_dict(), f"CNN_{feat_names[j]}_split.pt")"""

print("LSTMNetwork-No Split:")
for j in range(5):
    model = LSTMNetwork(input_size=766, linear_dim=8192).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters())
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
    losses = []
    for i in range(train_size, len(x)):
        spec2 = x[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y[i][j]])
        label = label.to(device)
        logits = model(spec2)
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j] + ":", sum(losses)/len(losses))
    torch.save(model.state_dict(), f"LSTM_{feat_names[j]}.pt")
plt.legend(feat_names)
plt.title("Per Epoch Loss Graph of CNNLSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

"""print("LSTMNetwork-Factor 5 Split:")
for j in range(5):
    model = LSTMNetwork(input_size=76, linear_dim=8192).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters())
    train_loss = []
    train(x_split, y_split, model, torch.nn.MSELoss(), adjuster, j, train_size_split, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
    losses = []
    for i in range(train_size_split, len(x_split)):
        spec2 = x_split[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y_split[i][j]])
        label = label.to(device)
        logits = model(spec2)
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j] + ":", sum(losses)/len(losses))
    torch.save(model.state_dict(), f"LSTM_{feat_names[j]}_split.pt")"""

print("CNNNetwork-No Split:")
for j in range(5):
    model = CNNNetwork(linear_dim=18816).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters())
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
    losses = []
    for i in range(train_size, len(x)):
        spec2 = x[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y[i][j]])
        label = label.to(device)
        logits = model(spec2)
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j] + ":", sum(losses)/len(losses))
    torch.save(model.state_dict(), f"CNN_{feat_names[j]}.pt")

"""print("CNNNetwork-Factor 5 Split:")
for j in range(5):
    model = CNNNetwork(linear_dim=3520).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters())
    train_loss = []
    train(x_split, y_split, model, torch.nn.MSELoss(), adjuster, j, train_size_split, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
    losses = []
    for i in range(train_size_split, len(x_split)):
        spec2 = x_split[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y_split[i][j]])
        label = label.to(device)
        logits = model(spec2)
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j] + ":", sum(losses)/len(losses))
    torch.save(model.state_dict(), f"CNN_{feat_names[j]}_split.pt")"""

"""plt.legend(feat_names)
plt.title("Per Epoch Loss Graph of CNNNetwork-No Split")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()"""




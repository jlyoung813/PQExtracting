import sklearn.metrics
import torch
import PQExtractor
from PQExtractor import LSTMNetwork
from PQExtractor import CNNNetwork
from PQExtractor import CNNLSTMNetwork
import torchaudio
import numpy as np
import os
import json
import random

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("cuda")

path = 'Audio Files'
postfix = 'ENSS.wav'
SAMPLE_RATE = 16000
NUM_SAMPLES = 320000
name_set = []

for file in os.listdir(path):
    name_set.append(file[:len(file)-8])
    random.seed(0)
random.shuffle(name_set)
data_file = open('PQDB.json', 'r')
labels = json.loads(data_file.read())
data = []
names = []
for name in name_set:
    if name in labels.keys():
        data.append([labels[name]["Roughness"],labels[name]["Breathiness"],labels[name]["Strain"], labels[name]["Pitch"],
                labels[name]["Loudness"],labels[name]["Resonance"], labels[name]["Weight"]])
        names.append(name)
data_np = np.array(data, dtype="float")

data_np = data_np/data_np.max(axis=0)


def load_item(idx, factor=10):
    waveform, sr = torchaudio.load(os.path.join(path, names[idx] + postfix))
    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    waveform = resampler(waveform)
    waveform = torch.mean(waveform,dim=0,keepdim=True)
    feat = torch.split(waveform, NUM_SAMPLES // factor, dim=1)
    return feat


def spectrogram(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128,
                                                    normalized=False)
    spec = mel_spec(waveform)
    return spec


def split_data(fts, lbls, factor=10):
    x = []
    y = []
    for i in range(len(fts)):
        feat = fts[i]
        for item in feat:
            if item.shape[1] == NUM_SAMPLES // factor:
                x.append(item)
                y.append(lbls[i])
    return x, y


def train(feats, ls, mod, loss_func, adj, target, train_size, output=[], num_epochs=10):
    idxs = [i for i in range(train_size)]
    for j in range(num_epochs):
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


waveforms = [load_item(i) for i in range(len(names))]
waveforms_split, data_split = split_data(waveforms, data_np)
data_idxs = [i for i in range(len(waveforms_split))]
random.shuffle(data_idxs)
features = [spectrogram(waveforms_split[i]) for i in data_idxs]

test_size = len(features)//10
train_size = len(features) - test_size
y = [data_split[i] for i in data_idxs]
feat_names = ["Roughness", "Breathiness", "Strain", "Pitch", "Loudness", "Resonance", "Weight"]

print("LSTMNetwork-Spectrogram Grad:")
for j in range(7):
    x = features
    input_size = x[0].shape[2]
    model = LSTMNetwork(input_size=input_size, input_dim2=x[0].shape[1], linear_dim=x[0].shape[1]*128*2, device=device).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    losses = []
    y_true = []
    y_pred = []
    mape = []
    for i in range(train_size, len(x)):
        spec2 = x[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y[i][j]])
        label = label.to(device)
        logits = model(spec2)
        y_true.append(round(label.item(), 2))
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j])
    print("Mean MSE:", sum(losses)/len(losses))
    print("True mean of test data:", sum(y_true)/len(y_true))
    print("Mean of predicted values:", sum(y_pred)/len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    torch.save(model.state_dict(), f"models/LSTM/LSTM_{feat_names[j]}_grad.pt")

print("CNNNetwork-Spectrograms:")
for j in range(7):
    x = features
    input_size = x[0].shape[2]
    model = CNNNetwork(linear_dim=5760, device=device).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    losses = []
    y_true = []
    y_pred = []
    mape = []
    for i in range(train_size, len(x)):
        spec2 = x[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y[i][j]])
        label = label.to(device)
        logits = model(spec2)
        y_true.append(round(label.item(), 2))
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j])
    print("Mean MSE:", sum(losses)/len(losses))
    print("True mean of test data:", sum(y_true)/len(y_true))
    print("Mean of predicted values:", sum(y_pred)/len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    torch.save(model.state_dict(), f"models/CNN/CNN_{feat_names[j]}_spec.pt")

print("CNNNetwork-Spectrogram Grads:")
for j in range(7):
    x = features
    input_size = x[0].shape[2]
    model = CNNNetwork(linear_dim=5760, device=device, use_grads=True).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    losses = []
    y_true = []
    y_pred = []
    mape = []
    for i in range(train_size, len(x)):
        spec2 = x[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y[i][j]])
        label = label.to(device)
        logits = model(spec2)
        y_true.append(round(label.item(), 2))
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j])
    print("Mean MSE:", sum(losses)/len(losses))
    print("True mean of test data:", sum(y_true)/len(y_true))
    print("Mean of predicted values:", sum(y_pred)/len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    torch.save(model.state_dict(), f"models/CNN/LSTM_{feat_names[j]}_grad.pt")

for j in range(0):
    x = features
    input_size = x[0].shape[2]
    model = CNNLSTMNetwork(input_size=6, linear_dim=294912, device=device).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    losses = []
    y_true = []
    y_pred = []
    mape = []
    for i in range(train_size, len(x)):
        spec2 = x[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y[i][j]])
        label = label.to(device)
        logits = model(spec2)
        y_true.append(round(label.item(), 2))
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())

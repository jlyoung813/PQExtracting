import sklearn.metrics
import torch
import torchvision.transforms
from torch import nn
import torchaudio
import numpy as np
import os
import json
import random
import csv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("cuda")

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
    def __init__(self, input_size, input_dim2, linear_dim=16384):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, bidirectional=True, num_layers=1)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-1)
        self.linear1 = nn.Linear(in_features=linear_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.output = nn.Sigmoid()
        self.input_dim2 = input_dim2
        self.relu = nn.ReLU()
        #sigma = torch.nn.Parameter(torch.ones(1))
        #self.tau = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input):
        h0 = torch.zeros(2, self.input_dim2, 128)
        h0 = h0.to(device)
        c0 = torch.zeros(2, self.input_dim2, 128)
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


def load_item(idx):
    waveform, sr = torchaudio.load(os.path.join(path, names[idx] + postfix))
    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    waveform = resampler(waveform)
    waveform = torch.mean(waveform,dim=0,keepdim=True)
    if waveform.shape[1] > NUM_SAMPLES:
        waveform = waveform[:, :NUM_SAMPLES]
    if waveform.shape[1] < NUM_SAMPLES:
        num_padding = NUM_SAMPLES - waveform.shape[1]
        last_dim_padding = (0, num_padding)
        waveform = torch.nn.functional.pad(waveform, last_dim_padding)
    return waveform

def spectrogram(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128,
                                                    normalized=True)
    spectrogram = mel_spec(waveform)
    return spectrogram

def edge_filter(spectrogram, mode='scharr'):
    filter_x = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    filter_y = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    if mode == 'sobel':
        filter_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        filter_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    if mode == 'scharr':
        filter_x = [[47, 0, -47], [162, 0, -162], [47, 0, -47]]
        filter_y = [[47, 162, 47], [0, 0, 0], [-47, -162, -47]]
    kernel_x = torch.FloatTensor(filter_x).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.FloatTensor(filter_y).unsqueeze(0).unsqueeze(0)
    G_x = torch.nn.functional.conv2d(spectrogram, kernel_x, stride=1, padding=1)
    G_y = torch.nn.functional.conv2d(spectrogram, kernel_y, stride=1, padding=1)
    Grad = torch.sqrt(torch.pow(G_y, 2) + torch.pow(G_x, 2))
    return torch.nn.functional.normalize(Grad)

def y_filter(spectrogram):
    filter_y = [[47, 162, 47], [0, 0, 0], [-47, -162, -47]]
    kernel_y = torch.FloatTensor(filter_y).unsqueeze(0).unsqueeze(0)
    G_y = torch.nn.functional.conv2d(spectrogram, kernel_y, stride=1, padding=1)
    return torch.nn.functional.normalize(G_y)

def DoG(spectrogram):
    Blur = torchvision.transforms.GaussianBlur(3, sigma=1.4)
    Blur2 = torchvision.transforms.GaussianBlur(3, sigma=0.5)
    blur = Blur(spectrogram)
    blur2 = Blur2(spectrogram)
    DoG = 1.5 * blur2 - 0.5 * blur
    relu = torch.nn.ReLU()
    DoG = (DoG > 0.1).float()
    return DoG


def concat(item1, item2, dim = 1):
    out = torch.clone(item1)
    torch.cat((out, item2, out), dim)
    return out


def split_data(fts, lbls, factor=10):
    x = []
    y = []
    for i in range(len(fts)):
        feat = torch.split(fts[i], fts[i].size()[1] // factor, dim=1)
        for j in range(factor):
            x.append(feat[j])
            y.append(lbls[i])
    return x, y


def train(feats, ls, mod, loss_func, adj, target, train_size, output=[]):
    idxs = [i for i in range(train_size)]
    for j in range(10):
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
grads = [edge_filter(features[i]) for i in range(len(features))]
grads_concat = [concat(features[i], grads[i]) for i in range(len(features))]
grad_diffs = [torch.sub(features[i], grads[i]) for i in range(len(features))]
grad_diffs_concat = [concat(features[i], grad_diffs[i]) for i in range(len(features))]
y_grads = [y_filter(features[i]) for i in range(len(features))]
y_grads_concat = [concat(features[i], y_grads[i]) for i in range(len(features))]
spec_dog = [DoG(features[i]) for i in range(len(features))]
spec_dog_concat = [concat(features[i], spec_dog[i]) for i in range(len(features))]
grad_dog = [DoG(grads[i]) for i in range(len(grads))]
grad_dog_concat = [concat(features[i], grad_dog[i]) for i in range(len(features))]
grady_dog = [DoG(y_grads[i]) for i in range(len(y_grads))]
grady_dog_concat = [concat(features[i], grady_dog[i]) for i in range(len(features))]
Blur = torchvision.transforms.GaussianBlur(3, sigma=1.4)
grad_sums = [torch.sub(features[i], spec_dog[i]) for i in range(len(features))]
"""x_split, y_split = split_data(features, data_np, factor=10)
x = features"""
y = [data_split[i] for i in data_idxs]

test_size = len(features)//10
train_size = len(features) - test_size

"""test_size_split = len(x_split)//10
train_size_split = len(x_split) - test_size_split"""
row_names = []
feat_names = ["Roughness", "Breathiness", "Strain", "Pitch", "Loudness", "Resonance", "Weight"]
"""print("CNNLSTMNetwork-No Split:")
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
    torch.save(model.state_dict(), f"CNNLSTM_{feat_names[j]}.pt")"""
"""
plt.legend(feat_names)
plt.title("Per Epoch Loss Graph of CNNLSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()"""
output = []
names = ["featurization", "quality", "mean-mse", "mape", "true-mean", "predicted-mean", "r2"]
output.append(names)
print("LSTMNetwork-Spectrograms:")
for j in range(7):
    x = features
    input_size = x[0].shape[2]
    model = LSTMNetwork(input_size=input_size, input_dim2=x[0].shape[1], linear_dim=x[0].shape[1]*128*2).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
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
        y_true.append(label.item())
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
        if(label).item() != 0:
            mape.append(abs((label.item()-logits.item())/label.item()))
    print(feat_names[j])
    print("Mean MSE:", sum(losses)/len(losses))
    print("Mean Average Percentage error:", sum(mape)/len(mape))
    print("True mean of test data:", sum(y_true)/len(y_true))
    print("Mean of predicted values:", sum(y_pred)/len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    output_inner = []
    output_inner.append("Spectrograms")
    output_inner.append(feat_names[j])
    output_inner.append(str(sum(losses)/len(losses)))
    output_inner.append(sum(mape)/len(mape))
    output_inner.append(str(sum(y_true)/len(y_true)))
    output_inner.append(str(sum(y_pred)/len(y_pred)))
    output_inner.append(str(sklearn.metrics.r2_score(y_true, y_pred)))
    output.append(output_inner)
    torch.save(model.state_dict(), f"models/LSTM-Spec/LSTM_{feat_names[j]}_spec.pt")

print("LSTMNetwork-Edge Filter:")
for j in range(7):
    x = grads
    input_size = x[0].shape[2]
    model = LSTMNetwork(input_size=input_size, input_dim2=x[0].shape[1], linear_dim=x[0].shape[1]*128*2).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
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
        y_true.append(label.item())
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
        if(label).item() != 0:
            mape.append(abs((label.item()-logits.item())/label.item()))
    print(feat_names[j])
    print("Mean MSE:", sum(losses) / len(losses))
    print("Mean Average Percentage error:", sum(mape) / len(mape))
    print("True mean of test data:", sum(y_true) / len(y_true))
    print("Mean of predicted values:", sum(y_pred) / len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    output_inner = []
    output_inner.append("Gradients")
    output_inner.append(feat_names[j])
    output_inner.append(str(sum(losses) / len(losses)))
    output_inner.append(sum(mape) / len(mape))
    output_inner.append(str(sum(y_true) / len(y_true)))
    output_inner.append(str(sum(y_pred) / len(y_pred)))
    output_inner.append(str(sklearn.metrics.r2_score(y_true, y_pred)))
    output.append(output_inner)
    torch.save(model.state_dict(), f"models/LSTM-Grad/LSTM_{feat_names[j]}_grad.pt")

print("LSTMNetwork-Y Gradient:")
for j in range(7):
    x = y_grads
    input_size = x[0].shape[2]
    model = LSTMNetwork(input_size=input_size, input_dim2=x[0].shape[1], linear_dim=x[0].shape[1]*128*2).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
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
        y_true.append(label.item())
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
        if(label).item() != 0:
            mape.append(abs((label.item()-logits.item())/label.item()))
    print(feat_names[j])
    print("Mean MSE:", sum(losses) / len(losses))
    print("Mean Average Percentage error:", sum(mape) / len(mape))
    print("True mean of test data:", sum(y_true) / len(y_true))
    print("Mean of predicted values:", sum(y_pred) / len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    output_inner = []
    output_inner.append("Y-Gradients")
    output_inner.append(feat_names[j])
    output_inner.append(str(sum(losses) / len(losses)))
    output_inner.append(sum(mape) / len(mape))
    output_inner.append(str(sum(y_true) / len(y_true)))
    output_inner.append(str(sum(y_pred) / len(y_pred)))
    output_inner.append(str(sklearn.metrics.r2_score(y_true, y_pred)))
    output.append(output_inner)
    torch.save(model.state_dict(), f"models/LSTM-YGrad/LSTM_{feat_names[j]}_ygrad.pt")

print("LSTMNetwork-Spectrogram DoG:")
for j in range(7):
    x = spec_dog
    input_size = x[0].shape[2]
    model = LSTMNetwork(input_size=input_size, input_dim2=x[0].shape[1], linear_dim=x[0].shape[1]*128*2).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
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
        y_true.append(label.item())
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
        if(label).item() != 0:
            mape.append(abs((label.item()-logits.item())/label.item()))
    print(feat_names[j])
    print("Mean MSE:", sum(losses) / len(losses))
    print("Mean Average Percentage error:", sum(mape) / len(mape))
    print("True mean of test data:", sum(y_true) / len(y_true))
    print("Mean of predicted values:", sum(y_pred) / len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    output_inner = []
    output_inner.append("Spectrogram-DoG")
    output_inner.append(feat_names[j])
    output_inner.append(str(sum(losses) / len(losses)))
    output_inner.append(sum(mape) / len(mape))
    output_inner.append(str(sum(y_true) / len(y_true)))
    output_inner.append(str(sum(y_pred) / len(y_pred)))
    output_inner.append(str(sklearn.metrics.r2_score(y_true, y_pred)))
    output.append(output_inner)
    torch.save(model.state_dict(), f"models/LSTM-SpecDoG/LSTM_{feat_names[j]}_specdog.pt")

print("LSTMNetwork- Gradient Dog:")
for j in range(7):
    x = grad_dog
    input_size = x[0].shape[2]
    model = LSTMNetwork(input_size=input_size, input_dim2=x[0].shape[1], linear_dim=x[0].shape[1]*128*2).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
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
        y_true.append(label.item())
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
        if(label).item() != 0:
            mape.append(abs((label.item()-logits.item())/label.item()))
    print(feat_names[j])
    print("Mean MSE:", sum(losses) / len(losses))
    print("Mean Average Percentage error:", sum(mape) / len(mape))
    print("True mean of test data:", sum(y_true) / len(y_true))
    print("Mean of predicted values:", sum(y_pred) / len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    output_inner = []
    output_inner.append("Gradient-DoG")
    output_inner.append(feat_names[j])
    output_inner.append(str(sum(losses) / len(losses)))
    output_inner.append(sum(mape) / len(mape))
    output_inner.append(str(sum(y_true) / len(y_true)))
    output_inner.append(str(sum(y_pred) / len(y_pred)))
    output_inner.append(str(sklearn.metrics.r2_score(y_true, y_pred)))
    output.append(output_inner)
    torch.save(model.state_dict(), f"models/LSTM-GradDoG/LSTM_{feat_names[j]}_gradog.pt")

print("LSTMNetwork-Y Gradient Dog:")
for j in range(7):
    x = grady_dog
    input_size = x[0].shape[2]
    model = LSTMNetwork(input_size=input_size, input_dim2=x[0].shape[1], linear_dim=x[0].shape[1]*128*2).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
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
        y_true.append(label.item())
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
        if(label).item() != 0:
            mape.append(abs((label.item()-logits.item())/label.item()))
    print(feat_names[j])
    print("Mean MSE:", sum(losses) / len(losses))
    print("Mean Average Percentage error:", sum(mape) / len(mape))
    print("True mean of test data:", sum(y_true) / len(y_true))
    print("Mean of predicted values:", sum(y_pred) / len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    output_inner = []
    output_inner.append("Y-Gradient-DoG")
    output_inner.append(feat_names[j])
    output_inner.append(str(sum(losses) / len(losses)))
    output_inner.append(sum(mape) / len(mape))
    output_inner.append(str(sum(y_true) / len(y_true)))
    output_inner.append(str(sum(y_pred) / len(y_pred)))
    output_inner.append(str(sklearn.metrics.r2_score(y_true, y_pred)))
    output.append(output_inner)
    torch.save(model.state_dict(), f"models/LSTM-YGradDoG/LSTM_{feat_names[j]}_ygraddog.pt")


print("LSTMNetwork-Gradient Difference:")
for j in range(7):
    x = grad_diffs
    input_size = x[0].shape[2]
    model = LSTMNetwork(input_size=input_size, input_dim2=x[0].shape[1], linear_dim=x[0].shape[1]*128*2).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
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
        y_true.append(label.item())
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
        if(label).item() != 0:
            mape.append(abs((label.item()-logits.item())/label.item()))
    print(feat_names[j])
    print("Mean MSE:", sum(losses) / len(losses))
    print("Mean Average Percentage error:", sum(mape) / len(mape))
    print("True mean of test data:", sum(y_true) / len(y_true))
    print("Mean of predicted values:", sum(y_pred) / len(y_pred))
    print("R2:", sklearn.metrics.r2_score(y_true, y_pred))
    output_inner = []
    output_inner.append("Gradient-Diffs")
    output_inner.append(feat_names[j])
    output_inner.append(str(sum(losses) / len(losses)))
    output_inner.append(sum(mape) / len(mape))
    output_inner.append(str(sum(y_true) / len(y_true)))
    output_inner.append(str(sum(y_pred) / len(y_pred)))
    output_inner.append(str(sklearn.metrics.r2_score(y_true, y_pred)))
    output.append(output_inner)
    torch.save(model.state_dict(), f"models/LSTM-GradDiffs/LSTM_{feat_names[j]}graddiffs.pt")

"""plt.legend(feat_names)
plt.title("Per Epoch Loss Graph of CNNLSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()"""

"""print("CNNNetwork-No Split:")
for j in range(5):
    model = CNNNetwork(linear_dim=8064).cuda()
    loss_fn = torch.nn.MSELoss()
    adjuster = torch.optim.Adam(model.parameters())
    train_loss = []
    train(x, y, model, torch.nn.MSELoss(), adjuster, j, train_size, train_loss)
    #plt.plot([i for i in range(len(train_loss))], train_loss)
    losses = []
    y_true = []
    y_pred = []
    for i in range(train_size, len(x)):
        spec2 = x[i]
        spec2 = spec2.to(device)
        label = torch.tensor([y[i][j]])
        label = label.to(device)
        logits = model(spec2)
        y_true.append(label.item())
        y_pred.append(logits.item())
        loss = loss_fn(logits.float(), label.float())
        losses.append(loss.item())
    print(feat_names[j] + ":", sum(losses)/len(losses))
    print(sklearn.metrics.r2_score(y_true, y_pred))
    torch.save(model.state_dict(), f"CNN_{feat_names[j]}.pt")"""

"""plt.legend(feat_names)
plt.title("Per Epoch Loss Graph of CNNNetwork-No Split")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()"""

with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output)




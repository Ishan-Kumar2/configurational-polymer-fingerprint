## Data Processing and Training Script
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

LR = 0.001
MAX_EPOCHS = 40
BATCH_SIZE = 20
DATASET_PATH = "./full_dataset"
PLOT_SAVE_PATH = "./1/loss_curves.png"
WEIGHTS_SAVE_PATH = "./trained_weights"

class Dataset:
    def __init__(self, data, train=True):

        super(Dataset, self).__init__()
        self.path = data
        self.train = train
        self.validation_data_length = 50000
        # print("data of size",len(self))

    def __len__(self):

        if self.train is True:
            return len(os.listdir(self.path)) - self.validation_data_length
        else:
            return self.validation_data_length

    def __getitem__(self, index):

        # First 50k is left out at Validation Data
        if self.train:
            index = index + self.validation_data_length
        file_name = self.path + "/" + str(index) + ".txt"
        f = open(file_name, "r")
        vals = f.read().split("\n")
        # if len(vals)>115:
        # 	vals = vals[110:]
        # 	re_pos =  vals[0].find('RE')
        # 	vals[0] = vals[0][re_pos:]
        # if not vals[-1].split(',')[0] == 'Density':
        # 	vals.pop()
        assert vals[0].split(",")[0] == "RE", print("Error", vals[0])
        re = float(vals[0].split(",")[1])

        assert vals[1].split(",")[0] == "RG"
        rg = float(vals[1].split(",")[1])

        assert vals[2].split(",")[0] == "Internal Energy"
        u = float(vals[2].split(",")[1])

        assert vals[3].split(",")[0] == "RG 3D"
        rg_3d = []
        rg_3d.append(float(vals[3].split(",")[1]))
        rg_3d.append(float(vals[4].split(",")[1]))
        rg_3d.append(float(vals[5].split(",")[1]))

        assert vals[6].split(",")[0] == "Shape Anisotropy"
        sa = float(vals[6].split(",")[1])

        assert vals[7].split(",")[0] == "SideX"
        side = []
        side.append(float(vals[7].split(",")[1]))
        side.append(float(vals[8].split(",")[1]))
        side.append(float(vals[9].split(",")[1]))

        assert vals[10].split(",")[0] == "SpringConstant"
        spring = float(vals[10].split(",")[1])

        pos = vals[11:-1]
        pos = [i.split(",") for i in pos]
        if len(pos[-1]) == 4:
            pos[-1] = pos[-1][:3]
            pos[-1][2] = pos[-1][2][:-7]

        pos = [[float(j) for j in i] for i in pos]
        y_density = float(vals[-1].split(",")[1])

        pos = torch.tensor(pos)

        re = torch.tensor(re).reshape(1, -1)
        rg = torch.tensor(rg).reshape(1, -1)
        u = torch.tensor(u).reshape(1, -1)
        rg_3d = torch.tensor(rg_3d).reshape(1, -1)
        sa = torch.tensor(sa).reshape(1, -1)
        side = torch.tensor(side).reshape(1, -1)
        spring = torch.tensor(spring).reshape(1, -1)

        desc = {
            "re": re,
            "rg": rg,
            "u": u,
            "rg3d": rg_3d,
            "sa": sa,
            "side": side,
            "spring": spring,
        }

        pos = pos.reshape(1, -1)
        # X = torch.cat((pos, re, rg, u, rg_3d, sa, side, spring), axis = 1)
        return pos, desc, y_density


d = Dataset(DATASET_PATH)
v = Dataset(DATASET_PATH, False)
print("Train Length", len(d))
print("Val Length", len(v))


train_dl = DataLoader(d, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(v, batch_size=BATCH_SIZE, shuffle=False)


class Encoder(nn.Module):
    def __init__(self):

        super().__init__()
        self.l1 = nn.Linear(300, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 32)
        self.relu = nn.LeakyReLU()

    def forward(self, x):

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        return x


class Decoder(nn.Module):
    def __init__(self):

        super().__init__()
        self.l1 = nn.Linear(32, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 300)
        self.relu = nn.LeakyReLU()

    def forward(self, x):

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        return x


encoder = Encoder()
decoder = Decoder()

criterion = nn.MSELoss()
optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

epoch_losses = []
epoch_losses_val = []


def val():

    running_loss_1 = 0.0
    encoder.eval()
    decoder.eval()

    for i, data in enumerate(val_dl):

        data = data[0]
        data = torch.squeeze(data)
        hidden = encoder(data)
        reconstructed_input = decoder(hidden)
        loss = criterion(reconstructed_input, data)
        running_loss_1 += loss.item()

    validation_loss_epoch = running_loss_1 / len(val_dl)
    print(f"		Val Loss: {validation_loss_epoch}")
    epoch_losses_val.append(validation_loss_epoch)
    return validation_loss_epoch


# Initialising the prev val loss as a random high value so that the first epoch gets saved.
prev_val_loss = 1000


def train():
    prev_val_loss = val()

    for epoch in range(MAX_EPOCHS):

        running_loss = 0.0
        print(f"Epoch {epoch+1}")
        encoder.train()
        decoder.train()

        for i, data in enumerate(train_dl):

            data = data[0]
            data = torch.squeeze(data)
            optimizer.zero_grad()
            hidden = encoder(data)
            reconstructed_input = decoder(hidden)
            loss = criterion(reconstructed_input, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 1000 == 0 and i != 0:
                print(f"		Iteration {i}: Running Loss: {running_loss/i}")

        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_dl)}")
        epoch_losses.append(running_loss / len(train_dl))
        val_loss = val()

        if val_loss < prev_val_loss:

            print("Saving")
            torch.save(encoder.state_dict(), f"{WEIGHTS_SAVE_PATH}/{epoch}_enc.pt")
            torch.save(decoder.state_dict(), f"{WEIGHTS_SAVE_PATH}/{epoch}_dec.pt")
            prev_val_loss = val_loss

    return


train()
plt.plot(epoch_losses, label="Train Loss")
plt.plot(epoch_losses_val, label="Validation Loss")
plt.show()
plt.savefig(PLOT_SAVE_PATH)
plt.close()

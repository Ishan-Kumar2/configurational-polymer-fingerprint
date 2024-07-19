# Property Prediction Model Check
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns

PRED_MODEL_PATH = "./2/Pred_33_enc.pt"
ENCODER_MODEL_PATH = "./2/39_enc.pt"
DATASET_PATH = "./full_dataset"


class PredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(43, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.float()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.sigmoid(self.l3(x))
        return x


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


model = PredModel()
model.load_state_dict(torch.load(PRED_MODEL_PATH))
encoder = Encoder()
encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH))


class Dataset:
    def __init__(self, data, train=True):
        super(Dataset, self).__init__()
        self.path = data
        self.train = train
        # print("data of size",len(self))

    def __len__(self):
        return len(os.listdir(self.path)) - 2

    def __getitem__(self, index):
        # print(index)
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

        desc = torch.cat((re, rg, u, rg_3d, sa, side, spring), axis=1)

        pos = pos.reshape(1, -1)
        # X = torch.cat((pos, re, rg, u, rg_3d, sa, side, spring), axis = 1)
        return pos, desc, y_density


d = Dataset(DATASET_PATH)
val_dl = DataLoader(d, batch_size=20, shuffle=False)

print("Length ", len(d))


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


m = 0


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


criterion = nn.BCELoss()
criterion_2 = RMSELoss()
criterion_3 = nn.L1Loss()

x = []
loss = []
pred = [0]
actual = [0]


def val():
    running_loss_1 = 0.0
    running_loss_rmse = 0.0
    running_loss_mae = 0.0
    running_loss_r2 = 0.0
    m = 0.0
    count = 0
    encoder.eval()
    for i, data in enumerate(val_dl):
        pos = data[0]
        desc = data[1]
        label = data[2]
        pos = torch.squeeze(pos)
        desc = torch.squeeze(desc)
        label = torch.squeeze(label)

        learnt_descriptors = encoder(pos)
        X = torch.cat((learnt_descriptors, desc), axis=1)
        y_pred = model(X)
        y_pred = torch.squeeze(y_pred).double()
        # print(y_pred.dtype, label.dtype)
        loss = criterion(y_pred, label)
        running_loss_1 += loss.item()
        running_loss_rmse += criterion_2(y_pred, label).item()
        running_loss_mae += criterion_3(y_pred, label).item()
        running_loss_r2 += r2_loss(y_pred, label)
        actual.extend(label.tolist())
        pred.extend(y_pred.tolist())
        # Checking where the difference is greater than 0.5
        if label.tolist()[0] - y_pred.tolist()[0] > 0.5:
            print(f"Actual {label.tolist()[0]} Predicted {y_pred.tolist()[0]}")
            print(pos[0])
        # diff = np.array(label.tolist()) - np.array(y_pred.tolist())
        # diff = np.absolute(diff)
        # max_diff = max(diff.tolist())
        # index = np.where(diff == np.amax(diff))
        # m = max(m, max_diff)
        # if m>0.5:
        # 	# print(pos[index])
        # 	# print("----------------------")
        # 	count+=1

        # print(max_diff,m)
    # print(m, count)
    print(f"		Loss: {running_loss_1/len(val_dl)}")
    print(f"		RMSE: {running_loss_rmse/len(val_dl)}")
    print(f"		MAE: {running_loss_mae/len(val_dl)}")
    print(f"		R^2: {running_loss_r2/len(val_dl)}")


val()

actual = np.array(actual)
pred = np.array(pred)

# Residual Plot
plt.scatter(np.arange(len(actual)), actual - pred, s=1)
plt.legend()
plt.xlabel("X Label", fontsize=16)
plt.ylabel("Residuals", fontsize=16)
# plt.title("Residual Plot", fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.show()
plt.close()


# Distribution Plot of Actual and Predicted
sns.distplot(a=actual)
sns.distplot(a=pred)
plt.legend(labels=["Actual Probability", "Predicted Probability"], fontsize=16)
# plt.title("Probability Distribution Comparison between Actual and Predicted", fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Density", fontsize=16)
plt.show()

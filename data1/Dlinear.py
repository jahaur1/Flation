
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
# Set random seed
SEED = 2025
torch.manual_seed(SEED)
np.random.seed(SEED)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}.")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
from sklearn.preprocessing import StandardScaler

rcParams['figure.figsize'] = 13, 4

# Box
rcParams['axes.spines.top'] = False
rcParams['axes.spines.left'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.prop_cycle'] = cycler(color=['navy','goldenrod'])

# Grid and axis thickness, color
rcParams['axes.linewidth'] = 1
rcParams['axes.edgecolor'] = '#5B5859'
rcParams['axes.ymargin'] = 0
rcParams['axes.grid'] = True
rcParams['axes.grid.axis'] = 'y'
rcParams['axes.axisbelow'] = True
rcParams['grid.color'] = 'grey'
rcParams['grid.linewidth'] = 0.5

# xticks, yticks
rcParams['ytick.major.width'] = 0
rcParams['ytick.major.size'] = 0
rcParams['ytick.color'] = '#393433'
rcParams['xtick.major.width'] = 1
rcParams['xtick.major.size'] = 3
rcParams['xtick.color'] = '#393433'

# Line thickness
rcParams['lines.linewidth'] = 1.5

# Saving quality
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.dpi'] = 500
rcParams['savefig.transparent'] = True

site=626
csv_path = Path(f"E:\Project2024\canada\DATA1\WTS{site}_1.csv")
df = pd.read_csv(csv_path)
df["Date"] = pd.to_datetime(df["Date"])
df.drop(['Unnamed: 0'], axis=1, inplace=True)
#周期编码
# Day in a month
df["Day_of_month"] = df.Date.apply(lambda x: x.day)
# Day in a week
df["Day_of_week"] = df.Date.apply(lambda x: x.dayofweek)
# 24-hour based
df["Hour"] = df.Date.apply(lambda x: x.hour)
# Week in a year
df["Week"] = df.Date.apply(lambda x: x.week)

# Set "DateTime" column as row index
df = df.set_index("Date")
df.head()

batch_size = 32
in_seq_len=past_len=24# How far to look back
out_seq_len=pred_len =24# How far to look forward
forecast_length = out_seq_len# Hours ahead to predict
enc_in_size = df.shape[1] # Number of input features + target feature
dec_in_size = 6 # Number of known future features + target feature
output_size = 1 # Number of target features
hidden_size = 128 # Dimensions in hidden layers
num_layers = 2# Number of hidden layers
patch_len=6
num_epochs = 100
learning_rate = 1e-3
es_patience = 20
lr_patience = 10
model_save_path = "checkpoint_seq2seq.pth"
df['Qrate'] = np.log1p(df['Qrate'])
df['Rain'] = np.log1p(df['Rain'])
# 保存原始均值和标准差
scaler_qrate = StandardScaler()
scaler_rain = StandardScaler()
df['Qrate'] = scaler_qrate.fit_transform(df[['Qrate']])
df['Rain'] = scaler_rain.fit_transform(df[['Rain']])

# Move target to the last column
target_feature = "Qrate"
df.insert(len(df.columns)-1, target_feature, df.pop(target_feature))
data = df.values

testNum = validationNum = 8750
total_rows = len(df)
assert total_rows == testNum * 2 + (train_df_rows := total_rows - 2*testNum), "数据行数不支持这样的划分"
data_train = df[:train_df_rows].copy()
data_val = df[train_df_rows:train_df_rows+validationNum].copy()
data_test = df[train_df_rows+validationNum:train_df_rows+validationNum+testNum].copy()
print("Training Shape:", data_train.shape)
print("Validation Shape:", data_val.shape)
print("Testing Shape:", data_test.shape)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, past_len=past_len, pred_len=pred_len):
        # Create data sequences
        data_len = data.shape[0]
        # 27725
        X, Y = list(), list()

        for i in range(data_len):
            input_end = i + past_len

            output_end = input_end + pred_len

            # check if we are beyond the dataset
            if output_end > data_len:
                break
            else:
                X.append(data[i:input_end])
                Y.append(data.iloc[input_end:output_end, -1].values)

        # Shape (samples, seq_len, features)
        self.X = np.array(X)
        # Shape (samples, seq_len, ) : univariate
        self.Y = np.array(Y)

    def __len__(self):
        # return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # return one sample from the dataset
        features = self.X[idx]
        target = self.Y[idx]
        return features, target

train_dataset = TimeSeriesDataset(data_train)
val_dataset = TimeSeriesDataset(data_val)
test_dataset = TimeSeriesDataset(data_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))

print("Features shape:", X.size())
print("Target shape:", y.size())
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, enc_in, seq_len, pred_len, individual=True):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 3
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual

        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

        self.Linear_end=nn.Linear(self.channels,1)
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
# Reshape tensor dimension order: rearrange original dimension [Batch, Channel, Output length] to [Batch, Output length, Channel]
# 这一步是CICD章节实验的基础维度适配，确保后续线性层输入维度符合模型要求
# This step is the basic dimension adaptation for the experiment in the CICD chapter, ensuring the input dimension of the subsequent linear layer meets the model requirements
        x = x.permute(0, 2, 1)  

 # CICD Experiment Two-Option Configuration (Choose ONE)
# 选项1（注释此行）：禁用线性层 → 基准对照组 | Option 1 (comment): disable linear layer → baseline control group
# 选项2（取消注释）：启用线性层 → 实验组 | Option 2 (uncomment): enable linear layer → experimental group
        #x = self.Linear_end(x)
        return x


model = DLinear(dec_in_size,past_len, pred_len,individual=False).to(device)

total_params = sum(p.numel() for p in model.parameters())
learn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)



loss_func = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=learning_rate)


class EarlyStopping:

    def __init__(self, patience, model_save_path, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.model_save_path = model_save_path
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = 0
        self.early_stop = False


    def __call__(self, epoch, model, validation_loss):
        delta_loss = self.min_validation_loss - validation_loss
        # Check if val loss is smaller than min loss
        if delta_loss > self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # Save best model
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.model_save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early Stopping.")
                print(f"Save best model at epoch {self.best_epoch}")
                self.early_stop = True


lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=lr_patience, verbose=True)

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        X, y = X.float().to(device), y.float().to(device)

        # Forward pass
        output = model(X).squeeze()
        loss = loss_function(output, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_avg_loss = total_loss / num_batches

    return train_avg_loss


def val_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.float().to(device), y.float().to(device)

            output = model(X).squeeze()
            total_loss += loss_function(output, y).item()

    val_avg_loss = total_loss / num_batches

    return val_avg_loss

# Log losses for plotting
all_losses = []

# Initialize Early Stopping object
early_stopper = EarlyStopping(patience=es_patience, model_save_path=model_save_path)
for epoch in range(num_epochs):
    train_loss = train_model(train_loader, model, loss_func, opt)
    val_loss = val_model(val_loader, model, loss_func)
    all_losses.append([train_loss, val_loss])

    # Display
    print(f"\nEpoch [{epoch}/{num_epochs-1}]\t\tTrain loss: {train_loss:.6f} - Val loss: {val_loss:.6f}")

    # EarlyStopping
    early_stopper(epoch, model, val_loss)
    if early_stopper.early_stop:
        break
    # Adjust learning rate
    lr_scheduler.step(val_loss)


plt.title("LSTM Model", size=18, y=1.1)
plt.plot(all_losses, label=["Train loss", "Val loss"])
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("MSE", fontsize=13)
plt.legend(loc="upper right", fontsize=10)
plt.show()

model.load_state_dict(torch.load(model_save_path))

def predict(data_loader, model):
    pred, true = torch.tensor([]), torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X = X.double()
            y = y.double()
            y_pred = model(X.float().to(device))
            pred=pred.to(device).double()
            true =true.to(device).double()
            pred = torch.cat((pred, y_pred.to(device)), 0).double()
            true = torch.cat((true, y.to(device)), 0).double()

    return pred, true
y_pred_tensor, y_test_tensor = predict(test_loader, model)
y_pred, y_test = y_pred_tensor.cpu().numpy(), y_test_tensor.cpu().numpy()
y_pred = y_pred.squeeze()
# Inverse the transformation
y_pred_inv1 = scaler_qrate.inverse_transform(y_pred)
y_test_inv1 = scaler_qrate.inverse_transform(y_test)
y_pred_inv = np.expm1(y_pred_inv1)

y_test_inv = np.expm1(y_test_inv1)

# Hours ahead to predict
forecast_length = 24

truth = y_test_inv[:, :] 
forecast = y_pred_inv[:, :]  
print(truth.shape)

diff = np.subtract(truth, forecast)


mae = np.mean(np.abs(diff))  # MAE
mse = np.mean(np.square(diff))  # MSE
rmse = np.sqrt(mse)  # RMSE

# NSE
num = np.sum(np.square(diff))
den = np.sum(np.square(np.subtract(truth, truth.mean())))
nse = 1 - (num / den)
pbias = np.mean(diff / truth) * 100

# R^2
numerator = np.square(np.sum((truth - truth.mean()) * (forecast - forecast.mean())))
denominator = np.sum(np.square(truth - truth.mean())) * np.sum(np.square(forecast - forecast.mean()))
r_squared = numerator / denominator

# RSR
rsr = rmse / np.std(truth)

# Pbias

print(f"Overall forecast MAE : {mae:.4f}")
print(f"Overall forecast MSE: {rmse:.4f}")
print(f"Overall forecast NSE: {nse:.4f}")
print(f"Overall forecast R^2: {r_squared:.4f}")
print(f"Overall forecast RSR: {rsr:.4f}")
print(f"Overall forecast Pbias: {pbias:.2f}%")

truth = y_test_inv[:, 1]
forecast = y_pred_inv[:, 1]

df = pd.DataFrame({
    'Truth': truth,
    'Forecast': forecast
})

df.to_csv('DLinearresults.csv', index=False)


plt.title(f"{forecast_length}-Hour Ahead Forecasting lstm", size=16, y=1.1)
plt.plot(truth, label="Ground Truth", color="teal")
plt.plot(forecast, label="Prediction", color="darkred")
plt.xlabel("Observation")
plt.legend(fontsize=10)
plt.show()
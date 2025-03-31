#Read me
#In this part, the target model of this study RF-AM-MLP is trained through the training set and the verification set is verified. The trained MLP model is saved as rf_attention_model.pth for subsequent comparison with MLP and linear model
import numpy as np
import pandas as pd
import openpyxl
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
def read_excel_range(file_path, sheet_name, range_string):
    try:
        match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", range_string)
        if not match:
            raise ValueError(f"Invalid range format: {range_string}")
        col_start, row_start, col_end, row_end = match.groups()
        row_start, row_end = int(row_start), int(row_end)
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheet = wb[sheet_name]
        col_start_idx = openpyxl.utils.column_index_from_string(col_start)
        col_end_idx = openpyxl.utils.column_index_from_string(col_end)
        data = []
        for row in sheet.iter_rows(min_row=row_start, max_row=row_end,
                                   min_col=col_start_idx, max_col=col_end_idx):
            data.append([cell.value if cell.value is not None else 0 for cell in row])
        return np.array(data)
    except Exception as e:
        print(f"Error reading Excel range: {e}")
        sys.exit(1)


def calculate_amplitude_and_frequency_fft(C, window_size=2, sample_rate=1.0):
    n = len(C)
    frequencies = []
    amplitudes = []

    # 滑动窗口进行 FFT 计算
    for i in range(0, n - window_size + 1):
        window_data = C[i:i + window_size]

        # 计算 FFT
        fft_vals = np.fft.fft(window_data)
        freqs = np.fft.fftfreq(window_size, d=sample_rate)

        # 取正频率部分（window_size // 2）
        positive_freqs = freqs[:window_size // 2]
        positive_fft_vals = fft_vals[:window_size // 2]

        # 计算幅度
        amplitude = np.abs(positive_fft_vals)

        # 找到最大振幅对应的频率
        peak_freq = positive_freqs[np.argmax(amplitude)] if len(amplitude) > 0 else 0
        peak_amplitude = np.max(amplitude) if len(amplitude) > 0 else 0

        frequencies.append(peak_freq)
        amplitudes.append(peak_amplitude)

    return np.array(frequencies), np.array(amplitudes)


data_sources = [
    {
        "file_path": "C:/Users/hs/Desktop/Training data.xlsx",
        "ranges": [
            ("Sheet1", "B2:B2797", "C2:C2797"),
            ("Sheet1", "H2:H2691", "I2:I2691"),
            ("Sheet1", "N2:N3809", "O2:O3809"),
            ("Sheet1", "T2:T2289", "U2:U2289"),
            ("Sheet1", "Z2:Z6328", "AA2:AA6328"),
            ("Sheet1", "AF2:AF2582", "AG2:AG2582"),
            ("Sheet1", "AL2:AL4265", "AM2:AM4265")
        ]
    },
]

# 数据存储的列表
all_amplitude = []  # 振幅
all_frequency = []  # 频率
all_avg_C = []  # 每个窗口的电容平均值
all_delta_C_window = []  # 每个窗口的电容首尾差值
all_h = []  # h值

for source in data_sources:
    file_path = source["file_path"]
    for sheet_name, c_range, h_range in source["ranges"]:
        C = read_excel_range(file_path, sheet_name, c_range).flatten()
        h = read_excel_range(file_path, sheet_name, h_range).flatten()

        # 计算每个点的电容差值：每个点与前一个点的差
        delta_C_window = [C[i+1] - C[i] for i in range(len(C)-1)]  # 差值（每个点与前一个点的差值）

        # 计算每个点的频率和振幅
        frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)

        # 对齐数据长度
        min_len = min(len(C), len(delta_C_window), len(amplitude), len(frequency), len(h))
        C = C[:min_len]
        delta_C_window = delta_C_window[:min_len]
        amplitude = amplitude[:min_len]
        frequency = frequency[:min_len]
        h = h[:min_len]


        # 将每个点的电容值、电容差值、振幅、频率添加到列表中
        all_avg_C.extend(C)
        all_delta_C_window.extend(delta_C_window)
        all_amplitude.extend(amplitude)
        all_frequency.extend(frequency)
        all_h.extend(h)

# 转换为数组
all_avg_C = np.array(all_avg_C).reshape(-1, 1)
all_delta_C_window = np.array(all_delta_C_window).reshape(-1, 1)
all_amplitude = np.array(all_amplitude).reshape(-1, 1)
all_frequency = np.array(all_frequency).reshape(-1, 1)
all_h = np.array(all_h)

# 合并特征
X = np.hstack((all_avg_C, all_delta_C_window, all_amplitude, all_frequency))
y = all_h.flatten()

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)

rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)
X_train_leaf = rf_model.apply(X_train)
X_test_leaf = rf_model.apply(X_test)

scaler_rf = StandardScaler()
X_train_leaf = scaler_rf.fit_transform(X_train_leaf)
X_test_leaf = scaler_rf.transform(X_test_leaf)

X_train_tensor = torch.tensor(X_train_leaf, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_leaf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1)

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        attn_scores = F.softmax(self.attention_weights(x), dim=-1)
        x = x * attn_scores
        return x, attn_scores

class MLPWithAttention(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithAttention, self).__init__()
        self.attention = AttentionMechanism(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x, attn_scores = self.attention(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x, attn_scores

input_dim = X_train_leaf.shape[1]
model = MLPWithAttention(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_model(model, X_train, y_train, epochs=300):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs, attn_scores = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

train_model(model, X_train_tensor, y_train_tensor, epochs=400)

model.eval()
with torch.no_grad():
    nn_predictions, attn_scores = model(X_test_tensor)


validation_data_30 = [
    {
        "file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
        "ranges": [
            ("Sheet1", "B2:B6149", "C2:C6149"),
        ]
    }
]
validation_data_60 = [
    {
        "file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
        "ranges": [
            ("Sheet1", "I2:I3652", "J2:J3652"),
        ]
    }
]
validation_data_100 = [
    {
        "file_path": "C:/Users/hs/Desktop/Constant rates validation data.xlsx",
        "ranges": [
            ("Sheet1", "P2:P2703", "Q2:Q2703"),
        ]
    }
]


def plot_predictions_comparison(y_true, predictions, title, file_name):

    plt.figure(figsize=(10, 7.5))
    plt.plot(y_true, label='True Values', color='black', linewidth=2)
    plt.plot(predictions, label='Predictions', color='red', alpha=0.7)

    plt.title(f"Predictions vs True Values: {title}", fontsize=22)
    plt.xlabel('Sample Number', fontsize=20)
    plt.ylabel('Liquid Level (mm)', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"{title} - Predictions vs True Values plot saved as {file_name}")

def process_validation_data(data_sources, rf_model, scaler, model, process_var=1, measurement_var=0.001, plot_title="Validation"):
    all_amplitude = []  # 振幅
    all_frequency = []  # 频率
    all_avg_C = []  # 每个点的电容值
    all_delta_C_window = []  # 每个点的电容差值
    all_h = []  # h值

    for source in data_sources:
        file_path = source["file_path"]
        for sheet_name, c_range, h_range in source["ranges"]:
            C = read_excel_range(file_path, sheet_name, c_range).flatten()
            h = read_excel_range(file_path, sheet_name, h_range).flatten()

            # 计算每个点的电容差值：每个点与前一个点的差
            delta_C_window = [C[i + 1] - C[i] for i in range(len(C) - 1)]  # 差值（每个点与前一个点的差值）

            # 计算每个点的频率和振幅
            frequency, amplitude = calculate_amplitude_and_frequency_fft(C, window_size=2)

            # 对齐数据长度
            min_len = min(len(C), len(delta_C_window), len(amplitude), len(frequency), len(h))
            C = C[:min_len]
            delta_C_window = delta_C_window[:min_len]
            amplitude = amplitude[:min_len]
            frequency = frequency[:min_len]
            h = h[:min_len]

            # 将每个点的电容值、电容差值、振幅、频率添加到列表中
            all_avg_C.extend(C)
            all_delta_C_window.extend(delta_C_window)
            all_amplitude.extend(amplitude)
            all_frequency.extend(frequency)
            all_h.extend(h)

    # 转换为数组
    all_avg_C = np.array(all_avg_C).reshape(-1, 1)
    all_delta_C_window = np.array(all_delta_C_window).reshape(-1, 1)
    all_amplitude = np.array(all_amplitude).reshape(-1, 1)
    all_frequency = np.array(all_frequency).reshape(-1, 1)
    all_h = np.array(all_h)

    # 合并特征
    X_validation = np.hstack((all_avg_C, all_delta_C_window, all_amplitude, all_frequency))  # 现在是电容平均值、变化量、振幅和频率作为特征
    y_validation = all_h.flatten()

    X_validation_leaf = rf_model.apply(X_validation)

    try:
        X_validation_scaled = scaler.transform(X_validation_leaf)
    except Exception as e:
        print(f"Error scaling validation features: {e}")
        sys.exit(1)

    X_validation_tensor = torch.tensor(X_validation_scaled, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32).view(-1)

    model.eval()
    with torch.no_grad():
        try:
            validation_predictions, _ = model(X_validation_tensor).squeeze
        except Exception as e:
            print(f"Error during model prediction on validation data: {e}")
            sys.exit(1)

    smoothed_validation_predictions = validation_predictions
    smoothed_validation_predictions = smoothed_validation_predictions.cpu().numpy()  # 将 tensor 转换为 numpy 数组

    # 绘制真实值与预测值对比图
    plot_predictions_comparison(y_validation, smoothed_validation_predictions, title=plot_title, file_name=f"{plot_title}_comparison.png")

    mse_validation = mean_squared_error(y_validation, smoothed_validation_predictions)
    rmse_validation = np.sqrt(mse_validation)
    mae_validation = mean_absolute_error(y_validation, smoothed_validation_predictions)
    r2_validation = r2_score(y_validation, smoothed_validation_predictions)
    mape_validation = np.mean(np.abs((y_validation - smoothed_validation_predictions) / y_validation)) * 100

    return mse_validation, rmse_validation, mae_validation, r2_validation, mape_validation
print("\n=== 验证集评估（30转） ===")
mse_30, rmse_30, mae_30, r2_30, mape_30 = process_validation_data(validation_data_30, rf_model, scaler_rf, model, plot_title="30rpm Sulfuric Acid")
print(f"均方误差 (MSE): {mse_30:.4f}")
print(f"均方根误差 (RMSE): {rmse_30:.4f}")
print(f"平均绝对误差 (MAE): {mae_30:.4f}")
print(f"决定系数 (R²): {r2_30:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_30:.2f}%")


print("\n=== 验证集评估（60转） ===")
mse_35, rmse_35, mae_35, r2_35, mape_35 = process_validation_data(validation_data_60, rf_model, scaler_rf, model, plot_title="60rpm Sulfuric Acid")
print(f"均方误差 (MSE): {mse_35:.4f}")
print(f"均方根误差 (RMSE): {rmse_35:.4f}")
print(f"平均绝对误差 (MAE): {mae_35:.4f}")
print(f"决定系数 (R²): {r2_35:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_35:.2f}%")

print("\n=== 验证集评估（100转） ===")
mse_40, rmse_40, mae_40, r2_40, mape_40 = process_validation_data(validation_data_100, rf_model, scaler_rf, model, plot_title="100rpm Sulfuric Acid")
print(f"均方误差 (MSE): {mse_40:.4f}")
print(f"均方根误差 (RMSE): {rmse_40:.4f}")
print(f"平均绝对误差 (MAE): {mae_40:.4f}")
print(f"决定系数 (R²): {r2_40:.4f}")
print(f"平均绝对百分比误差 (MAPE): {mape_40:.2f}%")

torch.save(model.state_dict(), "rf_attention_model.pth")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler_rf, "scaler_rf.pkl")
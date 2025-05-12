import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os

# 載入本機資料集
def load_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案：{file_path}")
        exit(1)

    df = pd.read_csv(file_path)
    df = df[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
    df['TRIP_MINUTES'] = df['TRIP_SECONDS'] / 60
    return df

# 建立模型架構
def build_model(input_shape=2):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(units=1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# 訓練模型
def train_model(df, features, label, learning_rate=0.001, epochs=20, batch_size=50):
    X = df[features].values
    y = df[label].values

    # 顯示資料維度
    print(f"\n📐 Shape of X (features): {X.shape}  -> {X.shape[0]} samples × {X.shape[1]} features")
    print(f"📐 Shape of y (label):    {y.shape}  -> {y.shape[0]} labels")

    # 顯示前 5 筆資料
    print("\n🔍 First 5 rows of X:")
    print(X[:5])
    print("\n🔍 First 5 labels (y):")
    print(y[:5])

    model = build_model(input_shape=len(features))
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, history

# 顯示 loss curve
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 印出模型參數與預測公式
def print_model_info(model, features):
    weights, bias = model.layers[0].get_weights()
    print("\n📌 Model Parameters:")
    for i, w in enumerate(weights):
        print(f"  - Weight for {features[i]}: {w[0]:.4f}")
    print(f"  - Bias: {bias[0]:.4f}")

    equation = f"{weights[0][0]:.4f} * {features[0]}"
    for i in range(1, len(features)):
        equation += f" + {weights[i][0]:.4f} * {features[i]}"
    equation += f" + {bias[0]:.4f}"

    print("\n📌 Estimated Fare Formula:")
    print(f"  FARE = {equation}")

# 主執行流程
if __name__ == "__main__":
    # 本機檔案位置
    file_path = r"C:\Users\irene\OneDrive\桌面\ML_Google\chicago_taxi_train.csv"

    # 超參數
    learning_rate = 0.001
    epochs = 20
    batch_size = 50
    features = ['TRIP_MILES', 'TRIP_MINUTES']
    label = 'FARE'

    print("📥 Reading local dataset...")
    df = load_dataset(file_path)

    print("\n🔧 Hyperparameters:")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - epochs: {epochs}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - features: {features}")
    print(f"  - label: {label}")

    print("\n🧠 Training model...")
    model, history = train_model(df, features, label, learning_rate, epochs, batch_size)

    print_model_info(model, features)
    plot_loss(history)

    print("\n✅ Training complete. Exiting program.")
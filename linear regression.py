import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 載入套件與資料
# 資料: x是車重, y是油耗(MPG)
x = np.array([3.5, 3.69, 3.44, 3.43, 3.34, 4.42, 2.37]).reshape(-1, 1)
y = np.array([18, 15, 18, 16, 15, 14, 24])

# 2. 建立模型與訓練
model = LinearRegression()
model.fit(x, y)

# 3. 預測與計算誤差
y_pred = model.predict(x)
mse = mean_squared_error(y, y_pred)

# 4. 顯示結果
print(f"斜率(Weigh): {model.coef_[0]:.3f}")
print(f"截距(Bias): {model.intercept_:.3f}")
print(f"均平方差(MSE): {mse: .3f}")

# 5. 畫圖觀察預測線
plt.scatter(x, y, color='blue', label='actual value')
plt.plot(x, y_pred, color = 'orange', label = 'predition')
plt.xlabel('Weight(1000s of pounds)')
plt.ylabel('Fuel Efficiency (MPG)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()
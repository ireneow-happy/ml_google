import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv("C:/Users/irene/OneDrive/桌面/ML_Google/chicago_taxi_train.csv")

# 選擇數值欄位
df = df[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'TIP_RATE']]

# 計算相關矩陣
corr = df.corr(numeric_only=True)

# 顯示相關矩陣熱力圖
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# 顯示 pairplot 散佈圖
sns.pairplot(df[['TRIP_MILES', 'TRIP_SECONDS', 'FARE']])
plt.suptitle("Pairplot of Taxi Trip Features", y=1.02)  # 調整標題位置
plt.tight_layout()
plt.show()

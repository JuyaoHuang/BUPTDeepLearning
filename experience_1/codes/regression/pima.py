import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 可从https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data下载
df = pd.read_csv('./pima/diabetes.csv')

# pd.set_option('display.max_columns', None)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.shape)
# print(df.columns)

# ======== 填充缺失值 ============
df_nan = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[df_nan] = df[df_nan].replace(to_replace=0, value=np.nan)

median_vals = df[df_nan].median()
# print(f'要填充的中位数：\n{median_vals}')

df.fillna(median_vals, inplace=True)
# print((df[df_nan] == 0).sum())
# print(df[df_nan].describe())
# ====== 填充完毕 ========
# ====== 数据标准化 =======
# 从DF中分离出特征矩阵 X和目标向量 y
X = df.drop('Outcome', axis=1)
y = df['Outcome']
# 切分训练集，测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# 执行标准化并将数据格式从 数组转回 DF
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# 在 X_test (DataFrame) 上进行转换
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
# ====== 数据标准化完毕 =======
# ======= 逻辑回归训练 =======
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 创建逻辑回归模型的实例
# random_state 是为了保证每次运行结果一致，便于调试
model = LogisticRegression(random_state=42)

# 使用标准化的训练数据来训练模型
model.fit(X_train_scaled_df, y_train)

# 在训练集和测试集上检查性能
y_train_pred = model.predict(X_train_scaled_df)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test_scaled_df)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'训练集准确率：{train_accuracy:.4%}\n测试集准确率：{test_accuracy:.4%}')
# =========== 逻辑回归训练结束 =================

# =========== 绘制逻辑回归散点图 ===============
# 为了可视化决策边界，只使用两个特征来训练一个新模型
feat_plot = ['Glucose','BMI']
X_train_plt = X_train_scaled_df[feat_plot]
y_train_plt = y_train

model_plot = LogisticRegression(random_state=42)
model_plot.fit(X_train_plt, y_train_plt)

plt.figure(figsize=(10,6))
# 绘制决策边界
# 获取两个特征的最小值和最大值
x_min,x_max = X_train_plt.iloc[:,0].min() - 1,X_train_plt.iloc[:,0].max()+1
y_min,y_max = X_train_plt.iloc[:,1].min() - 1,X_train_plt.iloc[:,1].max()+1
# 使用 np.meshgrid 生成网格点坐标矩阵
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),
                    np.arange(y_min,y_max,0.02))

print(xx)
print(type(xx))

# 在网格上进行预测
# ravel()将矩阵展平，np.c_[]将它们按列拼接，以便模型一次性预测所有点
Z = model_plot.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和数据点
plt.contourf(xx,yy,Z,alpha=0.4,cmap=plt.cm.coolwarm)

# 绘制原始数据点的散点图
plot_df = pd.DataFrame(X_train_plt).copy()
plot_df['Outcome'] = y_train_plt.values

sns.scatterplot(x='Glucose',y='BMI',hue='Outcome',data=plot_df,
                palette={0:'blue',1:'red'},edgecolor='k')

plt.title('Logistic Regression Scatter Plot and Decision Boundary (Glucose vs BMI)', fontsize=16)
plt.xlabel('Glucose (scaler)', fontsize=12)
plt.ylabel('BMI (scaler)', fontsize=12)
plt.legend(title='is suffered', labels=['0: no', '1: yes'])
plt.grid(True)
# plt.savefig('./pima.png',dpi=300)
plt.show()
# =========== 绘制逻辑回归散点图结束 ===============

# ====== 第三问：在拆分的数据集上测试模型准确率 ======
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X_split = df.drop('Outcome', axis=1)
y_split = df['Outcome']
# 创建一个字典存储不同比例下的准确率
res = {}
train_split_size = [0.75,0.8,0.85]
print("\n开始在不同的数据集拆分比例下评估模型")

for size in train_split_size:
    print(f'当前训练集比例：{size}')

    X_train,X_test,y_train,y_test = train_test_split(
        X_split,y_split,train_size=size,random_state=42
    )
    print(f'训练集样本数{X_train.shape[0]}，测试集样本数{X_test.shape[0]}')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42).fit(X_train_scaled,y_train)

    y_test_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test,y_test_pred)
    print(f'模型在测试集上的准确率：{accuracy:.4f}')

    res[size] = accuracy
print(f'\n所有比例测试完成')

res_df=pd.DataFrame(list(res.items()),columns=['Train Size','Test Accuracy'])
print(res_df)
# ====== 第三问结束 ==========

# ======== 第四问开始 =============
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=42)
print(f'开始训练模型...')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(random_state=42).fit(X_train_scaled,y_train)
print(f'模型训练结束')
# 获取模型系数
coefficients = model.coef_[0]
# print(coefficients)
# 将模型系数与特征名称（列名）绑定
feat_names = X_train.columns
# 构建新的 DataFrame
crucial_feat = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefficients})
# 使用绝对值给特征权重排序
crucial_feat['Abs Coefficient'] = crucial_feat['Coefficient'].abs()
crucial_feat.sort_values('Abs Coefficient', ascending=False, inplace=True)
print(crucial_feat)
# 绘制条形图
plt.figure(1, figsize=(18, 8))
# y轴是特征名称，x轴是其重要性（系数绝对值）
ax = sns.barplot(x='Abs Coefficient', y='Feature', data=crucial_feat, palette='viridis')
# ax = crucial_feat.plot(kind='bar',rot=0)
plt.title('The impact of different characteristics on diabetes',fontsize=16)
plt.xlabel('Absolute Coefficient',fontsize=14)
plt.ylabel('Feature',fontsize=12)
plt.xlim(0, 1.2)

for p in ax.patches:
    # 获取条形的宽度，对于水平条形图，宽度就是其代表的数值
    width = p.get_width()

    # 获取条形的y坐标和高度，用于确定文本的垂直位置
    y = p.get_y()
    height = p.get_height()

    # 在条形的末端右侧添加文本
    ax.text(x=width + 0.01,
            y=y + height / 2,
            s=f'{width:.4f}',
            va='center',
            ha='left'
            )
plt.grid(axis='x', linestyle='--', alpha=0.5)

# plt.savefig('./imgs/feature_impact_Diabetes.png',dpi=500)
plt.show()
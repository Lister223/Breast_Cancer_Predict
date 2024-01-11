'''
import csv
import numpy as np
import pickle
import gzip

with open(r'D:\昱呈專用資料夾\程式語言\專案\乳癌\BCA含欄位.csv', newline='', encoding='utf-8') as csvfile:
    insert = list(csv.reader(csvfile))
import pandas as pd
df = pd.DataFrame(insert[1:], columns=insert[0])
#檢查數據格式
print(df.dtypes)

#改為浮點數
trans = ['M_Radius', 'M_Texture', 'M_perimeter', 'M_area', 'M_smoothness', 'M_compactness', 'M_concavity', 'M_concavepoints',
         'M_symmetry', 'M_fractaldimension', 'SE_Radius', 'SE_Texture', 'SE_perimeter', 'SE_area', 'SE_smoothness', 'SE_compactness',
         'SE_concavity', 'SE_concavepoints', 'SE_symmetry', 'SE_fractaldimension', 'W_Radius', 'W_Texture', 'W_perimeter',
         'W_area', 'W_smoothness', 'W_compactness',  'W_concavity', 'W_concavepoints', 'W_symmetry', 'W_fractaldimension']
df[trans] = df[trans].apply(pd.to_numeric, errors='coerce')
print(df.dtypes)

#轉numpy
df = df.to_numpy()

#定義standard函數
def standard(data):
    #資料標準化正規化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data, scaler

#將表準化後數據、參數導入
df[:,2:],scaler = standard(df[:,2:])

#儲存標準化參數
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


#指派數據
X = df[:, 2:]
y = df[:, 1]

#分割數據
from sklearn.model_selection import train_test_split
# 假設 X 為數據的特徵，y 為數據的標籤
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##訓練
from sklearn.linear_model import LogisticRegression

# 建立Logistic模型
Logistic = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs',max_iter=3000)

# 在訓練集上訓練模型
Logistic.fit(X_train, y_train)

# 使用已經訓練好的模型對測試集進行預測
y_pred = Logistic.predict(X_test)
##準確率評估
print('模型性能指標')
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy:{accuracy}')
result = classification_report(y_test, y_pred)
print(result)
print(confusion_matrix(y_test, y_pred))

##PR、ROC、AUG評估
# 載入相關套件
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# 使用已經訓練好的模型對測試集進行預測
y_pred_proba = Logistic.predict_proba(X_test)[:, 1]

# 計算 PR 曲線
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba, pos_label='M')

# 計算 ROC 曲線
fpr, tpr, _ = roc_curve(y_test, y_pred_proba,  pos_label='M')

# 計算 AUC
auc_score = auc(fpr, tpr)

# 繪製 PR 曲線
import matplotlib.pyplot as plt
plt.plot(recall, precision, color='blue', label='PR Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# 繪製 ROC 曲線
plt.plot(fpr, tpr, color='red', label='ROC Curve (AUC = %0.2f)' % auc_score)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.show()

###學習曲線測試是否有過度擬合
import numpy as np
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(Logistic, X, y, cv=10)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

##將模型儲存

with gzip.GzipFile('model/Logistic_breastcancer.pgz', 'w') as f:
    pickle.dump(Logistic, f)

'''
import pickle
import gzip
import numpy as np

#讀取標準化參數
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
#讀取Model
with gzip.open('model/Logistic_breastcancer.pgz', 'r') as f:
    Logistic = pickle.load(f)
    data = np.array([[15.34,14.26,102.5,704.4,0.1073,0.2135,0.2077,0.09756,0.2521,0.07032,0.4388,0.7096,3.384,44.91,0.006789,0.05328,0.06446,0.02252,0.03672,0.004394,18.07,19.08,125.1,980.9,0.139,0.5954,0.6305,0.2393,0.4667,0.09946]
                    ,[9.173,13.86,59.2,260.9,0.07721,0.08751,0.05988,0.0218,0.2341,0.06963,0.4098,2.265,2.608,23.52,0.008738,0.03938,0.04312,0.0156,0.04192,0.005822,10.01,19.23,65.59,310.1,0.09836,0.1678,0.1397,0.05087,0.3282,0.0849]
                    ])
    data = scaler.transform(data)
    pred = Logistic.predict(data)
    print(pred)








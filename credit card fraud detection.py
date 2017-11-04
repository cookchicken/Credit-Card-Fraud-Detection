import numpy as np
import pandas as pd
import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('whitegrid')
import missingno as msno

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import  recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format',lambda x: '%.4f' %x)  # 修改excel格式防止报错
from imblearn.over_sampling import SMOTE
import itertools

# ①-------------数据获取与解析------------
datapath = 'I:/Learning/数据挖掘/creditcard.csv'
pwd = os.getcwd()    #pd.read_csv读取时只能传文件不能传文件地址
os.chdir(os.path.dirname(datapath))
data_cr = pd.read_csv(os.path.basename(datapath), encoding='latin-1')
os.chdir(pwd)
# print(data_cr.shape) #查看数据集大小
# print(data_cr.info())#查看数据基本信息
# print(data_cr.describe())#查看数据基本统计信息
# msno.matrix(data_cr) #查看数据缺失值

# ②------------数据特征------------
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
sns.countplot(x='Class', data=data_cr, ax=axs[0])
axs[0].set_title("Frequency of each Class")
data_cr['Class'].value_counts().plot(x=None, y=None, kind='pie', ax=axs[1], autopct='%1.2f%%')
axs[1].set_title("Percentage of each Class")
# plt.show()
# print(data_cr.groupby('Class').size()) # 查看目标列情况
data_cr['Hour'] = data_cr['Time'].apply(lambda x: divmod(x, 3600)[0])  # [0]取整数部分舍弃余数

# 查看正常用户与被盗刷用户之间的区别
Xfraut = data_cr.loc[data_cr['Class'] == 1]
Xnonfraut = data_cr.loc[data_cr['Class'] == 0]
# 正常用户
correlationNonfraut = Xnonfraut.ix[:, data_cr.columns != 'Class'].corr()  # 获取未被盗取用户除class以外别的数据之间的相关系数
mask = np.zeros_like(correlationNonfraut)  # 获得与correlation对应的零矩阵
indices = np.triu_indices_from(correlationNonfraut)  # 获得correlation矩阵的梯形上升矩阵
mask[indices] = True
grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize=(14,9))
cmap = sns.diverging_palette(220, 8, as_cmap=True)  # 选择对称色调
ax1 = sns.heatmap(correlationNonfraut, ax=ax1, vmin=-1, vmax=1, cmap=cmap, \
                  square=False, linewidths=0.5, mask=mask, cbar=False)  # 用热度图表示相关系数
ax1.set_xticklabels(ax1.get_xticklabels(), rotation='vertical', size=16)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation='vertical', size=16)
ax1.set_title('Normal', size=20)
# 被盗用户
correlationfraut = Xfraut.ix[:, data_cr.columns != 'Class'].corr()
ax2 = sns.heatmap(correlationfraut, ax=ax2, vmin=-1, vmax=1, cmap=cmap, \
                  square=False, linewidths=0.5, mask=mask, yticklabels=False, \
                  cbar_ax=cbar_ax, cbar_kws={'orientation': 'vertical', \
                                             'ticks': [-1, -0.5, 0, 0.5, 1]})
ax2.set_xticklabels(ax2.get_xticklabels(), size=16)
ax2.set_title('Fraud', size=20)

cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size=14)
# plt.show()
# 盗刷交易，交易金额和交易次数的关系
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 4))
bins = 30
ax1.hist(data_cr["Amount"][data_cr["Class"] == 1], bins=bins)  # 用直方图的形式表示
ax1.set_title('Fraud')
ax2.hist(data_cr["Amount"][data_cr["Class"] == 0], bins=bins)
ax2.set_title('Normal')

plt.xlabel('Amount($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')  # 在y轴上缩放
# plt.show()
# 交易时间
sns.factorplot(x='Hour', data=data_cr, kind="count", palette="ocean", size=6, aspect=3)

# 盗刷交易，交易金额和交易时间的关系
f, (ax1, ax2)=plt.subplots(2, 1, sharex=True, figsize=(16,6))
ax1.scatter(data_cr['Hour'][data_cr['Class'] == 1], data_cr['Amount'][data_cr['Class'] == 1])
ax1.set_title('Fraud')

ax2.scatter(data_cr['Hour'][data_cr['Class'] == 0], data_cr['Amount'][data_cr['Class'] == 0])
ax2.set_title('Normal')

plt.xlabel('Time (in hours)')
plt.ylabel('Amount')
#plt.show()

# print("Fraud Stats Summary")
# print(data_cr['Amount'][data_cr['Class'] == 1].describe())
# print()
# print("Normal Stats Summary")
# print(data_cr['Amount'][data_cr['Class'] == 0].describe())

# 数据清洗（通过分析单变量的不同分布情况选择需要的数据源）
v_feat = data_cr.ix[:, 1:29].columns
plt.figure(figsize=(16, 28*4))
gs = gridspec.GridSpec(28, 1)
for i,cn in enumerate(v_feat):
    ax=plt.subplot(gs[i])
    sns.distplot(data_cr[cn][data_cr['Class'] == 1], bins=50)
    sns.distplot(data_cr[cn][data_cr['Class'] == 0], bins=100)
    ax.set_xlabel('')
    ax.set_title('histogram of feature'+str(cn))
    #  plt.show()
#  剔除无明显区别数据
droplist = ['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Time']
data_new = data_cr.drop(droplist, axis=1)
# print(data_new.shape)  # 新数据集大小

# 对Amount和Hour进行特征缩放
col = ['Amount', 'Hour']
sc = StandardScaler()  # 减去均值并处以方差
data_new[col] = sc.fit_transform(data_new[col])

# 对特征的重要性进行排序
# ①构建变量
x_feature = list(data_new.columns)
x_feature.remove('Class')
x_val = data_new[x_feature]
y_val = data_new['Class']
# ②利用随机森林算法对元素重要性进行排序
names = data_cr[x_feature].columns
clf = RandomForestClassifier(n_estimators=10, random_state=123)   # 为了让每次随机划分取值相同设随机种子为123
clf.fit(x_val, y_val)
names, clf.feature_importances_
for feature in zip(names, clf.feature_importances_):
    print(feature)
# ③绘图
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 6)
importances = clf.feature_importances_
feat_name = names
indices = np.argsort(importances)[::-1]  # 返回从大到小的索引值
fig = plt.figure(figsize=(20, 6))
plt.title("Feature importances by RandomTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue', align='center')
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_name[indices], rotation='vertical', fontsize=14)
plt.xlim([-1, len(indices)])
# plt.show()

# ③------------模型训练-----------
# 因为数据集中正反样本数量差距过大对模型学习造成困扰，所以采用SMOTE过采样算法
x = data_cr[x_feature]
y = data_cr['Class']
n_sample = y.shape[0]
n_pos_sample = y[y == 0].shape[0]
n_neg_sample = y[y == 1].shape[0]
# print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
#                                               n_pos_sample / n_sample,
#                                               n_neg_sample / n_sample))
# print("特征维数：", x.shape[1])
sm = SMOTE(random_state=42)
x, y = sm.fit_sample(x, y)
# print("通过SMOTE算法平衡后的正反样本")
n_sample = y.shape[0]
n_pos_sample = y[y == 0].shape[0]
n_neg_sample = y[y == 1].shape[0]
# print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
#                                                    n_pos_sample / n_sample,
#                                                    n_neg_sample / n_sample))

# 构建分类器进行训练
clf1 = LogisticRegression()
clf1.fit(x, y)
predicted1 = clf1.predict(x)
#print("Test set accuracy score:{:.5f}".format(accuracy_score(predicted1, y)))
# 对分类器进行评价(混淆矩阵与ROC曲线)
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    theresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > theresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y, predicted1)
np.set_printoptions(precision=2)  # float point output
class_name = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, class_name)
# plt.show()

y_pred1_prob = clf1.predict_proba(x)[:, 1]  # 被分类为1和0的概率
fpr, tpr, thresholds = roc_curve(y, y_pred1_prob)
roc_auc = auc(fpr,tpr)  # 获得AUC值

plt.title("Receiver Operating Characteristic")
plt.plot(fpr, tpr, 'b', label='AUC = %0.5f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()

# ④------------模型评估与优化----------
'''
    在③中构建的模型由于训练和测试都在同一数据集中进行，这样可能导致模型产生过拟合的风险
    一般，数据集的划分有三种方式 ：1.留出法，2.交叉验证法，3.自助法
    在这里采用交叉验证法，将数据划分为三部分：训练集，验证集，测试集。模型在训练集上学习，
在验证集上参数调优，在测试集上进行模型性能评估
    模型调优采用网格搜索调优参数，通过构建参数候选集合，然后网格搜索穷举各种参数组合，根据
评分机制找到最好的一组
    具体操作采用scikit-learn模块model_selection中的GridSearchCV方法
'''
# cross-validation+grid search
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)  #为保证每次切分都一样random_state置0
param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=10)
grid_search.fit(x_train, y_train)
results = pd.DataFrame(grid_search.cv_results_)
best = np.argmax(results.mean_test_score.values)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score:{:.5f}".format(grid_search.best_score_))
y_pred = grid_search.predict(x_test)
print("Test set accuracy score:{:.5f}".format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
# 混淆矩阵表示
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print("Recall metric in the testing dataset: ", cnf_matrix[1, 1]/(cnf_matrix[1, 1]+cnf_matrix[1, 0]))
class_name = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, class_name)
# plt.show()

# 模型评估(混淆矩阵和PRC曲线)
y_pred_prob = grid_search.predict_proba(x_test)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  #测试不同阈值的查全率
plt.figure(figsize=(15, 10))
j = 1
for i in thresholds:
    y_test_prediction_high_recall = y_pred_prob[:, 1] > i
    plt.subplot(3,3,j)
    j += 1
    cnf_matrix = confusion_matrix(y_test, y_test_prediction_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))
    class_name = [0, 1]
    plot_confusion_matrix(cnf_matrix, class_name)

colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red',
                          'yellow', 'green', 'blue', 'black'])
plt.figure(figsize=(12, 7))
j = 1
for i, color in zip(thresholds, colors):
    y_test_predictions_prob = y_pred_prob[:, 1] > i
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_predictions_prob)
    area = auc(recall, precision)
    plt.plot(recall, precision, color=color, label='Threshold: %s, AUC=%0.5f' % (i, area))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    # plt.show()
'''
    precision和recall是一组矛盾的变量。从上面混淆矩阵和PRC曲线可以看到，阈值越小，recall
    值越大，模型能找出信用卡被盗刷的数量也就更多，但换来的代价是误判的数量也较大。随着
    阈值的提高，recall值逐渐降低，precision值也逐渐提高，误判的数量也随之减少。通过调整
    模型阈值，控制模型反信用卡欺诈的力度，若想找出更多的信用卡被盗刷就设置较小的阈值,
    反之，则设置较大的阈值。
'''
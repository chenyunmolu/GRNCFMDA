import numpy as np
import pandas as pd
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# 基于HMDAD数据库实现CV1、CV2、CV3的AUC值，可视化为柱状图

GMF_HMDAD = pd.read_excel('./ablation_experiment.xlsx', sheet_name='GMF_HMDAD_AUC', header=None)
GMF_HMDAD_auc = auc(GMF_HMDAD[0], GMF_HMDAD[1])
GMF_Disbiome = pd.read_excel('./ablation_experiment.xlsx', sheet_name='GMF_Disbiome_AUC', header=None)
GMF_Disbiome_auc = auc(GMF_Disbiome[0], GMF_Disbiome[1])

MLP_HMDAD = pd.read_excel('./ablation_experiment.xlsx', sheet_name='MLP_HMDAD_AUC', header=None)
MLP_HMDAD_auc = auc(MLP_HMDAD[0], MLP_HMDAD[1])
MLP_Disbiome = pd.read_excel('./ablation_experiment.xlsx', sheet_name='MLP_Disbiome_AUC', header=None)
MLP_Disbiome_auc = auc(MLP_Disbiome[0], MLP_Disbiome[1])

NCF_HMDAD = pd.read_excel('./ablation_experiment.xlsx', sheet_name='NCF_HMDAD_AUC', header=None)
NCF_HMDAD_auc = auc(NCF_HMDAD[0], NCF_HMDAD[1])
NCF_Disbiome = pd.read_excel('./ablation_experiment.xlsx', sheet_name='NCF_Disbiome_AUC', header=None)
NCF_Disbiome_auc = auc(NCF_Disbiome[0], NCF_Disbiome[1])
#
plt.figure(figsize=(10, 8))
# 示例数据
labels = ['HMDAD', 'Disbiome']

GMF = [GMF_HMDAD_auc, GMF_Disbiome_auc]
MLP = [MLP_HMDAD_auc, MLP_Disbiome_auc]
NCF = [NCF_HMDAD_auc, NCF_Disbiome_auc]
print(GMF)
# 柱的宽度
bar_width = 0.25

# x轴的位置
r1 = np.arange(len(labels))
r2 = [x + bar_width + 0.05 for x in r1]
r3 = [x + bar_width + 0.05 for x in r2]

# 计算每组的中心位置
group_centers = [r + bar_width for r in r1]  # 每组中心位置
# 绘制柱状图
bars1 = plt.bar(r1, GMF, color='#ffc089', width=bar_width, edgecolor='grey', label='GMF')
bars2 = plt.bar(r2, MLP, color='#62a0ca', width=bar_width, edgecolor='grey', label='MLP')
bars3 = plt.bar(r3, NCF, color='#9ad19a', width=bar_width, edgecolor='grey', label='NCF')


def add_value_labels(bars, fontsize='21'):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom',
                 fontsize=fontsize)


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.xticks(fontsize='24')
plt.yticks(fontsize='24')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置 Y 轴刻度格式为两位小数
plt.ylim(0.8, 1.0)
# plt.xlabel('dataset', fontsize='24')
plt.xticks([r + bar_width + 0.05 for r in range(len(labels))], labels)
plt.ylabel('AUC', fontsize='24')

plt.legend(loc='upper center', framealpha=0, bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize='22')
plt.savefig('./AUC.png')
# plt.show()

# 基于HMDAD数据库实现CV1、CV2、CV3的AUPR值，可视化为柱状图

GMF_HMDAD = pd.read_excel('./ablation_experiment.xlsx', sheet_name='GMF_HMDAD_AUPR', header=None)
GMF_HMDAD_aupr = auc(GMF_HMDAD[0], GMF_HMDAD[1])
GMF_Disbiome = pd.read_excel('./ablation_experiment.xlsx', sheet_name='GMF_Disbiome_AUPR', header=None)
GMF_Disbiome_aupr = auc(GMF_Disbiome[0], GMF_Disbiome[1])

MLP_HMDAD = pd.read_excel('./ablation_experiment.xlsx', sheet_name='MLP_HMDAD_AUPR', header=None)
MLP_HMDAD_aupr = auc(MLP_HMDAD[0], MLP_HMDAD[1])
MLP_Disbiome = pd.read_excel('./ablation_experiment.xlsx', sheet_name='MLP_Disbiome_AUPR', header=None)
MLP_Disbiome_aupr = auc(MLP_Disbiome[0], MLP_Disbiome[1])

NCF_HMDAD = pd.read_excel('./ablation_experiment.xlsx', sheet_name='NCF_HMDAD_AUPR', header=None)
NCF_HMDAD_aupr = auc(NCF_HMDAD[0], NCF_HMDAD[1])
NCF_Disbiome = pd.read_excel('./ablation_experiment.xlsx', sheet_name='NCF_Disbiome_AUPR', header=None)
NCF_Disbiome_aupr = auc(NCF_Disbiome[0], NCF_Disbiome[1])
#
plt.figure(figsize=(10, 8))
# 示例数据
labels = ['HMDAD', 'Disbiome']
GMF = [GMF_HMDAD_aupr, GMF_Disbiome_aupr]
MLP = [MLP_HMDAD_aupr, MLP_Disbiome_aupr]
NCF = [NCF_HMDAD_aupr, NCF_Disbiome_aupr]
print(GMF)
# 柱的宽度
bar_width = 0.25

# x轴的位置
r1 = np.arange(len(labels))
r2 = [x + bar_width + 0.05 for x in r1]
r3 = [x + bar_width + 0.05 for x in r2]

# 绘制柱状图
bars1 = plt.bar(r1, GMF, color='#ffc089', width=bar_width, edgecolor='grey',
                label='GMF')
bars2 = plt.bar(r2, MLP, color='#62a0ca', width=bar_width, edgecolor='grey', label='MLP')
bars3 = plt.bar(r3, NCF, color='#9ad19a', width=bar_width, edgecolor='grey', label='NCF')


def add_value_labels(bars, fontsize='21'):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom',
                 fontsize=fontsize)


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.xticks(fontsize='24')
plt.yticks(fontsize='24')
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置 Y 轴刻度格式为两位小数
plt.ylim(0.0, 1.0)
# plt.xlabel('dataset', fontsize='24')
plt.xticks([r + bar_width + 0.05 for r in range(len(labels))], labels)
plt.ylabel('AUPR', fontsize='24')

plt.legend(loc='upper center', framealpha=0, bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize='22')
plt.savefig('./AUPR.png')
# plt.show()


# 基于ACC数据绘制柱状图
GMF_HMDAD_acc = 0.9219
GMF_Disbiome_acc = 0.8075

MLP_HMDAD_acc = 0.8990
MLP_Disbiome_acc = 0.8179

NCF_HMDAD_acc = 0.9387
NCF_Disbiome_acc = 0.8348
#
plt.figure(figsize=(10, 8))
# 示例数据
labels = ['HMDAD', 'Disbiome']
GMF = [GMF_HMDAD_acc, GMF_Disbiome_acc]
MLP = [MLP_HMDAD_acc, MLP_Disbiome_acc]
NCF = [NCF_HMDAD_acc, NCF_Disbiome_acc]
print(GMF)
# 柱的宽度
bar_width = 0.25

# x轴的位置
r1 = np.arange(len(labels))
r2 = [x + bar_width + 0.05 for x in r1]
r3 = [x + bar_width + 0.05 for x in r2]

# 绘制柱状图
bars1 = plt.bar(r1, GMF, color='#ffc089', width=bar_width, edgecolor='grey',
                label='GMF')
bars2 = plt.bar(r2, MLP, color='#62a0ca', width=bar_width, edgecolor='grey', label='MLP')
bars3 = plt.bar(r3, NCF, color='#9ad19a', width=bar_width, edgecolor='grey', label='NCF')


def add_value_labels(bars, fontsize='21'):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom',
                 fontsize=fontsize)


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.xticks(fontsize='24')
plt.yticks(fontsize='24')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置 Y 轴刻度格式为两位小数
plt.ylim(0.8, 1.0)
# plt.xlabel('dataset', fontsize='24')
plt.xticks([r + bar_width + 0.05 for r in range(len(labels))], labels)
plt.ylabel('ACC', fontsize='24')

plt.legend(loc='upper center', framealpha=0, bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize='22')
plt.savefig('./ACC.png')
# plt.show()


# 基于F1-score数据绘制柱状图
GMF_HMDAD_F1 = 0.7171
GMF_Disbiome_F1 = 0.4570

MLP_HMDAD_F1 = 0.6629
MLP_Disbiome_F1 = 0.5021

NCF_HMDAD_F1 = 0.7315
NCF_Disbiome_F1 = 0.511
#
plt.figure(figsize=(10, 8))
# 示例数据
labels = ['HMDAD', 'Disbiome']
GMF = [GMF_HMDAD_F1, GMF_Disbiome_F1]
MLP = [MLP_HMDAD_F1, MLP_Disbiome_F1]
NCF = [NCF_HMDAD_F1, NCF_Disbiome_F1]
print(GMF)
# 柱的宽度
bar_width = 0.25

# x轴的位置
r1 = np.arange(len(labels))
r2 = [x + bar_width + 0.05 for x in r1]
r3 = [x + bar_width + 0.05 for x in r2]

# 绘制柱状图
bars1 = plt.bar(r1, GMF, color='#ffc089', width=bar_width, edgecolor='grey',
                label='GMF')
bars2 = plt.bar(r2, MLP, color='#62a0ca', width=bar_width, edgecolor='grey', label='MLP')
bars3 = plt.bar(r3, NCF, color='#9ad19a', width=bar_width, edgecolor='grey', label='NCF')


def add_value_labels(bars, fontsize='21'):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom',
                 fontsize=fontsize)


add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.xticks(fontsize='24')
plt.yticks(fontsize='24')
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置 Y 轴刻度格式为两位小数
plt.ylim(0.0, 1.0)
# plt.xlabel('dataset', fontsize='24')
plt.xticks([r + bar_width + 0.05 for r in range(len(labels))], labels)
plt.ylabel('F1-score', fontsize='24')

plt.legend(loc='upper center', framealpha=0, bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize='22')
plt.savefig('./F1-score.png')
# plt.show()

import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 基于Disbiome数据库的ROC曲线绘制
plt.figure(figsize=(10, 8))

sheetName = 'GRAND_Disbiome_AUC'
GRAND = pd.read_excel('./ablation_experiment_GRAND.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GRAND[0], GRAND[1])
plt.plot(GRAND[0], GRAND[1], color="#D81C38", lw=3, label='GRAND(AUC=%0.4f)' % mean_auc)

sheetName = 'GAT_Disbiome_AUC'
GAT = pd.read_excel('./ablation_experiment_GRAND.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GAT[0], GAT[1])
plt.plot(GAT[0], GAT[1], color='#ffc089', lw=3, label='GAT(AUC=%0.4f)' % mean_auc)

sheetName = 'GCN_Disbiome_AUC'
GCN = pd.read_excel('./ablation_experiment_GRAND.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GCN[0], GCN[1])
plt.plot(GCN[0], GCN[1], color='#9b7ebb', lw=3, label='GCN(AUC=%0.4f)' % mean_auc)

sheetName = 'RAW_Disbiome_AUC'
RAW = pd.read_excel('./ablation_experiment_GRAND.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(RAW[0], RAW[1])
plt.plot(RAW[0], RAW[1], color='#9ad19a', lw=3, label='RAW(AUC=%0.4f)' % mean_auc)

# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize='24')
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize='24')
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.tick_params(axis='both', which='major', direction='in', length=8)
plt.tick_params(axis='both', which='minor', direction='in', length=4)
plt.xlabel("False Positive Rate", fontsize='24')
plt.ylabel("True Positive Rate", fontsize='24')
plt.title("Receiver Operating Characteristic Curve", fontsize='24')
# 将图例固定在右下
plt.legend(loc=4, framealpha=0, bbox_to_anchor=(1, 0), borderaxespad=0.5, fontsize='22')
plt.savefig('./Disbiome_AUC.png')
# plt.show()

# 基于Disbiome数据库的PR曲线绘制
plt.figure(figsize=(10, 8))
sheetName = 'GRAND_Disbiome_AUPR'
GRAND = pd.read_excel('./ablation_experiment_GRAND.xlsx', sheet_name=sheetName, header=None)
mean_aupr = sklearn.metrics.auc(GRAND[0], GRAND[1])
plt.plot(GRAND[0], GRAND[1], color="#D81C38", lw=3, label='GRAND(AUPR=%0.4f)' % mean_aupr)

sheetName = 'GAT_Disbiome_AUPR'
GAT = pd.read_excel('./ablation_experiment_GRAND.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GAT[0], GAT[1])
plt.plot(GAT[0], GAT[1], color='#ffc089', lw=3, label='GAT(AUPR=%0.4f)' % mean_auc)

sheetName = 'GCN_Disbiome_AUPR'
GCN = pd.read_excel('./ablation_experiment_GRAND.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GCN[0], GCN[1])
plt.plot(GCN[0], GCN[1], color='#9b7ebb', lw=3, label='GCN(AUPR=%0.4f)' % mean_auc)

sheetName = 'RAW_Disbiome_AUPR'
RAW = pd.read_excel('./ablation_experiment_GRAND.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(RAW[0], RAW[1])
plt.plot(RAW[0], RAW[1], color='#9ad19a', lw=3, label='RAW(AUPR=%0.4f)' % mean_auc)

plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize='24')
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize='24')
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.tick_params(axis='both', which='major', direction='in', length=8)
plt.tick_params(axis='both', which='minor', direction='in', length=4)
plt.xlabel('Recall', fontsize='24')
plt.ylabel('Precision', fontsize='24')
plt.title('Precision-Recall Curve', fontsize='24')
plt.legend(loc=1, framealpha=0, bbox_to_anchor=(1, 1), borderaxespad=0.5, fontsize='22')
plt.savefig('./Disbiome_AUPR.png')
# plt.show()

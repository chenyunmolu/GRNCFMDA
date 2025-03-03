import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 基于Disbiome数据库的ROC曲线绘制
plt.figure(figsize=(10, 8))

sheetName = 'Disbiome_AUC'
GRNCFMDA = pd.read_excel('./data/GRNCFMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GRNCFMDA[0], GRNCFMDA[1])
plt.plot(GRNCFMDA[0], GRNCFMDA[1], color="#D81C38", lw=3, label='GRNCFMDA(AUC=%0.4f)' % mean_auc)

MNNMDA = pd.read_excel('./data/MNNMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(MNNMDA[0], MNNMDA[1])
plt.plot(MNNMDA[0], MNNMDA[1], color="#62a0ca", lw=3, label='MNNMDA(AUC=%0.4f)' % mean_auc)

GATMDA = pd.read_excel('./data/GATMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GATMDA[0], GATMDA[1])
plt.plot(GATMDA[0], GATMDA[1], color='#ffc089', lw=3, label='GATMDA(AUC=%0.4f)' % mean_auc)

NTSHMDA = pd.read_excel('./data/NTSHMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(NTSHMDA[0], NTSHMDA[1])
plt.plot(NTSHMDA[0], NTSHMDA[1], color='#9ad19a', lw=3, label='NTSHMDA(AUC=%0.4f)' % mean_auc)

LRLSHMDA = pd.read_excel('./data/LRLSHMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(LRLSHMDA[0], LRLSHMDA[1])
plt.plot(LRLSHMDA[0], LRLSHMDA[1], color='#9b7ebb', lw=3, label='LRLSHMDA(AUC=%0.4f)' % mean_auc)

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
plt.show()

# 基于Disbiome数据库的PR曲线绘制
plt.figure(figsize=(10, 8))
sheetName = 'Disbiome_AUPR'
GRNCFMDA = pd.read_excel('./data/GRNCFMDA.xlsx', sheet_name=sheetName, header=None)
mean_aupr = sklearn.metrics.auc(GRNCFMDA[0], GRNCFMDA[1])
plt.plot(GRNCFMDA[0], GRNCFMDA[1], color="#D81C38", lw=3, label='GRNCFMDA(AUPR=%0.4f)' % mean_aupr)

MNNMDA = pd.read_excel('./data/MNNMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(MNNMDA[0], MNNMDA[1])
plt.plot(MNNMDA[0], MNNMDA[1], color="#62a0ca", lw=3, label='MNNMDA(AUPR=%0.4f)' % mean_auc)

GATMDA = pd.read_excel('./data/GATMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GATMDA[0], GATMDA[1])
plt.plot(GATMDA[0], GATMDA[1], color='#ffc089', lw=3, label='GATMDA(AUPR=%0.4f)' % mean_auc)

NTSHMDA = pd.read_excel('./data/NTSHMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(NTSHMDA[0], NTSHMDA[1])
plt.plot(NTSHMDA[0], NTSHMDA[1], color='#9ad19a', lw=3, label='NTSHMDA(AUPR=%0.4f)' % mean_auc)

LRLSHMDA = pd.read_excel('./data/LRLSHMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(LRLSHMDA[0], LRLSHMDA[1])
plt.plot(LRLSHMDA[0], LRLSHMDA[1], color='#9b7ebb', lw=3, label='LRLSHMDA(AUPR=%0.4f)' % mean_auc)

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
plt.show()

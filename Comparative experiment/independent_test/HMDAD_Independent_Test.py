import numpy as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 基于HMDAD数据库的ROC曲线绘制
plt.figure(figsize=(10, 8))

sheetName = 'HMDAD_AUC'
GRNCFMDA = pd.read_excel('./data/GRNCFMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(GRNCFMDA[0], GRNCFMDA[1])
plt.plot(GRNCFMDA[0], GRNCFMDA[1], color="#D81C38", lw=3, label='GRNCFMDA(AUC=%0.4f)' % mean_auc)

SABMDA = pd.read_excel('./data/SABMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(SABMDA[0], SABMDA[1])
plt.plot(SABMDA[0], SABMDA[1], color='#62a0ca', lw=3, label='SABMDA(AUC=%0.4f)' % mean_auc)

DVAMDA = pd.read_excel('./data/DVAMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(DVAMDA[0], DVAMDA[1])
plt.plot(DVAMDA[0], DVAMDA[1], color='#ffc089', lw=3, label='DVAMDA(AUC=%0.4f)' % mean_auc)

WTHMDA = pd.read_excel('./data/WTHMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(WTHMDA[0], WTHMDA[1])
plt.plot(WTHMDA[0], WTHMDA[1], color="#9ad19a", lw=3, label='WTHMDA(AUC=%0.4f)' % mean_auc)

CMFHMDA = pd.read_excel('./data/CMFHMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(CMFHMDA[0], CMFHMDA[1])
plt.plot(CMFHMDA[0], CMFHMDA[1], color='#9b7ebb', lw=3, label='CMFHMDA(AUC=%0.4f)' % mean_auc)

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
plt.savefig('./HMDAD_AUC.png')
# plt.show()

# 基于HMDAD数据库的PR曲线绘制
plt.figure(figsize=(10, 8))
sheetName = 'HMDAD_AUPR'
GRNCFMDA = pd.read_excel('./data/GRNCFMDA.xlsx', sheet_name=sheetName, header=None)
mean_aupr = sklearn.metrics.auc(GRNCFMDA[0], GRNCFMDA[1])
plt.plot(GRNCFMDA[0], GRNCFMDA[1], color="#D81C38", lw=3, label='GRNCFMDA(AUPR=%0.4f)' % mean_aupr)

SABMDA = pd.read_excel('./data/SABMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(SABMDA[0], SABMDA[1])
plt.plot(SABMDA[0], SABMDA[1], color='#62a0ca', lw=3, label='SABMDA(AUPR=%0.4f)' % mean_auc)

DVAMDA = pd.read_excel('./data/DVAMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(DVAMDA[0], DVAMDA[1])
plt.plot(DVAMDA[0], DVAMDA[1], color='#ffc089', lw=3, label='DVAMDA(AUPR=%0.4f)' % mean_auc)

WTHMDA = pd.read_excel('./data/WTHMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(WTHMDA[0], WTHMDA[1])
plt.plot(WTHMDA[0], WTHMDA[1], color="#9ad19a", lw=3, label='WTHMDA(AUPR=%0.4f)' % mean_auc)

CMFHMDA = pd.read_excel('./data/CMFHMDA.xlsx', sheet_name=sheetName, header=None)
mean_auc = sklearn.metrics.auc(CMFHMDA[0], CMFHMDA[1])
plt.plot(CMFHMDA[0], CMFHMDA[1], color='#9b7ebb', lw=3, label='CMFHMDA(AUPR=%0.4f)' % mean_auc)

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
plt.legend(loc=3, framealpha=0, bbox_to_anchor=(0, 0), borderaxespad=0.5, fontsize='22')
plt.savefig('./HMDAD_AUPR.png')
# plt.show()

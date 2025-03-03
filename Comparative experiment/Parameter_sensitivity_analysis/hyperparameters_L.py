import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 示例数据
from matplotlib import font_manager, ticker
from matplotlib.ticker import FixedLocator

'''
MLP的层数:L
'''
data = pd.read_excel("Parameter_sensitivity_analysis.xlsx", sheet_name='HMDAD_L', header=0)
L = data['L']
auc_values = data['AUC']
aupr_values = data['AUPR']
# 创建双 y 轴
fig, ax1 = plt.subplots(figsize=(10, 8))

# 绘制 AUC 曲线
auc_line, = ax1.plot(L, auc_values, color='red', marker='p', lw=4, linestyle='dashed', label='AUC',
                     markersize=20)  # 红色，虚线
ax1.set_xlabel('L', fontsize='32')
ax1.set_xticks(L)

ax1.set_ylabel('AUC Values under five-fold CV', color='red', fontsize='24', fontweight='bold')
ax1.tick_params(axis='x', labelsize='24', which='major', direction='in', length=8)
ax1.tick_params(axis='y', labelcolor='red', labelsize='24', direction='in', length=8)
ax1.set_ylim(0.9, 1.0)

# 添加第二个 y 轴
ax2 = ax1.twinx()
aupr_line, = ax2.plot(L, aupr_values, color='#1D71AA', marker='*', lw=4, linestyle='dashed', label='AUPR',
                      markersize=20)  # 蓝色，星型点
ax2.set_ylabel('AUPR Values under five-fold CV', color='#1D71AA', fontsize='24', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#1D71AA', labelsize='24', direction='in', length=8)
ax2.set_ylim(0.0, 1.0)

# 设置字体大小
font_properties = font_manager.FontProperties(size='28')  # 设置字体大小为10
# 创建图例并同时放在右上角
fig.legend(handles=[auc_line, aupr_line],  # 曲线句柄
           labels=['AUC', 'AUPR'],  # 图例标签
           loc='lower right',  # 位置在右上角
           bbox_to_anchor=(1, 0),  # 图例锚点
           bbox_transform=ax1.transAxes,  # 使用第一个坐标轴的变换
           borderaxespad=0.3,  # 与坐标轴的距离
           ncol=1,  # 每列显示一个图例
           frameon=True,
           prop=font_properties)  # 取消边框

# 显示图形
plt.title('', fontsize='36')
plt.savefig('hyperparameters_L.png')
# plt.show()

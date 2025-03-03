import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import font_manager

'''
参数敏感性分析：构造微生物或者疾病图邻接矩阵的阈值t
'''
data = pd.read_excel("Parameter_sensitivity_analysis.xlsx", sheet_name='HMDAD_t', header=0)
t = data['t']
auc_values = data['AUC']
aupr_values = data['AUPR']
# 创建双 y 轴
fig, ax1 = plt.subplots(figsize=(10, 8))

# 绘制 AUC 曲线
auc_line, = ax1.plot(t, auc_values, color='red', marker='p', lw=4, linestyle='dashed', label='AUC',
                     markersize=20)  # 红色，虚线
ax1.set_xlabel('t', fontsize='32')
ax1.set_xticks(np.linspace(0, 1, 11))
ax1.set_ylabel('AUC Values under five-fold CV', color='red', fontsize='24', fontweight='bold')
ax1.tick_params(axis='x', labelsize='24', which='major', direction='in', length=8)
ax1.tick_params(axis='y', labelcolor='red', labelsize='24', direction='in', length=8)
ax1.set_ylim(0.9, 1.0)

# 设置不同刻度长度
for i, tick in enumerate(ax1.xaxis.get_major_ticks()):  # 遍历每个刻度对象
    if i % 2 == 0:  # 如果刻度索引是双数
        tick.tick1line.set_markersize(8)  # 设置主刻度线长度
        tick.tick2line.set_markersize(8)  # 设置次刻度线长度（对称轴）
    else:  # 如果刻度索引是单数
        tick.tick1line.set_markersize(4)
        tick.tick2line.set_markersize(4)

# 添加第二个 y 轴
ax2 = ax1.twinx()
aupr_line, = ax2.plot(t, aupr_values, color='#1D71AA', marker='*', lw=4, linestyle='dashed', label='AUPR',
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
plt.savefig('hyperparameters_t.png')
# plt.show()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

label = pd.read_csv('dataframe/Disbiome_raw_labels.csv', index_col=0)
label = np.array(label).astype(int).flatten()
all_features_input = pd.read_csv('dataframe/Disbiome_raw_input.csv', index_col=0)

tsne = TSNE(n_components=2, random_state=36, perplexity=30, n_iter=1000)
tsne_embedding = tsne.fit_transform(all_features_input)

# 绘制散点图
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=tsne_embedding[:, 0],
    y=tsne_embedding[:, 1],
    hue=label,
    palette={0: "#486FB7", 1: "#D48143"},  # 指定颜色
    edgecolor="white",  # 边框颜色
    s=15  # 点的大小
)

# 隐藏刻度值
plt.xticks([])  # 隐藏x轴刻度值
plt.yticks([])  # 隐藏y轴刻度值
# 添加标题和坐标轴标签
plt.title("", fontsize=24)
plt.xlabel("", fontsize=24)
plt.ylabel("", fontsize=24)
# 调整图例
plt.legend(title=None, loc="upper right", fontsize='xx-large')
plt.savefig("RAW_Tsne.png")
plt.show()

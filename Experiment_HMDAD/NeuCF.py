import os
import random

import numpy as np
import torch
import xlsxwriter
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_distances
from torch import nn
from collections import Counter


def draw_ROC_curve(FPR, TPR, dataset):
    mean_fpr = np.linspace(0, 1, 1000)
    tprs = []
    for i in range(len(FPR)):
        tprs.append(np.interp(mean_fpr, FPR[i], TPR[i]))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    filepath = "./Result/%s/" % dataset
    os.makedirs(filepath, exist_ok=True)
    data_toExcel(mean_fpr, mean_tpr, filepath + "AUC_%.4f_mean.xlsx" % mean_auc,
                 "%s_AUC" % dataset)
    '''
    修改刻度长度，并且显示双数，隐藏单数，建议根据需求进行更改
    '''
    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, color='red', label='Mean ROC (AUC = %0.4f)' % mean_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.tick_params(axis='both', which='major', direction='in', length=6)
    plt.tick_params(axis='both', which='minor', direction='in', length=3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    # 将图例固定在右下
    plt.legend(loc=4)
    plt.show()


def draw_PR_curve(test_label_all, test_predict_prob_all, dataset):
    y_real = np.concatenate(test_label_all)
    y_proba = np.concatenate(test_predict_prob_all)
    precisions, recalls, _ = precision_recall_curve(y_real, y_proba, pos_label=1)
    mean_aupr = metrics.auc(recalls, precisions)

    filepath = "./Result/%s/" % dataset
    os.makedirs(filepath, exist_ok=True)
    data_toExcel(recalls, precisions, filepath + "AUPR_%.4f_mean.xlsx" % mean_aupr,
                 "%s_AUPR" % dataset)

    plt.figure(figsize=(10, 8))
    plt.plot(recalls, precisions, color='red', label='Mean PR (AUPR = %0.4f)' % mean_aupr)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.tick_params(axis='both', which='major', direction='in', length=6)
    plt.tick_params(axis='both', which='minor', direction='in', length=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    # 将图例固定在右下
    plt.legend(loc=4)
    plt.show()


# xlsxwriter库储存数据到excel：以下示例是将AUC曲线的fpr，tpr分别作为x，y轴数据
def data_toExcel(x, y, fileName, sheet):
    # 创建一个新的Excel工作簿
    workbook = xlsxwriter.Workbook(fileName)
    # 根据提供的索引添加一个新的工作表，并激活该工作表
    worksheet1 = workbook.add_worksheet(sheet)
    worksheet1.activate()
    # 遍历x和y的数据，并将它们写入工作表中
    for i in range(len(x)):
        insertData = [x[i], y[i]]  # 准备要写入的数据行
        row = 'A' + str(i + 1)  # 计算要写入数据的行号
        worksheet1.write_row(row, insertData)  # 写入数据行
    # 关闭工作簿，将数据保存到文件
    workbook.close()

# 在KMeans聚类的基础上使用cosine_distances计算样本与中心的距离,从而选择高质量的负样本
def get_negative_sample_by_KMeans_and_cosine_distances(NEGATIVE_SAMPLE_CHA_ALL, positive_sample_number):
    kmeans = KMeans(n_clusters=23, random_state=36).fit(NEGATIVE_SAMPLE_CHA_ALL)
    kmeans_labels = kmeans.labels_
    kmeans_cluster_centers = kmeans.cluster_centers_
    # plotKMeansResult(NEGATIVE_SAMPLE_CHA_ALL, kmeans_labels, kmeans_cluster_centers)
    type = [[] for _ in range(23)]
    for i in range(len(kmeans_labels)):
        type[kmeans_labels[i]].append(NEGATIVE_SAMPLE_CHA_ALL[i])
    mytype = [[] for _ in range(23)]
    for j in range(23):
        # mytype[j] = random.sample(type[j], positive_sample_number // 23)
        type_numpy = np.array(type[j])
        kmeans_cluster_centers_numpy = np.array(kmeans_cluster_centers[j])
        cosine = cosine_distances(type_numpy, kmeans_cluster_centers_numpy.reshape(1, -1))
        sorted_index = np.argsort(cosine.ravel())
        mytype[j] = type_numpy[sorted_index[:positive_sample_number // 23]]
    mytype_np = np.array(mytype)
    NEGATIVE_SAMPLE_CHA = mytype_np.reshape(-1, NEGATIVE_SAMPLE_CHA_ALL.shape[1])
    NEGATIVE_SAMPLE_CHA_LABEL = np.zeros((NEGATIVE_SAMPLE_CHA.shape[0],))
    return NEGATIVE_SAMPLE_CHA, NEGATIVE_SAMPLE_CHA_LABEL


def setup_seed(seed):
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 可选设置


class NeuCF(nn.Module):
    def __init__(self, args, num_mic, num_dis):
        super(NeuCF, self).__init__()
        self.num_mic = num_mic
        self.num_dis = num_dis
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp = int(args.layers[0] / 2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.embedding_mic_mlp = nn.Embedding(num_embeddings=self.num_mic, embedding_dim=self.factor_num_mlp)
        self.embedding_dis_mlp = nn.Embedding(num_embeddings=self.num_dis, embedding_dim=self.factor_num_mlp)

        self.embedding_mic_mf = nn.Embedding(num_embeddings=self.num_mic, embedding_dim=self.factor_num_mf)
        self.embedding_dis_mf = nn.Embedding(num_embeddings=self.num_dis, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=args.layers[1] + args.layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_mic_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_dis_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_mic_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_dis_mf.weight, std=0.01)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, mic_indices, dis_indices):

        mlp_vector = torch.cat([mic_indices, dis_indices], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(mic_indices, dis_indices)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

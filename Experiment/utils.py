import os

import dgl
import torch
import numpy as np
import random

import xlsxwriter
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 可选设置


def build_heterograph(new_microbe_disease_matrix, microbeSimi, diseaseSimi):
    matAdj_microbe = np.where(microbeSimi > 0.5, 1, 0)
    matAdj_disease = np.where(diseaseSimi > 0.5, 1, 0)
    h_adjmat_1 = np.hstack((matAdj_microbe, new_microbe_disease_matrix))
    h_adjmat_2 = np.hstack((new_microbe_disease_matrix.transpose(), matAdj_disease))
    Heterogeneous = np.vstack((h_adjmat_1, h_adjmat_2))
    # heterograph
    g = dgl.heterograph(
        data_dict={
            ('microbe_disease', 'interaction', 'microbe_disease'): Heterogeneous.nonzero()},
        num_nodes_dict={
            'microbe_disease': new_microbe_disease_matrix.shape[0] + new_microbe_disease_matrix.shape[1]
        })
    return g


def train_feature_choose(rel_adj_mat, features_embedding):
    mic_num = rel_adj_mat.size()[0]
    features_embedding_mic = features_embedding[0:mic_num, :]
    features_embedding_dis = features_embedding[mic_num:features_embedding.size()[0], :]

    train_mic_feature_input, train_dis_feature_input, train_label = [], [], []
    # 添加正样本及其标签
    positive_index_tuple = torch.where(rel_adj_mat == 1)
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
    for (m, d) in positive_index_list:
        train_mic_feature_input.append(features_embedding_mic[m, :].unsqueeze(0))
        train_dis_feature_input.append(features_embedding_dis[d, :].unsqueeze(0))
        train_label.append(rel_adj_mat[m, d].item())

    # 添加负样本及其标签
    negative_index_tuple = torch.where(rel_adj_mat == 0)
    negative_index_list_all = list(zip(negative_index_tuple[0], negative_index_tuple[1]))
    negative_index_list = random.sample(negative_index_list_all, len(positive_index_list))
    for (m, d) in negative_index_list:
        train_mic_feature_input.append(features_embedding_mic[m, :].unsqueeze(0))
        train_dis_feature_input.append(features_embedding_dis[d, :].unsqueeze(0))
        train_label.append(rel_adj_mat[m, d].item())
    train_mic_feature_input = torch.cat(train_mic_feature_input, dim=0).to(device)
    train_dis_feature_input = torch.cat(train_dis_feature_input, dim=0).to(device)
    train_label = torch.FloatTensor(np.array(train_label)).unsqueeze(1).to(device)
    return train_mic_feature_input, train_dis_feature_input, train_label


def test_feature_choose(rel_adj_mat, features_embedding):
    mic_num, dis_num = rel_adj_mat.size()[0], rel_adj_mat.size()[1]
    features_embedding_mic = features_embedding[0:mic_num, :]
    features_embedding_dis = features_embedding[mic_num:features_embedding.size()[0], :]
    test_mic_feature_input, test_dis_feature_input, test_lable = [], [], []

    for i in range(mic_num):
        for j in range(dis_num):
            test_mic_feature_input.append(features_embedding_mic[i, :].unsqueeze(0))
            test_dis_feature_input.append(features_embedding_dis[j, :].unsqueeze(0))
            test_lable.append(rel_adj_mat[i, j].item())

    test_mic_feature_input = torch.cat(test_mic_feature_input, dim=0).to(device)
    test_dis_feature_input = torch.cat(test_dis_feature_input, dim=0).to(device)
    test_lable = torch.FloatTensor(np.array(test_lable)).unsqueeze(1).to(device)
    return test_mic_feature_input, test_dis_feature_input, test_lable


def consis_loss(logps, temp, lam):
    # 如果 p 是对数概率分布（比如通过 log_softmax 得到的对数概率），
    # th.exp(p) 可以将其转换回标准概率分布。这样可以直观地表示概率，便于进一步分析或可视化。
    # ps = [torch.exp(p) for p in logps]
    ps = torch.stack(logps, dim=2)
    # 获取所有分布的平均值：Z_hat(avg_p):这个技巧倒是挺巧妙的
    avg_p = torch.mean(ps, dim=2)
    # （2708，7）
    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    # （2708，7，1）
    sharp_p = sharp_p.unsqueeze(2)
    loss = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2), dim=1, keepdim=True))

    loss = lam * loss
    return loss


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


def draw_ROC_curve(FPR, TPR):
    mean_fpr = np.linspace(0, 1, 1000)
    tprs = []
    for i in range(len(FPR)):
        tprs.append(np.interp(mean_fpr, FPR[i], TPR[i]))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    '''
    修改刻度长度，并且显示双数，隐藏单数，建议根据需求进行更改
    '''
    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, color='red', label='Mean ROC (AUC = %0.4f)' % mean_auc)
    plt.plot(FPR[0], TPR[0], color='blue', label='Mean ROC (AUC = %0.4f)' % mean_auc)
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


def draw_PR_curve(test_label_all, test_predict_prob_all):
    y_real = np.concatenate(test_label_all)
    y_proba = np.concatenate(test_predict_prob_all)
    precisions, recalls, _ = precision_recall_curve(y_real, y_proba, pos_label=1)
    mean_aupr = metrics.auc(recalls, precisions)

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

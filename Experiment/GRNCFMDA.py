import argparse
import math
import os
import numpy as np
import pandas as pd
import timeit
import warnings
import random

import sklearn
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, \
    roc_curve, precision_recall_curve
from torch import optim
import torch.nn.functional as F

from model import GRAND
from utils import setup_seed, build_heterograph, consis_loss, data_toExcel, draw_ROC_curve, draw_PR_curve

start_time = timeit.default_timer()
# 设置模型运行参数
parser = argparse.ArgumentParser(description='GRNCFMDA')
# data source params
parser.add_argument('--dataset', type=str, default='Disbiome', choices=['HMDAD', 'Disbiome'], help='Name of dataset.')
# device params
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
# training params
parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
parser.add_argument('--early_stopping', type=int, default=200, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 reg.')
parser.add_argument('--seed', type=int, default=36, help='Random seed.')
# GRAND model params
parser.add_argument('--init_dim', type=int, default=128, help='Initialize embedding dimension')
parser.add_argument("--n_class", type=int, default=1, help='The number of categories, default binary classification')
parser.add_argument('--dropnode_rate', type=float, default=0.5,
                    help='Dropnode rate (1 - keep probability).')
parser.add_argument('--order', type=int, default=8, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Coefficient of consistency regularization')
# NCF model params
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
parser.add_argument("--layers", nargs='+', default=[256, 128, 64],
                    help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")

args = parser.parse_args()
dataset = args.dataset
device = args.device
seed = args.seed
setup_seed(seed)
warnings.filterwarnings('ignore')

print("dataset: {}".format(dataset))
# 读取关联矩阵和特征矩阵信息
print("-------------------------------------------读取关联矩阵和特征矩阵信息-----------------------------------------------")
association_matrix = pd.read_csv("../Dataset/{}/mircobe_disease_association_matrix.csv".format(dataset), index_col=0)
microbefeature = pd.read_csv("../Dataset/{}/microbe_similarity_fusion_matrix.csv".format(dataset), index_col=0)
diseasefeature = pd.read_csv("../Dataset/{}/disease_similarity_fusion_matrix.csv".format(dataset), index_col=0)
MD = np.array(association_matrix.values)
MM = np.array(microbefeature.values)
DD = np.array(diseasefeature.values)

microbeSim_tensor = torch.from_numpy(MM).to(torch.float32).to(device)
diseaseSim_tensor = torch.from_numpy(DD).to(torch.float32).to(device)

microbe_disease_matrix = np.copy(MD)
microbe_number = microbe_disease_matrix.shape[0]
disease_number = microbe_disease_matrix.shape[1]

positive_index_tuple = np.where(microbe_disease_matrix == 1)
positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
random.shuffle(positive_index_list)
positive_split = math.ceil(len(positive_index_list) / 5)
# 定义评价指标
all_auc = []
all_aupr = []
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_mcc = []
all_prediction_matrix = []
# 用于绘制ROC、PR曲线的参数列表
FPR = []
TPR = []
PRECISION = []
RECALL = []
test_label_all = []
test_predict_prob_all = []
print("-------------------------------------------开始进行五折交叉验证-----------------------------------------------")
count = 0
for i in range(0, len(positive_index_list), positive_split):
    count = count + 1
    print(
        f"-------------------------------------------五折交叉验证：第 {count} 折-----------------------------------------------")
    positive_index_to_zero = positive_index_list[i:i + positive_split]
    new_microbe_disease_matrix = microbe_disease_matrix.copy()
    for index in positive_index_to_zero:
        new_microbe_disease_matrix[index[0], index[1]] = 0

    new_microbe_disease_matrix_tensor = torch.from_numpy(new_microbe_disease_matrix).to(device)
    graph = build_heterograph(new_microbe_disease_matrix, MM, DD).to(device)

    model = GRAND(microbe_number, disease_number, args.init_dim, args.n_class, args, args.sample,
                  args.order, args.dropnode_rate).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    '''Training'''
    model.train()
    for epoch in range(args.epochs):
        loss_sup = 0
        train_logits, train_labels = model(graph, microbeSim_tensor, diseaseSim_tensor,
                                           new_microbe_disease_matrix_tensor, True)
        # calculate supervised loss
        for k in range(args.sample):
            loss_sup += F.binary_cross_entropy(train_logits[k], train_labels[k])
        loss_sup = loss_sup / args.sample
        # calculate consistency loss
        # loss_consis = consis_loss(train_logits, args.tem, args.lam)
        loss_train = loss_sup
        opt.zero_grad()
        loss_train.backward()
        opt.step()
        print('Epoch: {:04d} | loss_train: {:.4f}'.format(epoch + 1, loss_train.data.item()))
    '''Testing'''
    model.eval()
    with torch.no_grad():
        test_logits, test_labels = model(graph, microbeSim_tensor, diseaseSim_tensor,
                                         new_microbe_disease_matrix_tensor, False)

    test_predict_prob = test_logits.cpu().detach().numpy()
    test_label = test_labels.cpu().detach().numpy()
    test_predict = np.where(test_predict_prob > 0.5, 1, 0)

    # 使用test_labels_predict
    accuracy = accuracy_score(test_label, test_predict)
    precision = precision_score(test_label, test_predict, average='macro')
    recall = recall_score(test_label, test_predict, average='macro')
    f1 = f1_score(test_label, test_predict, average='macro')
    mcc = matthews_corrcoef(test_label, test_predict)
    # test_labels_predict_positive_proba
    auc = roc_auc_score(test_label, test_predict_prob)
    fpr, tpr, thresholds1 = roc_curve(test_label, test_predict_prob, pos_label=1)
    pre, rec, thresholds2 = precision_recall_curve(test_label, test_predict_prob, pos_label=1)
    aupr = sklearn.metrics.auc(rec, pre)

    FPR.append(fpr)
    TPR.append(tpr)
    PRECISION.append(pre)
    RECALL.append(rec)
    test_label_all.append(test_label)
    test_predict_prob_all.append(test_predict_prob)

    print("auc:{}".format(auc))
    print("aupr:{}".format(aupr))
    print("accuracy:{}".format(accuracy))
    print("precision:{}".format(precision))
    print("recall:{}".format(recall))
    print("f1_score:{}".format(f1))
    print("mcc:{}".format(mcc))
    all_auc.append(auc)
    all_aupr.append(aupr)
    all_accuracy.append(accuracy)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    all_mcc.append(mcc)

    # 选择将fpr、tpr（pre、rec）写入Excel表格
    filepath = "./Result/%s/" % (dataset)
    os.makedirs(filepath, exist_ok=True)
    data_toExcel(fpr, tpr, filepath + "AUC_%.4f.xlsx" % (auc), "%s_AUC" % (dataset))
    data_toExcel(rec, pre, filepath + "AUPR_%.4f.xlsx" % (aupr), "%s_AUPR" % (dataset))

mean_auc = np.around(np.mean(np.array(all_auc)), 4)
mean_aupr = np.around(np.mean(np.array(all_aupr)), 4)
mean_accuracy = np.around(np.mean(np.array(all_accuracy)), 4)
mean_precision = np.around(np.mean(np.array(all_precision)), 4)
mean_recall = np.around(np.mean(np.array(all_recall)), 4)
mean_f1 = np.around(np.mean(np.array(all_f1)), 4)
mean_mcc = np.around(np.mean(np.array(all_mcc)), 4)
# 计算标准差
std_auc = np.around(np.std(np.array(all_auc)), 4)
std_aupr = np.around(np.std(np.array(all_aupr)), 4)
std_accuracy = np.around(np.std(np.array(all_accuracy)), 4)
std_precision = np.around(np.std(np.array(all_precision)), 4)
std_recall = np.around(np.std(np.array(all_recall)), 4)
std_f1 = np.around(np.std(np.array(all_f1)), 4)
std_mcc = np.around(np.std(np.array(all_mcc)), 4)
print()
print("MEAN AUC:{} ± {}".format(mean_auc, std_auc))
print("MEAN AUPR:{} ± {}".format(mean_aupr, std_aupr))
print("MEAN ACCURACY:{} ± {}".format(mean_accuracy, std_accuracy))
print("MEAN PRECISION:{} ± {}".format(mean_precision, std_precision))
print("MEAN RECALL:{} ± {}".format(mean_recall, std_recall))
print("MEAN F1_SCORE:{} ± {}".format(mean_f1, std_f1))
print("MEAN MCC:{} ± {}".format(mean_mcc, std_mcc))
end_time = timeit.default_timer()
print("Running time: %s Seconds" % (end_time - start_time))

# 绘制ROC、PR曲线
draw_ROC_curve(FPR, TPR)
draw_PR_curve(test_label_all, test_predict_prob_all)

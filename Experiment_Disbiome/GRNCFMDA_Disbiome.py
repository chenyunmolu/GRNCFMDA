import argparse
import os
import random
import timeit
import warnings

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, \
    roc_curve, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import optim

from NeuCF import setup_seed, NeuCF, get_negative_sample_by_KMeans_and_cosine_distances, \
    draw_ROC_curve, draw_PR_curve, data_toExcel

start_time = timeit.default_timer()
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置模型运行参数
parser = argparse.ArgumentParser(description='GRNCFMDA')
# data source params
parser.add_argument('--dataset', type=str, default='Disbiome', choices=['HMDAD', 'Disbiome'], help='Name of dataset.')
# training params
parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 reg.')
parser.add_argument('--seed', type=int, default=36, help='Random seed.')
# NeuCF model params
parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
parser.add_argument("--layers", nargs='+', default=[512, 256, 128],
                    help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")

args = parser.parse_args()

dataset = args.dataset
print("dataset:", dataset)
seed = args.seed
setup_seed(seed)

# 读取微生物和疾病的相似性融合矩阵，以及微生物和疾病关联矩阵
MD_association_matrix = pd.read_csv('../Dataset/Disbiome/mircobe_disease_association_matrix.csv', index_col=0)
microbe_similarity_fusion_matrix = pd.read_csv('../Dataset/Disbiome/microbe_similarity_fusion_matrix.csv', index_col=0)
disease_similarity_fusion_matrix = pd.read_csv('../Dataset/Disbiome/disease_similarity_fusion_matrix.csv', index_col=0)

MD = np.array(MD_association_matrix)
MM = np.array(microbe_similarity_fusion_matrix)
DD = np.array(disease_similarity_fusion_matrix)

mic_nums = MD.shape[0]
dis_nums = MD.shape[1]

feat_outputs = pd.read_csv("feat_outputs_dataframe.csv", index_col=0)
feat_outputs = np.array(feat_outputs)
scaler = StandardScaler()
feat_outputs = scaler.fit_transform(feat_outputs)

# 根据feat_outputs生成复杂的特征向量
features_embedding_mic = feat_outputs[0:mic_nums, :]
features_embedding_dis = feat_outputs[mic_nums:feat_outputs.shape[0], :]
# 根据标签，将mic向量和dis向量进行拼接，生成新的特征向量
positive_index_tuple = np.where(MD == 1)
positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
all_features_input = []
all_label = []
for (r, d) in positive_index_list:
    all_features_input.append(np.hstack((features_embedding_mic[r, :], features_embedding_dis[d, :])))
    all_label.append(1)

negative_index_tuple = np.where(MD == 0)
negative_index_list = list(zip(negative_index_tuple[0], negative_index_tuple[1]))
NEGATIVE_SAMPLE_CHA_ALL = []
NEGATIVE_SAMPLE_CHA_LABEL_ALL = []
for (r, d) in negative_index_list:
    NEGATIVE_SAMPLE_CHA_ALL.append(np.hstack((features_embedding_mic[r, :], features_embedding_dis[d, :])))
    NEGATIVE_SAMPLE_CHA_LABEL_ALL.append(0)
# 使用KMeans聚类方法选择最佳的负样本
NEGATIVE_SAMPLE_CHA_ALL = np.array(NEGATIVE_SAMPLE_CHA_ALL)
NEGATIVE_SAMPLE_CHA, NEGATIVE_SAMPLE_CHA_LABEL = get_negative_sample_by_KMeans_and_cosine_distances(
    NEGATIVE_SAMPLE_CHA_ALL,
    len(positive_index_list))
all_features_input = np.array(all_features_input)
all_features_input = np.vstack([all_features_input, NEGATIVE_SAMPLE_CHA])
all_label = np.array(all_label)
all_label = np.hstack([all_label, NEGATIVE_SAMPLE_CHA_LABEL])

print("----------------------------------------------五折交叉验证----------------------------------------------------")
all_auc = []
all_aupr = []
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_mcc = []
# 用于绘制ROC、PR曲线的参数列表
FPR = []
TPR = []
PRECISION = []
RECALL = []
test_label_all = []
test_predict_prob_all = []
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

count = 0
for train_index, test_index in kfold.split(all_features_input):
    count = count + 1
    print(
        f"-------------------------------------------五折交叉验证：第 {count} 折-----------------------------------------------")
    train_features_input, train_label = all_features_input[train_index], all_label[train_index]
    test_features_input, test_label = all_features_input[test_index], all_label[test_index]

    scaler = StandardScaler()
    train_features_input = scaler.fit_transform(train_features_input)
    test_features_input = scaler.transform(test_features_input)

    train_features_input_tensor = torch.from_numpy(train_features_input).to(torch.float32).to(device)
    test_features_input_tensor = torch.from_numpy(test_features_input).to(torch.float32).to(device)
    train_label_tensor = torch.FloatTensor(train_label).unsqueeze(1).to(device)
    test_label_tensor = torch.FloatTensor(test_label).unsqueeze(1).to(device)

    slice_index = train_features_input_tensor.shape[1] // 2

    model = NeuCF(args, mic_nums, dis_nums).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 模型训练
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        train_prediction = model(train_features_input_tensor[:, :slice_index],
                                 train_features_input_tensor[:, slice_index:])
        loss = F.binary_cross_entropy(train_prediction, train_label_tensor)
        loss.backward()
        optimizer.step()
        print('Epoch: {:04d}, loss: {:.4f}'.format(epoch + 1, loss.item()))

    # 模型评估
    model.eval()
    with torch.no_grad():
        test_prediction = model(test_features_input_tensor[:, :slice_index],
                                test_features_input_tensor[:, slice_index:])

    test_predict_prob = test_prediction.cpu().detach().numpy()
    test_label = test_label_tensor.cpu().detach().numpy()
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
    data_toExcel(fpr, tpr, filepath + "AUC_%.4f.xlsx" % (auc), "%s_AUC" % dataset)
    data_toExcel(rec, pre, filepath + "AUPR_%.4f.xlsx" % (aupr), "%s_AUPR" % dataset)

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
draw_ROC_curve(FPR, TPR, dataset)
draw_PR_curve(test_label_all, test_predict_prob_all, dataset)

import argparse
import random
import timeit
import warnings

import dgl
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch import nn, optim
import torch as th
import torch.nn.functional as F
import dgl.function as fn

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置模型运行参数
parser = argparse.ArgumentParser(description='GRAND')
# data source params
parser.add_argument('--dataset', type=str, default='Disbiome', choices=['HMDAD', 'Disbiome'], help='Name of dataset.')
# training params
parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7, help='L2 reg.')
parser.add_argument('--seed', type=int, default=36, help='Random seed.')
# GRAND model params
parser.add_argument('--init_dim', type=int, default=128, help='Initialize embedding dimension')
parser.add_argument('--dropnode_rate', type=float, default=0.5, help='Dropnode rate (1 - keep probability).')
parser.add_argument('--order', type=int, default=8, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Coefficient of consistency regularization')

parser.add_argument('--features_embedding_size', type=int, default=256, help='Features embedding size')
parser.add_argument('--drop_rate', type=float, default=0.1, help='Drop rate.')


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


class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size  # 指定嵌入大小和丢弃率
        self.drop_rate = drop_rate

        def init_weights(m):  # 初始化模型的权重
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),  # //表示整数除法操作符，表示对两个数进行除法操作，然后取结果的整数部分，以确保结果是整数
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        ).to(device)
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result


def drop_node(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = th.FloatTensor(np.ones(n) * drop_rate)

    if training:
        # 生成一个与 drop_rates 形状相同的张量，其中的值为 0 或 1，表示每个特征是否被保留。这里，1 表示保留该特征，0 表示丢弃该特征。
        masks = th.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats

    else:
        feats = feats * (1. - drop_rate)

    return feats


class GATLayer(nn.Module):
    def __init__(self, G, in_dim, out_dim):
        super(GATLayer, self).__init__()

        # self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        # self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        self.G = G
        self.slope = 0.2
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.m_fc = nn.Linear(292, in_dim, bias=False)
        self.d_fc = nn.Linear(39, in_dim, bias=False)
        self.dropout = nn.Dropout(0.5)
        # self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # print('SRC size:', edges.src['z'].size())
        # print('DST size: ', edges.dst['z'].size())
        # z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # a = self.attn_fc(z2)
        # return {'e': a}
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': F.elu(h)}

    def forward(self, h):
        z = self.fc(h)
        self.G.ndata['z'] = z

        self.G.apply_edges(self.edge_attention)
        self.G.update_all(self.message_func, self.reduce_func)

        return self.G.ndata.pop('h')


# 此代码的实现相当于一个简化的 GCN 模型，其中 y 累积了多阶卷积的节点特征，最终的 y 可以用于进一步的任务（如分类或回归）。
def GRANDConv(graph, feats, order):
    '''
    Parameters
    -----------
    graph: dgl.Graph
        The input graph
    feats: Tensor (n_nodes * feat_dim)
        Node features
    order: int
        Propagation Steps
    '''
    with graph.local_scope():
        ''' Calculate Symmetric normalized adjacency matrix   \hat{A} '''
        degs = graph.in_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5).to(feats.device).unsqueeze(1)

        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))

        ''' Graph Conv '''
        x = feats
        y = 0 + feats
        # 循环控制图卷积的阶数，即在图上传播多少步。order 表示图卷积的次数（或阶数）。较高的阶数会考虑更远的邻居。
        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

    return y / (order + 1)


class GRAND(nn.Module):
    def __init__(self,
                 graph,
                 in_micfeat_size, in_disfeat_size,
                 in_dim,
                 S,
                 K,
                 node_dropout,
                 features_embedding_size,
                 drop_rate):

        super(GRAND, self).__init__()
        self.graph = graph
        self.in_micfeat_size = in_micfeat_size
        self.in_disfeat_size = in_disfeat_size
        self.in_dim = in_dim
        self.S = S
        self.K = K
        self.dropout = node_dropout

        self.features_embedding_size = features_embedding_size
        self.drop_rate = drop_rate
        # 图注意层（多头）
        # self.att_layer = GATv2Conv(self.in_dim, self.in_dim, 1, 0.1, 0.1, 0.3)

        # 定义投影算子
        self.W_mic = nn.Parameter(torch.zeros(size=(self.in_micfeat_size, self.in_dim)))
        self.W_dis = nn.Parameter(torch.zeros(size=(self.in_disfeat_size, self.in_dim)))
        # 初始化投影算子，尾部的_表示"in-place"（原地操作）即：修改原值
        nn.init.xavier_uniform_(self.W_mic.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_dis.data, gain=1.414)

        self.gat = GATLayer(self.graph, self.in_dim, self.in_dim)

        # MLP
        self.mlp_prediction = MLP(self.features_embedding_size, self.drop_rate)

    def forward(self, graph, mic_feature_tensor, dis_feature_tensor, association_matrix, training=True):
        # print("----------------------------------将microbe和disease映射到同一维度----------------------------------")
        mic_mic_f = mic_feature_tensor.mm(self.W_mic)
        dis_dis_f = dis_feature_tensor.mm(self.W_dis)
        init_feats = torch.cat((mic_mic_f, dis_dis_f), dim=0)
        # X:(331,128)
        X = torch.cat((mic_mic_f, dis_dis_f), dim=0)
        S = self.S
        # print("----------------------------------使用GRAND进行特征提取----------------------------------")
        if training:  # Training Mode
            output_list = []
            labels = []
            for s in range(S):
                drop_feat = drop_node(X, self.dropout, True)  # Drop node
                feat = GRANDConv(graph, drop_feat, self.K)  # Graph Convolution
                feat = self.gat(feat)
                feat_outputs = th.cat([feat, init_feats], dim=1)

                # print("----------------------------------根据feat_outputs,生成用于训练的正负样本----------------------------------")
                mic_nums = association_matrix.size()[0]
                features_embedding_mic = feat_outputs[0:mic_nums, :]  # 对特征矩阵进行切片操作，将其分为两大部分
                features_embedding_dis = feat_outputs[mic_nums:feat_outputs.size()[0], :]
                train_features_input, train_lable = [], []
                # positive position index
                positive_index_tuple = torch.where(association_matrix == 1)
                positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))

                for (r, d) in positive_index_list:
                    # positive samples
                    # 将正样本的特征乘积结果作为输入，添加到train_features_input列表中。
                    train_features_input.append(
                        (features_embedding_mic[r, :] * features_embedding_dis[d, :]).unsqueeze(0))
                    # 将标签值1添加到train_lable列表中
                    train_lable.append(1)
                # negative samples
                # 接下来的代码块处理负样本
                negative_index_tuple = torch.where(association_matrix == 0)
                negative_index_list_temp = list(zip(negative_index_tuple[0], negative_index_tuple[1]))

                negative_index_list = random.sample(negative_index_list_temp, len(positive_index_list))
                for (r, d) in negative_index_list:
                    train_features_input.append(
                        (features_embedding_mic[r, :] * features_embedding_dis[d, :]).unsqueeze(0))
                    # 将标签值1添加到train_lable列表中
                    train_lable.append(0)
                # 将训练数据列表和标签列表转换为tensor，以便后续操作
                train_features_input = torch.cat(train_features_input, dim=0).to(device)
                train_lable = torch.FloatTensor(np.array(train_lable)).unsqueeze(1).to(device)
                train_mlp_result = self.mlp_prediction(train_features_input)

                output_list.append(train_mlp_result)
                labels.append(train_lable)

            return output_list, labels
        else:  # Inference Mode
            drop_feat = drop_node(X, self.dropout, False)
            feat = GRANDConv(graph, drop_feat, self.K)
            feat0 = self.gat(feat)
            feat_outputs = th.cat([feat0, init_feats], dim=1)

            mic_nums, dis_nums = association_matrix.size()[0], association_matrix.size()[1]
            features_embedding_mic = feat_outputs[0:mic_nums, :]  # 对特征矩阵进行切片操作，将其分为两大部分
            features_embedding_dis = feat_outputs[mic_nums:feat_outputs.size()[0], :]
            test_features_input, test_lable = [], []
            for i in range(mic_nums):
                for j in range(dis_nums):
                    test_features_input.append(
                        (features_embedding_mic[i, :] * features_embedding_dis[j, :]).unsqueeze(0))
                    test_lable.append(association_matrix[i, j].item())  # 将tensor类型转为float，因为np.array函数无法接受tensor

            test_features_input = torch.cat(test_features_input, dim=0).to(device)
            test_lable = torch.FloatTensor(np.array(test_lable)).unsqueeze(1).to(device)
            test_mlp_result = self.mlp_prediction(test_features_input)

            return test_mlp_result, test_lable, feat_outputs


if __name__ == '__main__':
    start_time = timeit.default_timer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    dataset = args.dataset
    print("dataset:", dataset)
    print("---------------------------------->利用GRAND进行特征提取<----------------------------------")
    seed = args.seed
    # 设置随机数，保证结果的可复现性
    setup_seed(seed)

    MD_association_matrix = pd.read_csv("../Dataset/Disbiome/mircobe_disease_association_matrix.csv", index_col=0)
    microbe_similarity_fusion_matrix = pd.read_csv("../Dataset/Disbiome/microbe_similarity_fusion_matrix.csv", index_col=0)
    disease_similarity_fusion_matrix = pd.read_csv("../Dataset/Disbiome/disease_similarity_fusion_matrix.csv", index_col=0)

    mic_nums = MD_association_matrix.shape[0]
    dis_nums = MD_association_matrix.shape[1]

    MD = np.array(MD_association_matrix)
    MM = np.array(microbe_similarity_fusion_matrix)
    DD = np.array(disease_similarity_fusion_matrix)

    g = build_heterograph(MD, MM, DD).to(device)

    MM_tensor = torch.from_numpy(MM).to(torch.float32).to(device)
    DD_tensor = torch.from_numpy(DD).to(torch.float32).to(device)
    MD_tensor = torch.from_numpy(MD).to(torch.float32).to(device)

    model = GRAND(g, mic_nums, dis_nums, args.init_dim, args.sample, args.order, args.dropnode_rate,
                  args.features_embedding_size, args.drop_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = args.epochs
    # 模型训练
    model.train()
    for epoch in range(epochs):
        # print("----------------------------------start {} training----------------------------------".format(epoch + 1))
        loss_sup = 0
        train_predict_result, train_labels = model(g, MM_tensor, DD_tensor, MD_tensor, training=True)
        # binary_cross_entropy：二元交叉熵损失函数，通常用于二分类问题中的神经网络训练，该函数可以根据train_predict_result自行计算类别索引，然后计算loss
        for k in range(args.sample):
            loss_sup += F.binary_cross_entropy(train_predict_result[k], train_labels[k])
        loss_sup = loss_sup / args.sample
        loss_consis = consis_loss(train_predict_result, args.tem, args.lam)
        loss_train = loss_sup + loss_consis
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        print(
            'Epoch %d | train Loss: %.4f' % (epoch + 1, loss_train.item()))  # 格式化字符串，两个占位符‘%d’、'%.4f',分别表示整数和保留4位小数的浮点数

    # 模型评估
    model.eval()
    with torch.no_grad():
        test_predict_result, test_lable, feat_outputs = model(g, MM_tensor, DD_tensor, MD_tensor, training=False)

    cnn_outputs = feat_outputs.cpu().detach().numpy()
    cnn_outputs_dataframe = pd.DataFrame(cnn_outputs)
    cnn_outputs_dataframe.to_csv("feat_outputs_dataframe.csv")

    end_time = timeit.default_timer()
    print("Execution Time: ", end_time - start_time)

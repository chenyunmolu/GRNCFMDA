import numpy as np
import torch
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

from utils import train_feature_choose, test_feature_choose

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class NeuCF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(NeuCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp = int(args.layers[0] / 2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=args.layers[1] + args.layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):

        mlp_vector = torch.cat([user_indices, item_indices], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(user_indices, item_indices)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating


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
                 in_micfeat_size, in_disfeat_size,
                 in_dim,
                 n_class,
                 args,
                 S=1,
                 K=3,
                 node_dropout=0.0):

        super(GRAND, self).__init__()
        self.in_micfeat_size = in_micfeat_size
        self.in_disfeat_size = in_disfeat_size
        self.in_dim = in_dim
        self.n_class = n_class
        self.S = S
        self.K = K
        self.dropout = node_dropout

        # 定义投影算子
        self.W_mic = nn.Parameter(torch.zeros(size=(self.in_micfeat_size, self.in_dim)))
        self.W_dis = nn.Parameter(torch.zeros(size=(self.in_disfeat_size, self.in_dim)))
        # 初始化投影算子，尾部的_表示"in-place"（原地操作）即：修改原值
        nn.init.xavier_uniform_(self.W_mic.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_dis.data, gain=1.414)
        self.neucf = NeuCF(args, self.in_micfeat_size, self.in_disfeat_size)
        self.node_dropout = nn.Dropout(node_dropout)

    def forward(self, graph, mic_feature_tensor, dis_feature_tensor, rel_matrix, training=True):
        mic_mic_f = mic_feature_tensor.mm(self.W_mic)
        dis_dis_f = dis_feature_tensor.mm(self.W_dis)
        # X:(331,128)
        X = torch.cat((mic_mic_f, dis_dis_f), dim=0)
        S = self.S

        if training:  # Training Mode
            output_list = []
            labels = []
            for s in range(S):
                drop_feat = drop_node(X, self.dropout, True)  # Drop node
                feat = GRANDConv(graph, drop_feat, self.K)  # Graph Convolution
                train_mic_feature_input, train_dis_feature_input, train_label = train_feature_choose(rel_matrix, feat)
                train_prediction_result = self.neucf(train_mic_feature_input, train_dis_feature_input)
                output_list.append(train_prediction_result)  # Prediction
                labels.append(train_label)
            return output_list, labels
        else:  # Inference Mode
            drop_feat = drop_node(X, self.dropout, False)
            X = GRANDConv(graph, drop_feat, self.K)
            test_mic_feature_input, test_dis_feature_input, test_label = test_feature_choose(rel_matrix, X)
            test_prediction_result = self.neucf(test_mic_feature_input, test_dis_feature_input)
            return test_prediction_result, test_label

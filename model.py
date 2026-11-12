import datetime
import math
import numpy as np
import torch
import random
from torch import cosine_embedding_loss, nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
import pickle
import os
import torch
import torch.nn as nn
from models.model_in_diffusion import *
from tqdm import tqdm

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, emb_size, n_node):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.n_node = n_node
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.w_co = nn.Linear(self.emb_size, self.emb_size)

        self.gate_w1 = nn.Linear(self.emb_size * 2, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def forward(self, adjacency, embedding, mo_embedding):
        id_embeddings = self.dropout30(embedding)
        mo_embedding = self.dropout30(mo_embedding)

        for i in range(self.layers):
            item_co_emb = self.get_embedding(adjacency, id_embeddings)
            id_embeddings = id_embeddings + item_co_emb
            item_mo_emb = self.get_embedding(adjacency, mo_embedding)
            mo_embedding = mo_embedding + item_mo_emb
        results = id_embeddings
        mo_results = mo_embedding
        return results, mo_results

    def get_embedding(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        # item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embedding)
        adjacency = trans_to_cuda(adjacency)
        # item_embeddings = torch.sparse.mm(adjacency, trans_to_cuda(embedding))
        item_embeddings = torch.mm(adjacency.to_dense(), trans_to_cuda(embedding))
        return item_embeddings


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



class Diff(Module):
    def __init__(self, opt, out_dims, in_dims,norm, n_node, adjacency, lr, co_layer, l2, item_beta, pro_beta, cri_beta, dataset, num_heads=4,
                 emb_size=100, text_emb_size=100, batch_size=100, num_negatives=100, neighbor_n=3):
        super(Diff, self).__init__()
        self.emb_size = emb_size
        self.text_emb_size = text_emb_size
        self.hidden_size = 300
        self.batch_size = batch_size
        self.n_node = n_node
        self.out_dims = out_dims
        self.in_dims = in_dims
        self.norm = norm
        self.L2 = l2
        self.lr = lr
        self.co_layer = co_layer
        self.item_beta = item_beta
        self.pro_beta = pro_beta
        self.cri_beta = cri_beta
        self.num_negatives = num_negatives
        self.num_heads = num_heads
        self.neighbor_n = neighbor_n  # 邻居会话数量
        # 保存超参数用于损失计算
        self.opt = opt
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.match_loss = MatchLoss(temperature=0.3)
        self.n_target = self.neighbor_n  # 使用统一的邻居数量
        self.w_ne = 1.7
        self.w = 20
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.adjacency = adjacency
        self.HyperGraph = HyperConv(self.co_layer, self.emb_size, self.n_node)
        # 使用绝对路径，确保在任何目录都能运行
        script_dir = os.path.dirname(os.path.abspath(__file__))
        text_path = os.path.join(script_dir, 'datasets', dataset, 'textMatrixpca100.npy')
        textWeights = np.load(text_path)
        self.text_embedding = nn.Embedding(self.n_node, text_emb_size)
        
        text_pre_weight = np.array(textWeights)
        self.text_embedding.weight.data.copy_(torch.from_numpy(text_pre_weight))
        self.m1 = 0.05 
        self.m2 = 0.25
        self.max_norm = 1.0
        self.per_emb_k = int(getattr(opt, "per_emb_k", 3))  # 你可在 args 里加 --per_sess_k 4
        self.sess_id_proto = nn.Linear(self.emb_size, self.per_emb_k * self.emb_size)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)
        self.embed_dropout = nn.Dropout(0.3)
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.sim_w_p1u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p1d = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p2u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_p2d = nn.Linear(self.emb_size, self.emb_size)

        self.sim_w_c1u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c2u = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c1d = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_c2d = nn.Linear(self.emb_size, self.emb_size)

        self.sim_w_i1 = nn.Linear(self.emb_size, self.emb_size)
        self.sim_w_i2 = nn.Linear(self.emb_size, self.emb_size)

        self.w_2 = nn.Linear(self.emb_size, 1)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)


        self.linear_one = nn.Linear(self.emb_size, self.emb_size)
        self.linear_two = nn.Linear(self.emb_size, self.emb_size)
        self.linear_three = nn.Linear(self.emb_size, self.emb_size)
        self.linear_transform = nn.Linear(2 * self.emb_size, self.emb_size)
        self.linear_one1 = nn.Linear(self.emb_size, self.emb_size)
        self.linear_two1 = nn.Linear(self.emb_size, self.emb_size)
        self.linear_three1 = nn.Linear(self.emb_size, self.emb_size)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # 优化：让原始嵌入权重更大，扩散结果作为补充
        self.channels = 5
        self.c_dim = self.emb_size // self.channels
        self.in_dim = self.emb_size
        self.c_dims_hidden = self.emb_size // self.channels
        self.linear_o = nn.Linear(self.c_dims_hidden, self.c_dims_hidden, bias=True)
        self.linear_tw = nn.Linear(self.c_dims_hidden, self.c_dims_hidden, bias=True)
        self.linear_th = nn.Linear(self.c_dims_hidden, 1, bias=False)
        self.linear_tran = nn.Linear(self.c_dims_hidden * 2, self.c_dims_hidden, bias=True)
        self.cor_weight = getattr(opt, 'cor_weight', 5)  # 添加相关性损失权重

        self.cor_loss = Cor_loss(self.cor_weight, self.channels, self.emb_size, self.c_dim)
        self.weight_list = []
        self.bias_list = []
        self.weight_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(self.in_dim, self.c_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.bias_list = nn.ParameterList(
            nn.Parameter(torch.empty(size=(1, self.c_dim), dtype=torch.float), requires_grad=True) for i in
            range(self.channels))
        self.diversity_fusion = getattr(opt, 'diversity_fusion', 'concat')
        self.fusion_weights = nn.Parameter(torch.tensor([3.0, 1.0]))
        self.moe_denoise_network = MoE_Denoise_Network(in_dims=self.in_dims, out_dims=self.out_dims, emb_size=self.emb_size, time_type="cat", norm=self.norm, dropout=0.5)
        # 使用 UNet 作为降噪网络（已支持动态维度）
        self.dnn = DNN(in_dims=self.in_dims, out_dims=self.out_dims, emb_size=self.emb_size, time_type="cat", norm=self.norm, dropout=0.5)
        self.transformer_dnn = TransformerDNN(in_dims=self.in_dims, out_dims=self.out_dims, emb_size=self.emb_size, time_type="cat", norm=self.norm, dropout=0.5)
        self.mlp = MLP_Denoise_Network(in_dims=self.in_dims, out_dims=self.out_dims, emb_size=self.emb_size, time_type="cat", norm=self.norm, dropout=0.5)
        self.unet = UNet(in_dims=self.in_dims, out_dims=self.out_dims, emb_size=self.emb_size, time_type="cat", norm=self.norm, dropout=0.5)
        self.loss_function = nn.CrossEntropyLoss()
        self.fusion_weights = nn.Parameter(torch.tensor([3.0, 1.0]))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, diffusion, item_embedding, text_embedding, session_item, session_len,
                          reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        
        item_emb_table = torch.cat([zeros, item_embedding], 0)
        text_emb_table = torch.cat([zeros, text_embedding], 0)
        mask = mask.float().unsqueeze(-1)
        mask = (session_item != 0).int()
        get = lambda i: item_emb_table[session_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        
        cor_id_hidden=trans_to_cuda(torch.tensor([]))
        for i in range(len(self.route_embedding(seq_h))):
            cor_id_hidden=torch.cat((cor_id_hidden,self.route_embedding(seq_h)[i]),2)
        get_text = lambda i: text_emb_table[session_item[i]]
        seq_text = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_text[i] = get_text(i)
        cor_mo_hidden=trans_to_cuda(torch.tensor([]))
        for i in range(len(self.route_embedding(seq_text))):
            cor_mo_hidden=torch.cat((cor_mo_hidden,self.route_embedding(seq_text)[i]),2)
        
        seq_channels = torch.split(seq_h, 20, dim=2)
        sess_embedding_id_list = []
        for seq_channel in (seq_channels):
            ht = seq_channel[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
            q1 = self.linear_o(ht).view(ht.shape[0], 1, ht.shape[1])
            q2 = self.linear_tw(seq_channel)
            alpha = self.linear_th(torch.sigmoid(q1 + q2))
            sess_embedding = torch.sum(alpha * seq_channel * mask.view(mask.shape[0], -1, 1).float(), 1)
            sess_embedding_id_list.append(sess_embedding)
        sess_id_emb = torch.cat(sess_embedding_id_list, dim=1)
        seq_text_channels = torch.split(seq_text, 20, dim=2)
        sess_embedding_mo_list = []
        for seq_text_channel in (seq_text_channels):
            ht1 = seq_text_channel[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
            q11 = self.linear_o(ht1).view(ht1.shape[0], 1, ht1.shape[1])
            q21 = self.linear_tw(seq_text_channel)
            alpha1 = self.linear_th(torch.sigmoid(q11 + q21))
            sess_embedding_mo = torch.sum(alpha1 * seq_text_channel * mask.view(mask.shape[0], -1, 1).float(), 1)
            sess_embedding_mo_list.append(sess_embedding_mo)
        sess_mo_emb = torch.cat(sess_embedding_mo_list, dim=1)
        return sess_id_emb, sess_mo_emb, cor_id_hidden, cor_mo_hidden


    

    def _create_distance_correlation(self,x,y):
        zero = trans_to_cuda(torch.tensor(0,dtype=float))
        def _create_centered_distance(X,zero):
            r = torch.sum(torch.square(X),1,keepdim=True)
            X_t = torch.transpose(X,1,0)
            r_t = torch.transpose(r,1,0)
            D = torch.sqrt(torch.maximum(r-2*torch.matmul(X,X_t)+r_t,zero)+1e-8)
            D = D - torch.mean(D,dim=0,keepdim=True)-torch.mean(D,dim=1,keepdim=True)+torch.mean(D)
            return D

        def _create_distance_covariance(D1,D2,zero):
            n_samples = D1.shape[0]
            n_samples = torch.tensor(n_samples,dtype=torch.float)
            sum = torch.sum(D1*D2)
            sum = torch.div(sum,n_samples*n_samples)
            dcov=torch.sqrt(torch.maximum(sum,zero)+1e-8)
            return dcov

        D1 = _create_centered_distance(x,zero)
        D2 = _create_centered_distance(y,zero)

        dcov_12 = _create_distance_covariance(D1, D2,zero)
        dcov_11 = _create_distance_covariance(D1, D1,zero)
        dcov_22 = _create_distance_covariance(D2, D2,zero)

        dcor = torch.sqrt(torch.maximum(dcov_11 * dcov_22, zero))+1e-10
        dcor = torch.div(dcov_12,dcor)

        return dcor

    def route_embedding(self, embedding):
        c_features = []
        for i in range(self.channels): 
            # 将原始 embedding 投影到第 i 个通道（Factor）的子空间
            z = torch.matmul(embedding, self.weight_list[i]) + self.bias_list[i]
            z = F.normalize(z, dim=1) # 归一化
            c_features.append(z)
        return c_features

    def forward(self, diffusion, session_item, flag, session_len, reversed_sess_item, mask, target):
        item_emb = trans_to_cuda(self.embedding.weight)
        item_emb = self.embed_dropout(item_emb)
        text_emb = trans_to_cuda(self.text_embedding.weight)
        item_emb, text_emb= self.HyperGraph(self.adjacency, item_emb, text_emb)
        inter_loss = self.match_loss(item_emb, text_emb)
        sess_id_emb, sess_mo_emb,cor_id_hidden, cor_mo_hidden= self.generate_sess_emb(diffusion, item_emb,text_emb,session_item, session_len, reversed_sess_item,mask)
        return item_emb, text_emb, sess_id_emb, sess_mo_emb,cor_id_hidden,cor_mo_hidden,inter_loss

class Cor_loss(Module):
    def __init__(self,cor_weight,channels,hidden_size,channel_size):
        super(Cor_loss, self).__init__()
        self.channel_num = channels
        self.cor_weight =cor_weight
        self.hidden_size = hidden_size
        self.channel_size = channel_size

    def forward(self,embedding):#解耦任意factor对
        if self.cor_weight==0:
            return 0
        else:
            embedding = embedding.view(-1, self.hidden_size)
            embedding_weight = torch.chunk(embedding, self.channel_num, dim=1)
            cor_loss = torch.tensor(0,dtype = torch.float)
            for i in range(self.channel_num):
                for j in range(i+1,self.channel_num):
                    x=embedding_weight[i]
                    y=embedding_weight[j]
                    cor_loss = cor_loss+self._create_distance_correlation(x, y)
            b=  (self.channel_num+1.0)* self.channel_num/2
            cor_loss = self.cor_weight * torch.div(cor_loss,b)
        return cor_loss
    def _create_distance_correlation(self,x,y):
        zero = trans_to_cuda(torch.tensor(0,dtype=float))
        def _create_centered_distance(X,zero):
            r = torch.sum(torch.square(X),1,keepdim=True)
            X_t = torch.transpose(X,1,0)
            r_t = torch.transpose(r,1,0)
            D = torch.sqrt(torch.maximum(r-2*torch.matmul(X,X_t)+r_t,zero)+1e-8)
            D = D - torch.mean(D,dim=0,keepdim=True)-torch.mean(D,dim=1,keepdim=True)+torch.mean(D)
            return D

        def _create_distance_covariance(D1,D2,zero):
            n_samples = D1.shape[0]
            n_samples = torch.tensor(n_samples,dtype=torch.float)
            sum = torch.sum(D1*D2)
            sum = torch.div(sum,n_samples*n_samples)
            dcov=torch.sqrt(torch.maximum(sum,zero)+1e-8)
            return dcov

        D1 = _create_centered_distance(x,zero)
        D2 = _create_centered_distance(y,zero)

        dcov_12 = _create_distance_covariance(D1, D2,zero)
        dcov_11 = _create_distance_covariance(D1, D1,zero)
        dcov_22 = _create_distance_covariance(D2, D2,zero)

        dcor = torch.sqrt(torch.maximum(dcov_11 * dcov_22, zero))+1e-10
        dcor = torch.div(dcov_12,dcor)
        return dcor
class MatchLoss(nn.Module):
    def __init__(self, temperature=0.3):
        super().__init__()
        self.T = temperature

    def forward(self, feature_left, feature_right):
        id_features = F.normalize(feature_left, dim=1)
        modal_features = F.normalize(feature_right, dim=1)
        similarity = torch.mm(id_features, modal_features.t()) / self.T
        exp_similarity = torch.exp(similarity)
        positive_sim = torch.diag(exp_similarity)
        denominator = exp_similarity.sum(dim=1)
        loss = -torch.log(positive_sim / denominator)
        return loss.mean()


def perform(model, diffusion, i, data):
    tar, flag, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    flag = trans_to_cuda(torch.Tensor(flag).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb, cor_id_hidden, cor_mo_hidden,inter_loss  = model(diffusion,session_item, flag, session_len, reversed_sess_item, mask,tar)
    loss_cor1 = model.cor_loss(cor_id_hidden)
    loss_cor2 = model.cor_loss(cor_mo_hidden)
    diff_id_emb = diffusion.training_losses1(model.moe_denoise_network,sess_id_emb,reweight=True)
    diff_mo_emb = diffusion.training_losses1(model.moe_denoise_network,sess_mo_emb,reweight=True)
    loss_id = diffusion.get_reconstruct_loss(sess_id_emb,diff_id_emb)
    loss_mo = diffusion.get_reconstruct_loss(sess_mo_emb,diff_mo_emb)
    loss_id = loss_id.mean()
    loss_mo = loss_mo.mean()
    normalized_weights = torch.softmax(model.fusion_weights, dim=0)
    sess_id_emb = normalized_weights[0]*sess_id_emb + normalized_weights[1]*diff_id_emb
    sess_mo_emb = normalized_weights[0]*sess_mo_emb + normalized_weights[1]*diff_mo_emb
    score_id = torch.mm(sess_id_emb, item_id_emb.transpose(1, 0))
    score_mo = torch.mm(sess_mo_emb, item_mo_emb.transpose(1, 0))
    scores = score_id + 0.5*score_mo
    return tar, scores, inter_loss, loss_id, loss_mo, loss_cor1, loss_cor2


def perform_infer(model, diffusion, i, data):
    tar, flag, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    flag = trans_to_cuda(torch.Tensor(flag).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb,_,_,_ = model(diffusion,session_item, flag, session_len, reversed_sess_item, mask,tar)
    diff_id_emb = diffusion.p_sample(model.moe_denoise_network, sess_id_emb, 32, False)
    diff_mo_emb = diffusion.p_sample(model.moe_denoise_network, sess_mo_emb, 32, False)
    normalized_weights = torch.softmax(model.fusion_weights, dim=0)
    sess_id_emb = normalized_weights[0]*sess_id_emb + normalized_weights[1]*diff_id_emb
    sess_mo_emb = normalized_weights[0]*sess_mo_emb + normalized_weights[1]*diff_mo_emb
    score_id = torch.mm(sess_id_emb, item_id_emb.transpose(1, 0))
    score_mo = torch.mm(sess_mo_emb, item_mo_emb.transpose(1, 0))
    scores = score_id + 0.5*score_mo
    return tar, scores

def train_test(model, diffusion, train_data, test_data):
    model.train()

    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(tqdm(slices), np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores,inter_loss, loss_id, loss_mo,loss_cor1,loss_cor2 = perform(model, diffusion, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss +loss_id*100+loss_mo*100+ 0.01*loss_cor1+0.01*loss_cor2+inter_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics[f'hit{K}'] = []
        metrics[f'mrr{K}'] = []
        metrics[f'ndcg{K}'] = []
        metrics[f'ild{K}'] = []
        metrics[f'hrd{K}'] = []
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    id_feats = model.embedding.weight.detach().cpu().numpy()
    text_feats = model.text_embedding.weight.detach().cpu().numpy()
    if getattr(model, 'diversity_fusion', 'concat') == 'concat':
        item_features = np.concatenate([id_feats, text_feats], axis=1)
    else:
        item_features = (id_feats + text_feats) / 2.0
    slices = test_data.generate_batch(model.batch_size)
    for i in tqdm(slices):
        tar, scores = perform_infer(model,diffusion, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        histories = []
        for idx in i:
            session = test_data.raw[idx]
            history = session[:-1]
            histories.append([int(item) for item in history if item != 0])

        n_items = model.n_node
        max_valid_idx = n_items - 1
        for K in top_K:
            topK_indices = index[:, :K]
            for batch_idx, (preds, target) in enumerate(zip(topK_indices, tar)):
                metrics[f'hit{K}'].append(np.isin(target, preds))
                hit_positions = np.where(preds == target)[0]
                if len(hit_positions) == 0:
                    metrics[f'mrr{K}'].append(0.0)
                    metrics[f'ndcg{K}'].append(0.0)
                else:
                    rank = hit_positions[0]
                    metrics[f'mrr{K}'].append(1.0 / (rank + 1))
                    metrics[f'ndcg{K}'].append(1.0 / np.log2(rank + 2))

                rec_items = [int(it) for it in preds if it != 0 and 0 <= it <= max_valid_idx]
                if len(rec_items) == 0:
                    continue

                if len(rec_items) > 1:
                    pair_dists = []
                    for i_idx in range(len(rec_items)):
                        for j in range(i_idx + 1, len(rec_items)):
                            item_a = rec_items[i_idx]
                            item_b = rec_items[j]
                            mo_embedding_a = item_features[item_a]
                            mo_embedding_b = item_features[item_b]
                            distance = np.linalg.norm(mo_embedding_a - mo_embedding_b)
                            pair_dists.append(distance)
                    ild_value = float(np.mean(pair_dists)) if len(pair_dists) > 0 else 0.0
                else:
                    ild_value = 0.0
                metrics[f'ild{K}'].append(ild_value)

                if batch_idx < len(histories):
                    history = [int(item) for item in histories[batch_idx] if item != 0 and 0 <= item <= max_valid_idx]
                else:
                    history = []

                if len(rec_items) > 0 and len(history) > 0:
                    hist_embeddings = item_features[history]
                    mean_emb = np.mean(hist_embeddings, axis=0)
                    hist_dists = []
                    for item in rec_items:
                        item_emb = item_features[item]
                        distance = np.linalg.norm(item_emb - mean_emb)
                        hist_dists.append(distance)
                    hrd_value = float(np.mean(hist_dists)) if len(hist_dists) > 0 else 0.0
                else:
                    hrd_value = 0.0
                metrics[f'hrd{K}'].append(hrd_value)

    return metrics
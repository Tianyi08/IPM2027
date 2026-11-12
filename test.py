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
        self.per_emb_k = int(getattr(opt, "per_emb_k", 4))  # 你可在 args 里加 --per_emb_k 4
        self.per_sess_k = self.per_emb_k  # 保持兼容性
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
        self.fusion_weights = nn.Parameter(torch.tensor([3.0, 1.0]))
        self.moe_denoise_network = MoE_Denoise_Network(in_dims=self.in_dims, out_dims=self.out_dims, emb_size=self.emb_size, time_type="cat", norm=self.norm, dropout=0.5)
        self.dnn = DNN(in_dims=self.in_dims, out_dims=self.out_dims, emb_size=self.emb_size, time_type="cat", norm=self.norm, dropout=0.5)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, text_embedding, session_item, session_len,
                          reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        mask = mask.float().unsqueeze(-1)
        item_emb_table = torch.cat([zeros, item_embedding], 0)
        text_emb_table = torch.cat([zeros, text_embedding], 0)
        get = lambda i: item_emb_table[session_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        mask = (session_item != 0).int()
        ht = seq_h[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(seq_h)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        sess_embedding = torch.sum(alpha * seq_h * mask.view(mask.shape[0], -1, 1).float(), 1)
        id_sess_pro_emb = torch.div(torch.sum(seq_h, 1), session_len.type(torch.cuda.FloatTensor))
        get_text = lambda i: text_emb_table[session_item[i]]
        seq_text = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_text[i] = get_text(i)
        mo_sess_pro_emb = torch.div(torch.sum(seq_text, 1), session_len.type(torch.cuda.FloatTensor))
        ht1 = seq_text[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q11 = self.linear_one1(ht1).view(ht1.shape[0], 1, ht.shape[1])
        q21 = self.linear_two1(seq_text)
        alpha1 = self.linear_three1(torch.sigmoid(q11 + q21))
        sess_embedding_mo = torch.sum(alpha1 * seq_text * mask.view(mask.shape[0], -1, 1).float(), 1)
        return sess_embedding, sess_embedding_mo, id_sess_pro_emb, mo_sess_pro_emb

    def diversity_loss(self, sess_id_emb):
        sess_id_emb_expanded_1 = sess_id_emb.unsqueeze(1)  # [batch_size, 1, per_emb_k, dim]
        sess_id_emb_expanded_2 = sess_id_emb.unsqueeze(2)  # [batch_size, per_emb_k, 1, dim]
        diversity = torch.square(sess_id_emb_expanded_1 - sess_id_emb_expanded_2).sum(-1)  # [batch_size, per_emb_k, per_emb_k]
        eye_matrix = torch.eye(self.per_emb_k, device=sess_id_emb.device, dtype=sess_id_emb.dtype)  # [per_emb_k, per_emb_k]
        mask11 = (1.0 - eye_matrix).unsqueeze(0)  # [1, per_emb_k, per_emb_k]
        m1_tensor = torch.tensor(self.m1, device=sess_id_emb.device, dtype=sess_id_emb.dtype)
        m2_tensor = torch.tensor(self.m2, device=sess_id_emb.device, dtype=sess_id_emb.dtype)
        term1 = (m1_tensor - diversity) * mask11  # [batch_size, per_emb_k, per_emb_k]
        term2 = (diversity - m2_tensor) * mask11  # [batch_size, per_emb_k, per_emb_k]
        loss1 = F.relu(term1).mean() + F.relu(term2).mean()
        return loss1

    def metric_scores_from_prototypes(self, sess_id_emb, item_id_emb):
        B = sess_id_emb.size(0)
        N = item_id_emb.size(0)
        H = item_id_emb.size(1)
        K = self.per_emb_k

        proto = self.sess_id_proto(sess_id_emb)             # [B, K*H]
        sess_k = proto.view(B, K, H)                        # [B, K, H]
        loss1 = self.diversity_loss(sess_k)
        # dist: [B, K, N]
        dist = (sess_k.unsqueeze(2) - item_id_emb.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)
        # scores: [B, N]
        scores = -dist.min(dim=1).values
        return scores, loss1
    
    def metric_scores_from_prototypes_test(self, sess_id_emb, item_id_emb):
        B = sess_id_emb.size(0)
        N = item_id_emb.size(0)
        H = item_id_emb.size(1)
        K = self.per_emb_k

        proto = self.sess_id_proto(sess_id_emb)             # [B, K*H]
        sess_k = proto.view(B, K, H)                        # [B, K, H]
        # dist: [B, K, N]
        dist = (sess_k.unsqueeze(2) - item_id_emb.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)
        # scores: [B, N]
        scores = -dist.min(dim=1).values
        return scores

    def contra_loss(self,diffu):
        temperature=0.07
        similarities = (F.cosine_similarity(diffu.unsqueeze(1), diffu.unsqueeze(0), dim=2) / temperature).to(diffu.device)
        
        # Positive pairs
        positives = torch.diag(similarities).to(diffu.device)
        
        # Negative pairs
        mask = torch.eye(len(diffu)).bool().to(diffu.device)
        mask = ~mask.to(diffu.device)
        negatives = similarities[mask].view(len(diffu), -1).to(diffu.device)
        
        # Calculate NCE Loss
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(len(diffu), dtype=torch.long).to(diffu.device)
        loss = F.cross_entropy(logits, labels)        
        return loss

    def loss_diffu_ce(self, rep_diffu):
        """
        ### norm scores
        item_emb_norm = F.normalize(self.embedding.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        contra=self.contra_loss(rep_diffu)
        return contra

    def diffu_rep_pre(self, rep_diffu, item_emb,sess_mo_emb,item_mo_emb):
        scores1 = torch.matmul(rep_diffu, torch.transpose(item_emb, 1, 0))
        scores2 = torch.matmul(sess_mo_emb, torch.transpose(item_mo_emb, 1, 0))
        scores = scores1 + scores2
        return scores

    def sim_cal_w(self, emb1, emb2, mat_w):
        sim = emb1 * mat_w(emb2)
        sim = torch.sum(sim, -1)
        return sim

    def transpose_for_scores(self, x, attention_head_size):
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, session_item, flag, session_len, reversed_sess_item, mask, target):
        item_emb = trans_to_cuda(self.embedding.weight)
        item_emb = self.embed_dropout(item_emb)
        text_emb = trans_to_cuda(self.text_embedding.weight)
        item_emb, text_emb= self.HyperGraph(self.adjacency, item_emb, text_emb)
        inter_loss = self.match_loss(item_emb, text_emb)
        sess_id_emb, sess_mo_emb,  sess_id_pro_emb, sess_mo_pro_emb = self.generate_sess_emb(item_emb,text_emb,session_item, session_len, reversed_sess_item,mask)
        return item_emb, text_emb, sess_id_emb, sess_mo_emb,sess_id_pro_emb, sess_mo_pro_emb, inter_loss


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
    item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb,sess_id_pro_emb,sess_mo_pro_emb,inter_loss  = model(session_item, flag, session_len, reversed_sess_item, mask,tar)
    diff_sess_id_emb = diffusion.training_losses1(model.dnn, sess_id_emb, reweight=True)
    diff_sess_mo_emb = diffusion.training_losses1(model.dnn, sess_mo_emb, reweight=True)
    loss_id = model.loss_diffu_ce(diff_sess_id_emb)
    loss_mo = model.loss_diffu_ce(diff_sess_mo_emb)
    normalized_weights = torch.softmax(model.fusion_weights, dim=0)
    sess_id_emb = normalized_weights[0] * sess_id_emb + normalized_weights[1] * diff_sess_id_emb
    sess_mo_emb = normalized_weights[0] * sess_mo_emb + normalized_weights[1] * diff_sess_mo_emb
    scores_id, loss1 = model.metric_scores_from_prototypes(sess_id_emb, item_id_emb)
    scores_mo, loss2 = model.metric_scores_from_prototypes(sess_mo_emb, item_mo_emb)

    # scores_id = torch.mm(sess_id_emb, torch.transpose(item_id_emb, 1, 0))
    # scores_mo = torch.mm(sess_mo_emb, torch.transpose(item_mo_emb, 1, 0))
    scores = scores_id + 0.5*scores_mo
    return tar, scores,loss_id,loss_mo,inter_loss,loss1,loss2


def perform_infer(model, diffusion, i, data):
    tar, flag, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    flag = trans_to_cuda(torch.Tensor(flag).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_id_emb, item_mo_emb, sess_id_emb, sess_mo_emb, sess_id_pro_emb, sess_mo_pro_emb,loss = model(session_item, flag, session_len, reversed_sess_item, mask, tar)
    diff_id = diffusion.p_sample(model.dnn, sess_id_emb, 32, False)
    diff_mo = diffusion.p_sample(model.dnn, sess_mo_emb, 32, False)
    # diff_id_neig_result = diff_id_neig_result.mean(dim=1)
    normalized_weights = torch.softmax(model.fusion_weights, dim=0)
    sess_id_emb = normalized_weights[0] * sess_id_emb + normalized_weights[1] * diff_id
    sess_mo_emb = normalized_weights[0] * sess_mo_emb + normalized_weights[1] * diff_mo
    scores_id = model.metric_scores_from_prototypes_test(sess_id_emb, item_id_emb)
    scores_mo = model.metric_scores_from_prototypes_test(sess_mo_emb, item_mo_emb)
    # scores_id = torch.mm(sess_id_emb, torch.transpose(item_id_emb, 1, 0))
    # scores_mo = torch.mm(sess_mo_emb, torch.transpose(item_mo_emb, 1, 0))
    scores =  scores_id + 0.5*scores_mo
    return tar, scores

def train_test(model, diffusion, train_data, test_data):
    model.train()

    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(tqdm(slices), np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores,loss_id,loss_mo,inter_loss,loss1,loss2 = perform(model, diffusion, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        diff_w = model.opt.diff_weight if hasattr(model.opt, 'diff_weight') else 8.0 
        sess_con_w = model.opt.session_conloss_weight if hasattr(model.opt, 'session_conloss_weight') else 0.05
        loss = loss + loss_id + loss_mo + inter_loss +loss1+loss2 
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
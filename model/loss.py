import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
import math

class ContrastiveLossLecard(nn.Module):
    def __init__(self, config):
        super(ContrastiveLossLecard, self).__init__()
        self.config = config
        self.sim_fct = self.cosine_sim if config.get('contra_loss', 'sim_fct') == 'cos' else self.dot_product
        self.loss_fct = nn.CrossEntropyLoss(reduce=False)
        self.temperature = config.getfloat('contra_loss', 'temperature')
    def forward(self, query_list, candidates_list, labels,sep_idx=None):
        candidate_list_fact=[]
        candidate_list_struct=[]
        for i in range(len(candidates_list)):
            candidate_list_fact.append(candidates_list[i][1])
            candidate_list_struct.append(candidates_list[i][0])
        candidate_struct=torch.cat(candidate_list_struct,dim=0)
        candidate_fact=torch.cat(candidate_list_fact,dim=0)
        label_value=[1 if i==3 else 0 for i in labels]
        score_list=self.sim_fct(query_list,candidate_fact)/self.temperature
        score_list1=self.sim_fct(query_list,candidate_struct)/self.temperature
        loss = self.one_hot_cross_entropy_loss(score_list, torch.tensor(label_value,dtype=torch.int32).to(score_list.device), reduction='mean')
        loss1 = self.one_hot_cross_entropy_loss(score_list1, torch.tensor(label_value,dtype=torch.int32).to(score_list.device), reduction='mean')
        loss=0.75*loss+0.25*loss1
        return loss+loss1
    @staticmethod
    def dot_product(emb1, emb2):
        return torch.matmul(emb1, emb2.permute(1, 0))
    @staticmethod
    def cosine_sim(emb1, emb2):
        return torch.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
    @staticmethod
    def one_hot_cross_entropy_loss(logits, target, weights=None, reduction='mean'):
        target=target.unsqueeze(0)
        if weights is None:
            weights = torch.ones_like(logits)
        logits_exp = torch.exp(logits)
        logits_exp_w = logits_exp * weights
        pos = torch.sum(logits_exp_w * target, dim=1)
        neg = torch.sum(logits_exp_w * (1 - target), dim=1)
        loss_list = - torch.log(pos / (pos + neg))
        if any(x == math.inf or x == -math.inf for x in loss_list):
            print(pos)
            print(neg)
            print(target)
        if reduction is None:
            loss = loss_list
        elif reduction == "mean":
            loss = torch.sum(loss_list) / len(loss_list)
        elif reduction == "sum":
            loss = torch.sum(loss_list)
        else:
            raise NotImplementedError

        return loss
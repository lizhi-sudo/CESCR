import torch
import torch.nn as nn
import math
from torch import Tensor as T
from typing import List
from .models import Encoder
from .loss import AttentionLoss, ContrastiveLoss, SimcseLoss,ContrastiveLossLecard
import time,copy
from collections import defaultdict

class CrossCaseCL(nn.Module):
    def __init__(self, config):
        super(CrossCaseCL, self).__init__()
        self.config = config
        if config.getboolean('train','struct_encoder'):
            self.encoder_query, self.encoder_fact,self.encoder_reasoning,self.encoder_judgment = self.get_encoder(config)
        else:
            self.encoder_query, self.encoder_candidate = self.get_encoder(config)
        self.simcse_loss = SimcseLoss(config)
        self.attn_loss = AttentionLoss(config)
        self.contra_loss = ContrastiveLoss(config)
        self.contra_lecard_loss=ContrastiveLossLecard(config)
        self.use_simcse = config.getboolean('simcse_loss', 'use')
        self.weights = nn.Parameter(torch.randn(768, 768)) 
        self.bias = nn.Parameter(torch.randn(768))
        self.weights1 = nn.Parameter(torch.randn(768, 768))  
        self.bias1 = nn.Parameter(torch.randn(768))
        self.weights = nn.Parameter(self.weights.to(torch.bfloat16))
        self.bias = nn.Parameter(self.bias.to(torch.bfloat16))
        self.weights1 = nn.Parameter(self.weights1.to(torch.bfloat16))
        self.bias1 = nn.Parameter(self.bias1.to(torch.bfloat16))
        if config.getboolean('train','struct_encoder'):
            self.linear_layer=nn.Linear(in_features=2304, out_features=768).to(torch.bfloat16)

    def forward(self, inputs, mode=None):
        if mode=='c':
            mode='candidate'
        else:
            mode='query'
        embeddings = self.get_embedding_list(inputs, sub_batch_size=1, mode=mode)
        if mode=='candidate' :
            try:
                sep_idx=inputs['sep_idx']
            except:
                sep_id_list=[]
                for j in range(inputs['input_ids'].size()[0]):   
                    sep_id=torch.where(inputs['input_ids'][j,:]==102)[0].unsqueeze(0)
                    sep_id_list.append(sep_id)
                sep_idx=torch.cat(sep_id_list,dim=0)
            fact_all=[]
            fact_raw_embedding=[]
            for idx in range(len(sep_idx)):
                fact_embed=embeddings[idx,0:sep_idx[idx][0],:]
                tokrn_num=fact_embed.size()[0]

                reason_embed=embeddings[idx,sep_idx[idx][0]+1:sep_idx[idx][1],:]#92*768
                judgment_embed=torch.mean(embeddings[idx,sep_idx[idx][1]+1:sep_idx[idx][2],:],dim=0).unsqueeze(1)#768,1
                reason_num=reason_embed.size()[0]
                r_j=torch.matmul(reason_embed,judgment_embed)*(1.0/math.sqrt(reason_num))
                t_softmax=torch.softmax(r_j.squeeze(1).unsqueeze(0),dim=1)
                w_r_j=torch.matmul(t_softmax,reason_embed).squeeze(0).unsqueeze(1)
                
                t_j=torch.matmul(fact_embed,w_r_j)*(1.0/math.sqrt(tokrn_num))
                t_softmax=torch.softmax(t_j.squeeze(1).unsqueeze(0),dim=1)
                fact_embed=torch.matmul(t_softmax,fact_embed)
                fact_all.append(fact_embed)
                fact_raw_embedding.append(embeddings[idx,0,:].unsqueeze(0))
            embeddings=torch.cat(fact_all,dim=0)
            embeddings_fact=torch.cat(fact_raw_embedding,dim=0)
            outputs=(embeddings,embeddings_fact)
        else:
            outputs=embeddings
        if mode =='candidate' and self.config.getboolean('train','struct_encoder'):
            self.linear_layer = self.linear_layer.to(torch.bfloat16) 
            embeddings=embeddings


        return outputs

    @staticmethod
    def get_encoder(config):
        encoder_query = Encoder(config)
        if config.getboolean('train','struct_encoder'):
            encode_fact=Encoder(config)
            encode_reasoning=Encoder(config)
            encode_judgment=Encoder(config)
            return encoder_query,encode_fact,encode_reasoning,encode_judgment
        else:
            encoder_candidate = Encoder(config)

        if config.getboolean('encoder', 'shared'):
            encoder_e = None

        return encoder_query, encoder_candidate

    def get_embedding_list(self, inputs, sub_batch_size=4, mode='fact'):
        """
        Calculate the embeddings of a large list of sentences.
        """
        output_list = list()

        if mode == 'query':
            model = self.encoder_query
            outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],mode=mode)
            return outputs
        if mode == 'candidate' and self.config.getboolean('train','struct_encoder')== False:
            model = self.encoder_candidate if self.encoder_candidate else self.encoder_query

            outputs = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],mode=mode)

        if self.config.getboolean('train','struct_encoder'):
            fact=defaultdict(list)
            reasoning=defaultdict(list)
            judgment=defaultdict(list)
            for key in ['input_ids','attention_mask']:
                fact[key]=inputs['input_ids'][:,0,:].squeeze(1)
                reasoning[key]=inputs['input_ids'][:,1,:].squeeze(1)
                judgment[key]=inputs['input_ids'][:,2,:].squeeze(1)
            fact_embedding=self.encoder_fact(input_ids=fact['input_ids'],
                        attention_mask=fact['attention_mask'])
            reasoning_embedding=self.encoder_reasoning(input_ids=reasoning['input_ids'],
                        attention_mask=reasoning['attention_mask'])
            judgment_embedding=self.encoder_judgment(input_ids=judgment['input_ids'],
                        attention_mask=judgment['attention_mask'])
            outputs=torch.cat([fact_embedding.unsqueeze(1),reasoning_embedding.unsqueeze(1),judgment_embedding.unsqueeze(1)],dim=1)
            

        return outputs

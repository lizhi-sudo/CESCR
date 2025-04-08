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
        # configurations
        self.simcse_loss = SimcseLoss(config)
        self.attn_loss = AttentionLoss(config)
        self.contra_loss = ContrastiveLoss(config)
        self.contra_lecard_loss=ContrastiveLossLecard(config)
        self.use_simcse = config.getboolean('simcse_loss', 'use')
        self.weights = nn.Parameter(torch.randn(768, 768))  # 初始化权重
        self.bias = nn.Parameter(torch.randn(768))
        self.weights1 = nn.Parameter(torch.randn(768, 768))  # 初始化权重
        self.bias1 = nn.Parameter(torch.randn(768))
        self.weights = nn.Parameter(self.weights.to(torch.bfloat16))
        self.bias = nn.Parameter(self.bias.to(torch.bfloat16))
        self.weights1 = nn.Parameter(self.weights1.to(torch.bfloat16))
        self.bias1 = nn.Parameter(self.bias1.to(torch.bfloat16))
        if config.getboolean('train','struct_encoder'):
            print(config.get('train','struct_encoder'))
            print(1)
            self.linear_layer=nn.Linear(in_features=2304, out_features=768).to(torch.bfloat16)
            # self.linear_layer_fact=nn.Linear(in_features=128, out_features=128).to(torch.bfloat16)
            # self.linear_layer_fact=nn.Linear(in_features=128, out_features=128).to(torch.bfloat16)
             

    def forward(self, inputs, mode=None):
        if mode=='c':
            mode='candidate'
        else:
            mode='query'
        embeddings = self.get_embedding_list(inputs, sub_batch_size=1, mode=mode)
        # if mode=='candidate' and inputs['input_ids'].size()[1]!=512:#消融实验没有内部结构
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
                # temp=torch.matmul(reason_embed,self.weights)+self.bias
                r_j=torch.matmul(reason_embed,judgment_embed)*(1.0/math.sqrt(reason_num))#92,1
                t_softmax=torch.softmax(r_j.squeeze(1).unsqueeze(0),dim=1)#1,92
                w_r_j=torch.matmul(t_softmax,reason_embed).squeeze(0).unsqueeze(1)#768,1
                
                # temp1=torch.matmul(self.weights1,w_r_j)+self.bias1.unsqueeze(1)
                t_j=torch.matmul(fact_embed,w_r_j)*(1.0/math.sqrt(tokrn_num))#129,1
                t_softmax=torch.softmax(t_j.squeeze(1).unsqueeze(0),dim=1)#1,129
                fact_embed=torch.matmul(t_softmax,fact_embed)#1,768
                fact_all.append(fact_embed)
                fact_raw_embedding.append(embeddings[idx,0,:].unsqueeze(0))
            embeddings=torch.cat(fact_all,dim=0)
            embeddings_fact=torch.cat(fact_raw_embedding,dim=0)
            outputs=(embeddings,embeddings_fact)
        else:
            outputs=embeddings
        if mode =='candidate' and self.config.getboolean('train','struct_encoder'):
            self.linear_layer = self.linear_layer.to(torch.bfloat16) 
            # embeddings=self.linear_layer(embeddings)s
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
            # encoder_candidate=(encode_fact,encode_reasoning,encode_judgment)
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
            # if inputs['input_ids'].size()[1]==512:
            #     model = self.encoder_query
            # else:
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
            # outputs=torch.cat([fact_embedding,reasoning_embedding,judgment_embedding],dim=1)
            # outputs=fact_embedding+reasoning_embedding+judgment_embedding
            outputs=torch.cat([fact_embedding.unsqueeze(1),reasoning_embedding.unsqueeze(1),judgment_embedding.unsqueeze(1)],dim=1)
            

        return outputs

    def get_document_level_features(self, features: T, offset: list,mode:str) -> List[dict]:
        """
        Get the features for each document.
        """
        document_features = []
        for oft in offset:
            tmp = dict()
            tmp['doc_all'] = features[oft[0]: oft[1]]
            tmp['doc_raw']=tmp['doc_all']
            tmp['doc_drop']=tmp['doc_all']
            document_features.append(tmp)
        
        return document_features

    @staticmethod
    def attention(queries: T, keys: T, values: T) -> dict:
        """ Attention (dot product or cosine) """
        # queries [14,768]values[15,768]
        dot_product = torch.matmul(queries, keys.permute(1, 0))#(14,768)*(768,15)=(14,15)
        attention_scores = torch.softmax(dot_product, dim=-1) #14*15,每一行代表一个fact对15个evidence的分数
        outputs = torch.matmul(attention_scores, values)#(14,15)*(15,768)=(14,768) 公式3 得到每一个fact的近似正值

        sim_scores = dict(dot=attention_scores)
        # sim_scores = dict(dot=dot_product)

        return {'attn_outputs': outputs, 'attn_scores': attention_scores, 'sim_scores': sim_scores}

    def get_attn_pos_outputs(self, doc_fact_list, doc_evidence_list, query_mode='evidence'):
        # q: query, v: value, a: attention output, s: attention scores
        outputs = dict(doc_list_q=dict(single=[], double=[]),
                       doc_list_v=dict(single=[], double=[]),
                       doc_list_a=list(),
                       doc_list_s=list())

        contra_query_list = doc_fact_list if query_mode == 'fact' else doc_evidence_list
        contra_value_list = doc_evidence_list if query_mode == 'fact' else doc_fact_list

        for doc_id in range(len(doc_fact_list)):
            single_key = 'doc_raw' if self.use_simcse else 'doc_all'
            queries = contra_query_list[doc_id][single_key]
            values = contra_value_list[doc_id][single_key]

            outputs['doc_list_q']['single'].append(queries)
            outputs['doc_list_v']['single'].append(values)
            if self.use_simcse:
                outputs['doc_list_q']['double'].append(contra_query_list[doc_id]['doc_all'])
                outputs['doc_list_v']['double'].append(contra_value_list[doc_id]['doc_all'])

            attn_output = self.attention(queries=queries, keys=values, values=values)
            # doc_list_s代表每一个fact和相关evidence的分数,doc_list_a得到近似正例
            outputs['doc_list_a'].append(attn_output['attn_outputs'])
            outputs['doc_list_s'].append(attn_output['sim_scores'])

        return outputs

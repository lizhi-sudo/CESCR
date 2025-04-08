import copy
import os.path
import random
from torch.utils.data import Dataset
import jsonlines
from tqdm import tqdm
import torch
from collections import defaultdict
from datasets import load_dataset
import pandas as pd
import numpy as np

class InputExample(object):
    """
    An example for a document, containing:
        a list of single facts:  [f1, f2, ..., fn]
        a list of single evidences doc: [[e1, e2], [e3, e4], [e5, e6]]
    """

    def __init__(self, candidates,query, candidate_raw=None,):
        self.query=query
        self.candidates_list=[]
        self.labels_list=[]
        self.fact_list=[]
        self.reasoning_list=[]
        self.judgment_list=[]
        self.candidates_raw_list=[]
        for c in candidates:
            self.candidates_list.append(c['content'])
            self.labels_list.append(c['label'])
            self.fact_list.append(c['fact'])
            self.reasoning_list.append(c['analysis'])
            self.judgment_list.append(c['result'])
        for c in candidate_raw:
            self.candidates_raw_list.append(c['content'])


class InputFeature(object):


    def __init__(self, inputs_query, inputs_candidates,inputs_labels,fact_r_j,sep_idx):
        self.inputs_query = inputs_query
        self.inputs_candidates = inputs_candidates
        self.inputs_labels=inputs_labels
        self.inputs_fact_r_j = fact_r_j
        self.sep_idx=sep_idx
  


class DataProcessor(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.input_examples = None
        self.input_features = None

    def read_example(self, input_file):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, index):
        features = self.input_features[index]

        return features


class CLProcessor(DataProcessor):
    def __init__(self, config, tokenizer, input_file):
        super(CLProcessor, self).__init__(config, tokenizer)
        self.config = config
        self.double_feature = config.getboolean('simcse_loss', 'use')
        self.query_key=config.get('data','query_key')
        self.double_feature = False
        self.input_examples = self.read_example(input_file,mode=self.query_key)
        self.input_features = self.convert_examples_to_features(mode=self.query_key)
        self.sub_batch_size_candidate=config.getint('train', 'sub_batch_size_gpu')

    def read_example(self, input_file,mode):
        # 读取文件
        examples = []
        raw_file=load_dataset('json',data_files=input_file)
        raw_file=raw_file['train'].select(range(1,107))
        data=raw_file.train_test_split(test_size=0.2,seed=10)
        data_train=data['train']
        data_valid_test=data['test'].train_test_split(test_size=0.5,seed=10)
        data_valid=data_valid_test['train']
        data_test=data_valid_test['test']
        examples_train=[]
    
        for train_file in data_train:
            example = InputExample(candidates=train_file['candidate'], query=train_file[mode],candidate_raw=train_file["candidate_raw"])
            examples_train.append(example)
        save_train_path=self.config.get('data', 'train_data_raw')
        save_valid_path=self.config.get('data', 'valid_data_raw')
        save_test_path=self.config.get('data', 'test_data_raw')
        if  not os.path.exists(save_train_path):
            # print(1)
            with jsonlines.open(save_train_path, 'w') as f:
                for train_file in data_train:
                    jsonlines.Writer.write(f, dict(query_non_crime=train_file['query_non_crime'],query_raw=train_file['query_raw'],query_crime=train_file['query_crime'], candidates=train_file['candidate'],candidate_raw=train_file["candidate_raw"])) 

        if  not os.path.exists(save_valid_path):
            # print(1)
            with jsonlines.open(save_valid_path, 'w') as f:
                for valid_file in data_valid:
                    jsonlines.Writer.write(f, dict(query_non_crime=valid_file['query_non_crime'],query_raw=valid_file['query_raw'],query_crime=valid_file['query_crime'],candidates=valid_file['candidate'],candidate_raw=valid_file["candidate_raw"])) 

        if not os.path.exists(save_test_path):
            with jsonlines.open(save_test_path, 'w') as f:
                for test_file in data_test:
                    # jsonlines.Writer.write(f, dict(query_non_crime=test_file['query_non_crime'],query_raw=test_file['query_raw'],query_crime=test_file['query_crime'],candidates=test_file['candidate']))
                    jsonlines.Writer.write(f, dict(query_non_crime=test_file['query_non_crime'],query_raw=test_file['query_raw'],query_crime=test_file['query_crime'],candidates=test_file['candidate'],candidate_raw=test_file["candidate_raw"]))
        
        return examples_train

    def convert_examples_to_features(self,mode):
        # feature_num，token_type构建文件名用到
        feature_num = 'double' if self.double_feature else 'single'
        token_type = self.config.get('encoder', 'backbone') + "_" + feature_num
        # data_path训练数据地址，cache_path保存文件地址
        # data_path = self.config.get('data', 'train_data')
        data_path = self.config.get('data', 'data_Lecard')
    
        # cache_path = data_path.replace('train_raw/', 'train_cache/').replace('.json', 'content-{}-{}-raw.jsonl'.format(token_type,mode))
        cache_path = data_path.replace('train_raw/', 'train_cache/').replace('.json', 'content-{}-{}-query_candidate_extract.jsonl'.format(token_type,mode))
        
        # 返回features
        features = []
        if os.path.exists(cache_path):
            inputs_list = jsonlines.open(cache_path)
            for inputs in tqdm(inputs_list, desc='Loading from {}'.format(cache_path)):
                # 构造InputFeature类型，两个属性inputs_facts，inputs_evidences
                #feature = InputFeature(inputs_facts=inputs['fact'], inputs_evidences=inputs['evidence'])
                feature = InputFeature(inputs_query=inputs['query'], inputs_candidates=inputs['candidates'],inputs_labels=inputs['labels'],fact_r_j=inputs['fact_r_j'],sep_idx=inputs['sep_idx'])
                features.append(feature)
        else:
            # 把文本进行tokenizer分词
            with jsonlines.open(cache_path, 'w') as f:
                for example in tqdm(self.input_examples, desc='Converting examples to features'):
                    #克隆
                    query_sent_list = copy.deepcopy(example.query)
                    candidates_list = copy.deepcopy(example.candidates_list)
                    candidates_raw_list = copy.deepcopy(example.candidates_raw_list)
                    labels_list=copy.deepcopy(example.labels_list)
                    fact_list=copy.deepcopy(example.fact_list)
                    reasoning_list=copy.deepcopy(example.reasoning_list)
                    judgment_list=copy.deepcopy(example.judgment_list)

                   
                    #分词
                    inputs_query = self.tokenizer(query_sent_list, padding='max_length', truncation=True, max_length=128)
                    inputs_candidates = defaultdict(list)
                    inputs_fact_reasoning_judgment = defaultdict(list)
                    sep_ids=[]
                    for candidate_sent in candidates_list:
                    # 初始数据进行消融实验
                    # for candidate_sent in candidates_raw_list:
                        candidate_inputs = self.tokenizer(candidate_sent, padding='max_length', truncation=True, max_length=512)
                        for key in candidate_inputs:
                            inputs_candidates[key].append(candidate_inputs[key])
                    # 内部结构数据分词
                    for f_sent,r_sent,j_sent in zip(fact_list,reasoning_list,judgment_list):
                        sentence_input=f_sent+'[SEP]'+r_sent+'[SEP]'+j_sent
                        inputs_fact_ = self.tokenizer(sentence_input, padding='max_length', \
                                                      truncation=True, max_length=512)
                        sep_id=torch.where(torch.tensor(inputs_fact_['input_ids'])==self.tokenizer.sep_token_id)[0].numpy().tolist()
                        sep_ids.append(sep_id)
                        for key in inputs_fact_:
                            inputs_fact_reasoning_judgment[key].append(inputs_fact_[key])

                    

                    if 'token_type_ids' in inputs_query and 'token_type_ids' in inputs_candidates:
                        del inputs_query['token_type_ids']
                        del inputs_candidates['token_type_ids']
                    inputs_labels=labels_list
                    #把分词之后的数据存储起来
                    jsonlines.Writer.write(f, dict(query=dict(inputs_query), candidates=dict(inputs_candidates),labels=list(inputs_labels),fact_r_j=dict(inputs_fact_reasoning_judgment),sep_idx=sep_ids))
                    features.append(InputFeature(inputs_query=inputs_query, inputs_candidates=inputs_candidates,inputs_labels=inputs_labels,fact_r_j=inputs_fact_reasoning_judgment,sep_idx=sep_ids))

            print('finished caching.')
        return features

    def collate_fn1(self,batch):
        batch_copy = copy.deepcopy(batch)
        output_batch = dict()

        inputs_query = {'input_ids': [], 'attention_mask': []}
        inputs_candidates_list=[]
        inputs_struct_list=[]
        sep_token_id_list=[]

        inputs_candidates = {'input_ids': [], 'attention_mask': []}

        # 分割candidate
        for _, b in enumerate(batch_copy):
            candidates = defaultdict(list)
            fact_r_j=defaultdict(list)
            for key in ['input_ids', 'attention_mask']:
                candidates[key].extend(b.inputs_candidates[key])

                fact_r_j[key].extend(b.inputs_fact_r_j[key])
 
            for idx in range(0,len(b.inputs_labels),self.sub_batch_size_candidate):

                inputs_candidates = defaultdict(list)
                inputs_fact_r_j = defaultdict(list)
                for key in ['input_ids', 'attention_mask']:
                    inputs_candidates[key].extend([candidates[key][idx:idx+self.sub_batch_size_candidate]])
                    inputs_candidates[key] = torch.tensor(inputs_candidates[key], dtype=torch.long)

                    inputs_fact_r_j[key].extend([fact_r_j[key][idx:idx+self.sub_batch_size_candidate]])
                    inputs_fact_r_j[key] = torch.tensor(inputs_fact_r_j[key], dtype=torch.long)
                inputs_fact_r_j['sep_idx']=[b.sep_idx[idx:idx+self.sub_batch_size_candidate]]
                inputs_fact_r_j['sep_idx'] = torch.tensor(inputs_fact_r_j['sep_idx'], dtype=torch.long)
                inputs_candidates_list.append(inputs_candidates)
                inputs_struct_list.append(inputs_fact_r_j)


            for key in ['input_ids', 'attention_mask']:
                inputs_query[key].extend(b.inputs_query[key])
                inputs_query[key] = torch.tensor(inputs_query[key], dtype=torch.long)
            labels=b.inputs_labels
        output_batch['inputs_query'] = inputs_query
        output_batch['inputs_candidates'] = inputs_candidates_list
        output_batch['struct_incase']= inputs_struct_list

        output_batch['labels']=labels
        output_batch['sep_idx']=sep_token_id_list



        return output_batch


    
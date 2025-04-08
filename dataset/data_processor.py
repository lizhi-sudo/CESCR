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

class InputExample(object):
    """
    An example for a document, containing:
        a list of single facts:  [f1, f2, ..., fn]
        a list of single evidences doc: [[e1, e2], [e3, e4], [e5, e6]]
    """

    def __init__(self, candidates, query):
        self.query=query
        self.candidates_list=[]
        self.labels_list=[]
        self.fact_list=[]
        self.reasoning_list=[]
        self.judgment_list=[]
        for c in candidates:
            self.candidates_list.append(c['result'])
            self.labels_list.append(c['label'])
            self.fact_list.append(c['fact'])
            self.reasoning_list.append(c['analysis'])
            self.judgment_list.append(c['result'])


class InputFeature(object):
    """
    构造数据格式
    A feature for a document, containing:
        for simcse:
            a list of doubled fact bert inputs: [f1, f1, f2, f2, ..., fn, fn]
            a list of doubled evidence bert inputs: [[e1, e1, e2, e2], [e3, e3, e4, e4], [e5, e5, e6, e6]]
        for non-simcse:
            a list of single fact bert inputs: [f1, f2, ..., fn]
            a list of single evidence bert inputs: [[e1, e2], [e3, e4], [e5, e6]]

    """
    def __init__(self, inputs_query, inputs_candidates,inputs_labels,inputs_fact,inputs_reasoning,inputs_judgment):
        self.inputs_query = inputs_query
        self.inputs_candidates = inputs_candidates
        self.inputs_labels=inputs_labels
        self.inputs_fact = inputs_fact
        self.inputs_reasoning = inputs_reasoning
        self.inputs_judgment = inputs_judgment


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
        data=raw_file['train'].train_test_split(test_size=0.2,seed=10)
        data_train=data['train']
        data_valid_test=data['test'].train_test_split(test_size=0.5,seed=10)
        data_valid=data_valid_test['train']
        data_test=data_valid_test['test']
        examples_train=[]
    
        for train_file in data_train:
            example = InputExample(candidates=train_file['candidate'], query=train_file[mode])
            examples_train.append(example)
        save_train_path=self.config.get('data', 'train_data')
        save_valid_path=self.config.get('data', 'valid_data')
        save_test_path=self.config.get('data', 'test_data')
        if  not os.path.exists(save_train_path):
            # print(1)
            with jsonlines.open(save_train_path, 'w') as f:
                for train_file in data_train:
                    jsonlines.Writer.write(f, dict(query_non_crime=train_file['query_non_crime'],query_raw=train_file['query_raw'],query_crime=train_file['query_crime'], candidates=train_file['candidate'])) 

        if  not os.path.exists(save_valid_path):
            # print(1)
            with jsonlines.open(save_valid_path, 'w') as f:
                for valid_file in data_valid:
                    jsonlines.Writer.write(f, dict(query_non_crime=valid_file['query_non_crime'],query_raw=valid_file['query_raw'],query_crime=valid_file['query_crime'],candidates=valid_file['candidate'])) 

        if not os.path.exists(save_test_path):
            with jsonlines.open(save_test_path, 'w') as f:
                for test_file in data_test:
                    jsonlines.Writer.write(f, dict(query_non_crime=test_file['query_non_crime'],query_raw=test_file['query_raw'],query_crime=test_file['query_crime'],candidates=test_file['candidate']))
        
        return examples_train

    def convert_examples_to_features(self,mode):
        # feature_num，token_type构建文件名用到
        feature_num = 'double' if self.double_feature else 'single'
        token_type = self.config.get('encoder', 'backbone') + "_" + feature_num
        # data_path训练数据地址，cache_path保存文件地址
        # data_path = self.config.get('data', 'train_data')
        data_path = self.config.get('data', 'data_Lecard')
    
        cache_path = data_path.replace('train_raw/', 'train_cache/').replace('.json', 'content-{}-{}.jsonl'.format(token_type,mode))
        
        
        # 返回features
        features = []
        if os.path.exists(cache_path):
            inputs_list = jsonlines.open(cache_path)
            for inputs in tqdm(inputs_list, desc='Loading from {}'.format(cache_path)):
                # 构造InputFeature类型，两个属性inputs_facts，inputs_evidences
                #feature = InputFeature(inputs_facts=inputs['fact'], inputs_evidences=inputs['evidence'])
                feature = InputFeature(inputs_query=inputs['query'], inputs_candidates=inputs['candidates'],inputs_labels=inputs['labels'],inputs_fact=inputs['fact'],inputs_reasoning=inputs['reasoning'],inputs_judgment=inputs['judgment'])
                features.append(feature)
        else:
            # 把文本进行tokenizer分词
            with jsonlines.open(cache_path, 'w') as f:
                for example in tqdm(self.input_examples, desc='Converting examples to features'):
                    #克隆
                    query_sent_list = copy.deepcopy(example.query)
                    candidates_list = copy.deepcopy(example.candidates_list)
                    labels_list=copy.deepcopy(example.labels_list)
                    fact_list=copy.deepcopy(example.fact_list)
                    reasoning_list=copy.deepcopy(example.reasoning_list)
                    judgment_list=copy.deepcopy(example.judgment_list)

                    if self.double_feature:
                        # double the inputs for simcse here事实和证据都复制两份
                        query_sent_list = sum([list(f) for f in zip(query_sent_list, query_sent_list)], [])
                        for c_id, candidate_doc in enumerate(candidates_list):
                            candidates_list[c_id] = sum([list(e) for e in zip(candidate_doc, candidate_doc)], [])
                        
                        for label_id, label in enumerate(labels_list):
                            labels_list[label_id] = [l for l in (label, label)]
                    else:
                        pass
                    #分词
                    inputs_query = self.tokenizer(query_sent_list, padding='max_length', truncation=True, max_length=128)
                    inputs_candidates = defaultdict(list)
                    inputs_fact = defaultdict(list)
                    inputs_reasoning = defaultdict(list)
                    inputs_judgment = defaultdict(list)
                    for candidate_sent in candidates_list:
                        candidate_inputs = self.tokenizer(candidate_sent, padding='max_length', truncation=True, max_length=128)
                        for key in candidate_inputs:
                            inputs_candidates[key].append(candidate_inputs[key])

                    for fact_sent in fact_list:
                        inputs_fact_ = self.tokenizer(fact_sent, padding='max_length', truncation=True, max_length=128)
                        for key in inputs_fact_:
                            inputs_fact[key].append(inputs_fact_[key])

                    for reasoning_sent in reasoning_list:
                        inputs_reasoning_ = self.tokenizer(reasoning_sent, padding='max_length', truncation=True, max_length=128)
                        for key in inputs_reasoning_:
                            inputs_reasoning[key].append(inputs_reasoning_[key])

                    for judgment_sent in judgment_list:
                        inputs_judgment_ = self.tokenizer(judgment_sent, padding='max_length', truncation=True, max_length=128)
                        for key in inputs_judgment_:
                            inputs_judgment[key].append(inputs_judgment_[key])

                    if 'token_type_ids' in inputs_query and 'token_type_ids' in inputs_candidates:
                        del inputs_query['token_type_ids']
                        del inputs_candidates['token_type_ids']
                    inputs_labels=labels_list
                    #把分词之后的数据存储起来
                    jsonlines.Writer.write(f, dict(query=dict(inputs_query), candidates=dict(inputs_candidates),labels=list(inputs_labels),fact=dict(inputs_fact),reasoning=dict(inputs_reasoning),judgment=dict(inputs_judgment)))
                    features.append(InputFeature(inputs_query=inputs_query, inputs_candidates=inputs_candidates,inputs_labels=inputs_labels,inputs_fact=inputs_fact,inputs_reasoning=inputs_reasoning,inputs_judgment=inputs_judgment))

            print('finished caching.')
        return features

    def collate_fn1(self,batch):
        batch_copy = copy.deepcopy(batch)
        output_batch = dict()

        inputs_query = {'input_ids': [], 'attention_mask': []}
        inputs_candidates_list=[]
        inputs_struct_list=[]

        inputs_candidates = {'input_ids': [], 'attention_mask': []}

        inputs_fact = {'input_ids': [], 'attention_mask': []}
        inputs_reasoning = {'input_ids': [], 'attention_mask': []}
        inputs_judgment = {'input_ids': [], 'attention_mask': []}

        offset = {'query': [], 'candidate': [], 'label': []}

        st1, st2 = 0, 0
        # 分割candidate
        for _, b in enumerate(batch_copy):
            candidates = defaultdict(list)
            fact=defaultdict(list)
            reasoning=defaultdict(list)
            judgment=defaultdict(list)
            for key in ['input_ids', 'attention_mask']:
                candidates[key].extend(b.inputs_candidates[key])

                fact[key].extend(b.inputs_fact[key])

                reasoning[key].extend(b.inputs_reasoning[key])

                judgment[key].extend(b.inputs_judgment[key])
            if self.double_feature:
                ed1 = st1 + len(b.inputs_query['input_ids'])
                ed2 = st2 + len(candidates['input_ids'])

                offset['query'].append([st1, ed1])
                offset['candidate'].append([st2, ed2])
                for j in b.inputs_labels:
                    offset['label'].extend(j)

                st1, st2 = ed1, ed2

            offset['label'].extend([b.inputs_labels])
            for idx in range(0,len(b.inputs_labels),self.sub_batch_size_candidate):
                inputs_candidates = defaultdict(list)
                inputs_fact = defaultdict(list)
                inputs_reasoning = defaultdict(list)
                inputs_judgment = defaultdict(list)
                inputs_struct_incase = defaultdict(list)
                for key in ['input_ids', 'attention_mask']:
                    inputs_candidates[key].extend([candidates[key][idx:idx+self.sub_batch_size_candidate]])
                    inputs_candidates[key] = torch.tensor(inputs_candidates[key], dtype=torch.long)

                    inputs_fact[key].extend([fact[key][idx:idx+self.sub_batch_size_candidate]])
                    inputs_fact[key] = torch.tensor(inputs_fact[key], dtype=torch.long)

                    inputs_reasoning[key].extend([reasoning[key][idx:idx+self.sub_batch_size_candidate]])
                    inputs_reasoning[key] = torch.tensor(inputs_reasoning[key], dtype=torch.long)

                    inputs_judgment[key].extend([judgment[key][idx:idx+self.sub_batch_size_candidate]])
                    inputs_judgment[key] = torch.tensor(inputs_judgment[key], dtype=torch.long)

                    inputs_struct_incase[key]=torch.cat([inputs_fact[key].squeeze(0).unsqueeze(1),inputs_reasoning[key].squeeze(0).unsqueeze(1),inputs_judgment[key].squeeze(0).unsqueeze(1)],dim=1)
                inputs_candidates_list.append(inputs_candidates)
                inputs_struct_list.append(inputs_struct_incase)
                # inputs_fact_list.append(inputs_fact)
                # inputs_reasoning_list.append(inputs_reasoning)
                # inputs_judgment_list.append(inputs_judgment)

            for key in ['input_ids', 'attention_mask']:
                inputs_query[key].extend(b.inputs_query[key])
                # inputs_candidates[key].extend([candidates[key]])
                

        for key in ['input_ids', 'attention_mask']:
            inputs_query[key] = torch.tensor(inputs_query[key], dtype=torch.long)
            # inputs_candidates[key] = torch.tensor(inputs_candidates[key], dtype=torch.long)

        output_batch['inputs_query'] = inputs_query
        output_batch['inputs_candidates'] = inputs_candidates_list
        output_batch['struct_incase']= inputs_struct_list

        if self.double_feature:
            output_batch['offset'] = offset
        else:
            output_batch['label']=offset['label']

        return output_batch


    def collate_fn(self, batch):
        # 把一个样本中的所有事实和证据分别放到一个列表里,offset表示相应事实和证据的索引位置
        batch_copy = copy.deepcopy(batch)
        output_batch = dict()

        inputs_facts = {'input_ids': [], 'attention_mask': []}
        inputs_evidences = {'input_ids': [], 'attention_mask': []}
        offset = {'fact': [], 'evidence': [], 'evidence_inner': []}

        st1, st2 = 0, 0
        for i, b in enumerate(batch_copy):
            if isinstance(b.inputs_evidences['input_ids'][0], list):
                # sample strategy: random sample two records and concatenate together as training evidence list
                # record_num多少个证据
                record_num = len(b.inputs_evidences['input_ids'])
                sample_num = self.config.getint('contra_loss', 'value_sample_num')
                indices = random.sample(range(record_num), min(sample_num, record_num))   # random sample 2 records
                tmp = defaultdict(list)
                for key in b.inputs_evidences:
                    for idx in indices:
                        tmp[key] += b.inputs_evidences[key][idx]
                evidence = tmp
            else:
                evidence = b.inputs_evidences

            ed1 = st1 + len(b.inputs_facts['input_ids'])
            ed2 = st2 + len(evidence['input_ids'])

            offset['fact'].append([st1, ed1])
            offset['evidence'].append([st2, ed2])

            st1, st2 = ed1, ed2

            for key in ['input_ids', 'attention_mask']:
                inputs_facts[key].extend(b.inputs_facts[key])
                inputs_evidences[key].extend(evidence[key])

        for key in ['input_ids', 'attention_mask']:
            inputs_facts[key] = torch.tensor(inputs_facts[key], dtype=torch.long)
            inputs_evidences[key] = torch.tensor(inputs_evidences[key], dtype=torch.long)

        output_batch['inputs_facts'] = inputs_facts
        output_batch['inputs_evidences'] = inputs_evidences
        output_batch['offset'] = offset

        return output_batch

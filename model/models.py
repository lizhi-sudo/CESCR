import torch
import torch.nn as nn
import json
import math
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel
from .utils import init_tokenizer, backbone_plm_dict


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], \
            "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state

        hidden_states = outputs.hidden_states

        if self.pooler_type == "cls":
            return outputs.pooler_output
        elif self.pooler_type == "cls_before_pooler":
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.pooler = Pooler(config.get('encoder', 'pooling'))
        self.model = AutoModel.from_pretrained(backbone_plm_dict[config.get('encoder', 'backbone')],trust_remote_code=True,torch_dtype=torch.bfloat16)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,mode=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        if mode=='query' :
            output = self.pooler(attention_mask, outputs)
        else:
            output=outputs['last_hidden_state']
        del outputs
        torch.cuda.empty_cache()
        return output


class AutoEncoder(nn.Module):
    def __init__(self, model_name, config):
        super(AutoEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(backbone_plm_dict[model_name], cache_dir='./data/ckpt/').cuda()
        self.pooler = Pooler(config.get('baseline', 'pooling'))
        self.tokenizer = init_tokenizer(model_name)

    def forward(self, input_text):
        inputs = self.tokenizer(input_text, padding="longest", truncation=True, max_length=128, return_tensors='pt')
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        outputs = self.bert(**inputs)
        output = self.pooler(inputs['attention_mask'], outputs)
        return output

    def search(self, query, corpus):
        query_embedding = self.forward(query)
        corpus_embeddings = self.forward(corpus)
        cos_scores = cosine_similarity(query_embedding.cpu(), corpus_embeddings.cpu())
        return cos_scores


class TFIDFSearcher(object):
    def __init__(self):
        super(TFIDFSearcher, self).__init__()
        self.stopwords = self.load_stopwords_list()
        self.token_dict = json.load(open('./data/token_dict.json', encoding='utf-8'))

        self.tf_dict = defaultdict(dict)
        self.inverted_index = defaultdict(list)
        self.idf_dict = dict()

    def search(self, queries, evidences):
        self.tf_dict, self.inverted_index = self.build_model(evidences)
        self.idf_dict = dict()

        sim_scores = np.zeros((len(queries), len(evidences)))
        for q_id, query in enumerate(queries):
            query_words = [w for w in self.token_dict['fact'][query].split("|") if w not in self.stopwords]

            for e_id, evidence in enumerate(evidences):
                tfidf_score = 0.0

                for word in query_words:
                    if word not in self.tf_dict[e_id]:
                        continue

                    tf = self.tf_dict[e_id][word]

                    if word not in self.idf_dict:
                        # idf(wi) = log[ #Docs. / (#Doc_contain_wi + 1)]
                        idf = math.log(len(evidences) / (len(set(self.inverted_index[word])) + 1))
                        self.idf_dict[word] = idf
                    else:
                        idf = self.idf_dict[word]

                    tfidf_score += tf * idf
                sim_scores[q_id][e_id] = tfidf_score

        return sim_scores

    def build_model(self, evidences):
        """
        Construct Inverted Index, TF lookup tables.
        """
        tf_dict = defaultdict(dict)
        inverted_index_dict = defaultdict(list)
        for e_id, e in enumerate(evidences):
            words = [w for w in self.token_dict['evidence'][e].split("|") if w not in self.stopwords]
            word_count_dic = Counter(words)

            for word in words:
                if not word.strip():
                    continue
                inverted_index_dict[word].append(e_id)
                tf_dict[e_id][word] = word_count_dic[word] / len(words)    # term frequency

        return tf_dict, inverted_index_dict

    @staticmethod
    def load_stopwords_list(stopwords_path='./data/stopwords.txt'):
        stopwords = [line.strip() for line in open(stopwords_path, "r", encoding="utf-8").readlines()]
        return stopwords



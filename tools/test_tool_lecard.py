import random
import numpy as np
import torch,math
import copy
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from model.utils import init_metric, init_tokenizer
from model.llm_sort_prompt import llm_score
import jsonlines,json
from modelscope import  AutoModelForCausalLM, AutoTokenizer

def read_example(input_file,mode):
    examples = []
    for case in jsonlines.open(input_file):
        example = dict(query=case[mode],candidates=case["candidates"],candidate_raw=case["candidate_raw"],query_raw=case["query_raw"])
        examples.append(example)
    return examples

def test(test_data_dir, model,struct_test=False, config=None, device=None, name=None, is_baseline=False, discrete=False,llm_second=False):
    model=model.to('cuda:1')
    model_path=r'/home/data/lz_data/Qwen2.5-7B-Instruct'
    mode=config.get('data','query_key')
    test_data=read_example(test_data_dir,mode=mode)
    k_list = [int(k) for k in config.get('test', 'k_list').split(',')]
    metric_list = [m.strip() for m in config.get('test', 'metric_list').split(',')]
    pos_score = config.getint('test', 'pos_score')
    tokenizer = init_tokenizer(config.get('encoder', 'backbone'))
    
    if llm_second:
        model_llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
        load_in_8bit=True,
        trust_remote_code=True
    )
        tokenizer_llm = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    eval_result_all = defaultdict(list)
    sentence_samples = []
    model.eval()
    test_data_ = copy.deepcopy(test_data)
    data_list=[]
    with torch.no_grad():
        for case_ in tqdm(test_data_, desc='Evaluating {}'.format(name)):
            case = copy.deepcopy(case_)
            query_sent = copy.deepcopy(case['query'])
            candidate_label_sent=copy.deepcopy(case['candidates'])
            query_sent_raw = copy.deepcopy(case['query_raw'])
            candidate_label_sent_raw=copy.deepcopy(case['candidate_raw'])
            candidate_content=[]
            candidate_list = []
            candidate_struct_list={'fact':[],'reasoning':[],'judgment':[]}
            # for c_l_content in candidate_label_sent:
            for c_l_content,c_l_content_raw in zip(candidate_label_sent,candidate_label_sent_raw):   
                if c_l_content['label']==-1:
                    c_l_content['label']=0
                if struct_test:
                    candidate_content.append((c_l_content['label'],c_l_content['fact'],\
                                               c_l_content['analysis'],c_l_content['result']))
                    candidate_struct_list['fact'].append(c_l_content['fact'][0])
                    candidate_struct_list['reasoning'].append(c_l_content['analysis'][0])
                    candidate_struct_list['judgment'].append(c_l_content['result'][0])

                else:

                    candidate_content.append((c_l_content['label'],c_l_content_raw['content']))
                    candidate_list.append(c_l_content['fact']+'[SEP]'+c_l_content['analysis']+'[SEP]'+c_l_content['result'])
            label_value=[0 if x['label'] == -1 else x['label'] for x in candidate_label_sent]
            if all(x == 0 for x in label_value):
                continue
            if not struct_test:
                inputs_query = tokenizer(query_sent, padding='longest', truncation=True, \
                                         max_length=128, return_tensors='pt')
                
                inputs_candidate = tokenizer(candidate_list, padding='longest', \
                                             truncation=True, max_length=512, return_tensors='pt')
                for ipt_key in inputs_query.keys():
                    inputs_query[ipt_key] = inputs_query[ipt_key].to('cuda:1')
                    inputs_candidate[ipt_key] = inputs_candidate[ipt_key].to('cuda:1')

                embedding_q = model(inputs=inputs_query,mode='q')

                embedding_c = model(inputs=inputs_candidate,mode='c')
                sim_scores = torch.cosine_similarity(embedding_q, 0.75*embedding_c[1]+0.25*embedding_c[0],dim=-1)
                candidate=dict()
                pred_ranking = copy.deepcopy(candidate_content)
                pred_indices = [(s, idx) for idx,s in enumerate(sim_scores)]
                pred_indices = sorted(pred_indices, key=lambda s: s[0], reverse=True)
                pred_ranking = [pred_ranking[idx] for _, idx in pred_indices]
                data_list.append({'candidate':pred_ranking,'query':query_sent_raw})

                if llm_second==True:
                    score_llm=llm_score(query_data=query_sent_raw,score_case=pred_ranking,model=model_llm,tokenizer=tokenizer_llm)
                    pred_ranking=score_llm
                eval_result = dict()
                for m_name in metric_list:
                    metric = init_metric(m_name)
                    if 'k' in m_name:
                        for k in k_list:
                            score = metric(pred_ranking, k=k, pos_score=pos_score)
                            eval_result[m_name.replace('k', str(k))] = score
                            eval_result_all[m_name.replace('k', str(k))].append(score)  # {"NDCG@5": [...],"P@5": [...]}
                    else:
                        score = metric(pred_ranking, pos_score=pos_score)
                        eval_result[m_name] = score
                        eval_result_all[m_name].append(score)     # {"NDCG@5": [...], "P@5": [...]}

                sentence_samples.append({
                                         'eval_result': eval_result,
                                         'pred_ranking': pred_ranking})
        with open('pred_ranking.json', 'w') as f:
            json.dump(data_list, f, indent=4)
        for m_name in eval_result_all:
            eval_result_all[m_name] = sum(eval_result_all[m_name]) / len(eval_result_all[m_name])

        eval_result_all['avg'] = sum(eval_result_all.values()) / len(eval_result_all.keys())

    return {'sent_avg': eval_result_all,
            'sentence_pred': sentence_samples}

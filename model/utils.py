from .metric import ndcg_at_k, precision_at_k, recall_at_k, mrr, mean_average_precision
from transformers import AutoTokenizer, BertTokenizerFast


backbone_plm_dict = {
    'bert': 'bert-base-chinese',
    'roberta': r'D:/doctor/baseline/roberta',
    'lawformer': r'/home/data/lz_data/Lawformer',
}



metric_dict = {
    "MRR": mrr,
    "NDCG@k": ndcg_at_k,
    "P@k": precision_at_k,
    "R@k": recall_at_k,
    "MAP": mean_average_precision,

}


def init_tokenizer(name, *args, **params):
    plm_path = backbone_plm_dict[name]
    return AutoTokenizer.from_pretrained(plm_path, cache_dir='./data/ckpt/')


def init_metric(name):
    if name in metric_dict:
        return metric_dict[name]
    else:
        raise NotImplementedError

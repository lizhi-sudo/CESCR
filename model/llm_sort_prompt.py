# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch,re
import numpy as np
from modelscope import  AutoModelForCausalLM, AutoTokenizer
def llm_score(query_data=None,score_case=None,model=None,tokenizer=None):
    score=[]
    for j in range(50):
        c=score_case[j][1]
        messages_score = [
            {"role": "user", 
             "content": f"请根据以下五个方面，评估两个法律案例的相似度并给出分析:\
                            1.案件事实：描述案件背景与关键事实的相似性。\
                            2.法律争议：两案的核心法律问题是否类似。\
                            3.法律依据：是否引用了相同或类似的法律条文、法规或判例。\
                            案例1:\n\
                            {query_data}\n\
                            案例2:\n\
                            {c}\n\
                            请按照以上五个方面分析，并给出综合相似度分数（0-100分），同时解释评分依据。在回答过程中，请先给出分数"}
        ]
        text = tokenizer.apply_chat_template(
            messages_score,
            tokenize=False,
            add_generation_prompt=True
        )
        sample_token=tokenizer([text],return_tensors="pt")
        gen_kwargs = {"max_new_tokens": 50}
        c=model.generate(**sample_token,**gen_kwargs)
        outputs = c[:, sample_token['input_ids'].shape[1]:]
        response=tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            s=float(re.findall(r'\d+', response)[0])
        except:
            s=0
        score.append(s)
    score_indices = [(s, idx) for idx,s in enumerate(score)]
    score_indices = sorted(score_indices, key=lambda s: s[0], reverse=True)
    
    pred_ranking_50 = [score_case[idx] for _, idx in score_indices]
    pred_ranking_50.extend(score_case[50:])
    return pred_ranking_50


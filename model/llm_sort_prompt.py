# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch,re
import numpy as np
from modelscope import  AutoModelForCausalLM, AutoTokenizer
def llm_score(query_data=None,score_case=None,model=None,tokenizer=None):
    # 排序后的数据score_case  [(c['label'],c['content']),(c['label'],c['content']),.....]
    model_llama=r'/home/data/lz_data/Baichuan2-7B-Base/'
    # model_llama=r'/home/data/lz_data/Qwen2.5-7B-base/'
    # model_path=r'/home/data/lz_data/glm-4-9b-chat'
    # model1 = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, 
    #                                          torch_dtype=torch.bfloat16, device_map="auto", 
    #                                          load_in_8bit=True,trust_remote_code=True)

    # tokenizer1 = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 计算历史案例嵌入
    c_embedding=[]
    score=[]
    for j in range(50):
        c=score_case[j][1][:8000]
        l=score_case[j][0]

        prompt = f"查询案例:{query_data}\n\n候选案例:{c}"
        # sample_token=tokenizer(f'<system> 你是一个法律案例分析助手,请根据输入的两个案例文本直接输出它们的相似度评分分数,分数为0-100之间。<user> 案例 1:{q}。案例 2:{c}。<assistant>',return_tensors="pt")
        # messages_score = [
        #     {"role": "system", "content": "你是一个法律类案判定助手,请从罪名、犯罪事实、涉及的法律条款、判决结果等方面，判断一下查询案例和候选案例的相似程度，并直接给出它们的相似度分数，评分大于0，并且小于100分，在回答过程中，请先给出分数"},
        #     {"role": "user", "content": prompt}
        # ]
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
        # messages_score = [
        #     {"role": "user", "content": "请判断给出的案例的相似程度，并直接给出它们的相似度分数，评分大于0，并且小于100分，在回答过程中，请先给出分数"+prompt}
        # ]
        text = tokenizer.apply_chat_template(
            messages_score,
            tokenize=False,
            add_generation_prompt=True
        )
        sample_token=tokenizer([text],return_tensors="pt")
        gen_kwargs = {"max_new_tokens": 50}
        c=model.generate(**sample_token,**gen_kwargs)
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(sample_token.input_ids, c)
        # ]

        # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        outputs = c[:, sample_token['input_ids'].shape[1]:]
        response=tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(response)
        try:
            s=float(re.findall(r'\d+', response)[0])
        except:
            # print([i,j])
            s=0
        # print(s)
        score.append(s)
    print(score)
    score_indices = [(s, idx) for idx,s in enumerate(score)]
    score_indices = sorted(score_indices, key=lambda s: s[0], reverse=True)
    
    pred_ranking_50 = [score_case[idx] for _, idx in score_indices]
    pred_ranking_50.extend(score_case[50:])
    return pred_ranking_50


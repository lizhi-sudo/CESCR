import logging
import torch
import json
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler
import os
import torch.distributed as dist
from tqdm import tqdm
from .test_tool_lecard import test
from torch.utils.data import DataLoader
from model.loss import ContrastiveLossLecard
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def checkpoint(filename, model, optimizer, trained_epoch, config, global_step,gpu_batch):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step,
        "batch_size_gpu":gpu_batch
    }
    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


# def get_output_folder(config):
#     # simcse configuration
#     simcse = "simcse-{}".format('parallel' if config.getboolean('simcse_loss', 'negatives_parallel') else 'cross')
#     # if not config.getboolean('contra_loss', 'positive_query'):
#     #     simcse = ''
#     if not config.getboolean('simcse_loss', 'use'):
#         simcse = ''

#     # contra configuration
#     rm = '_rm-hard{}{}'.format('-attn' if config.getboolean('contra_loss', 'remove_hard_attention') else "",
#                                '-qry' if config.getboolean('contra_loss', 'remove_hard_query') else "")
#     contra = "contra-{}-{}{}{}{}".format(config.get('contra_loss', 'query'),
#                                          int(config.getboolean('contra_loss', 'negatives_attention')),
#                                          int(config.getboolean('contra_loss', 'negatives_value')),
#                                          int(config.getboolean('contra_loss', 'negatives_query')),
#                                          rm if rm != '_rm-hard' else "")
#     if not config.getboolean('contra_loss', 'use'):
#         contra = ''

#     model_name = "{}{}".format(simcse, '-'+contra if simcse else contra)

#     # weighted positive
#     use_pos_weight = config.getboolean('positive_weight', 'use')

#     weight_source = config.get('positive_weight', 'source')  # dot or cos
#     weight_type = config.get('positive_weight', 'type')  # sum or norm
#     pos_range = config.get('positive_weight', 'range')  # in-case or in-batch
#     normalize = config.get('positive_weight', 'normalize')  # soft or hard
#     log_sum = "log" if config.getboolean('positive_weight', 'log_sum') else ""

#     # './data/train/train_harm_record-wo-test.jsonl'
#     train_data_path = config.get('data', 'train_data').replace("_record", "")
#     train_type = train_data_path[train_data_path.index('_')+1: train_data_path.index('.jsonl')]

#     output_path = os.path.join(config.get("output", "model_path"),
#                                train_type,
#                                'shared' if config.getboolean("encoder", "shared") else "non_shared",
#                                config.get('encoder', 'pooling'),
#                                config.get('encoder', 'backbone'),
#                                "{}{}".format(model_name, '-weighted' if use_pos_weight else ""),
#                                "{}-weight".format(weight_source) if use_pos_weight else "",
#                                weight_type if use_pos_weight else "",
#                                pos_range if use_pos_weight else "",
#                                "{}{}".format(normalize, "-log" if log_sum else "") if use_pos_weight else "")
#     os.makedirs(output_path, exist_ok=True)
#     return output_path


def train(parameters, config, gpu_list, local_rank=-1):
    # writer=SummaryWriter(log_dir='./12-4-logdir-step-2/')
    # example_input=torch.rand(4,3,10)
    # writer.add_graph(parameters['model'], (example_input,))
    loss_func=ContrastiveLossLecard(config)
    sub_batch_gpu=config.getint('train', 'sub_batch_size_gpu')
    struct_key=config.get('train', 'struct_key')
    candidate_key=config.get('train', 'candidate_key')
    epoch = config.getint('train', 'epoch')

    # output_path = get_output_folder(config)

    trained_epoch = parameters['trained_epoch'] + 1
    model = parameters['model']

    optimizer = parameters['optimizer']
    dataset = parameters['train_dataset']
    dataset_raw = parameters['Non_dataloader_train_dataset']
    global_step = parameters['global_step']

    step_size = config.getint('train', 'step_size')
    save_step = config.getint('train', 'save_step')
    gamma = config.getfloat('train', 'lr_multiplier')
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    grad_accumulate = config.getint('train', 'grad_accumulate')

    logger.info('Training start ... ')
    output_path='./parameters_fact_075_struct_025'
    os.makedirs(output_path, exist_ok=True)
    
    with open(output_path + '/logging.txt', 'w') as f:
        info = dict()
        info['training_data'] = config.get('data', 'train_data')

        info['encoder'] = dict(backbone=config.get('encoder', 'backbone'),
                               shared=config.get('encoder', 'shared'),
                               pooling=config.get('encoder', 'pooling'))
        info['attention'] = dict(sim_fct=config.get('attention', 'type'),
                                 scale=config.get('attention', 'scale'),
                                 temperature=config.get('attention', 'temperature'))
        info['simcse'] = dict(use=config.get('simcse_loss', 'use'),
                              parallel=config.get('simcse_loss', 'negatives_parallel'),
                              cross=config.get('simcse_loss', 'negatives_cross'),
                              single=config.get('simcse_loss', 'negatives_parallel_single'),
                              temperature=config.get('simcse_loss', 'temperature'))
        info['contra'] = dict(use=config.get('contra_loss', 'use'),
                              query=config.get('contra_loss', 'query'),
                              neg_attention=config.get('contra_loss', 'negatives_attention'),
                              neg_value=config.get('contra_loss', 'negatives_value'),
                              neg_query=config.get('contra_loss', 'negatives_query'))

        info['weighted'] = dict(use=config.get('positive_weight', 'use'),
                                source=config.get('positive_weight', 'source'),
                                type=config.get('positive_weight', 'type'),
                                range=config.get('positive_weight', 'range'),
                                hard_weighted=config.get('positive_weight', 'normalize'),
                                log=config.get('positive_weight', 'log_sum'))

        info['batch_size'] = config.get('train', 'batch_size')
        info['learning_rate'] = config.get('train', 'learning_rate')
        info['epoch'] = config.get('train', 'epoch')
        info['grad_accumulate'] = config.get('train', 'grad_accumulate')

        f.write(json.dumps(info, indent=4, ensure_ascii=False) + '\n')
    best_metric = 0
    
    result_epoch=defaultdict(list)
    for current_epoch in tqdm(range(trained_epoch, epoch)):
        model.train()
        model=model.to('cuda')
        scaler = torch.amp.GradScaler('cuda')
        # key=[inputs_query,inputs_candidate,label]
        loss_epoch=[]
        for step, data in enumerate(dataset):
            candidate_ouput_list=[]
            # if 3 not in data['label'][0]:
            #     continue
            with torch.amp.autocast('cuda'):
                for key in data:
                    if key == 'inputs_query':
                        query_data=data[key]
                        query_data={key_query:query_data[key_query].unsqueeze(0).to('cuda') for key_query in query_data}
                        model=model.to('cuda:0')
                        output_query=model(inputs=query_data,mode='q')
                    elif key == struct_key:
                        candidate_list_length=len(data[key])
                        # 每次计算24个样本的损失
                        for step_candidate in range(candidate_list_length):
                            # torch.cuda.empty_cache()
                            candidate_data=data[key][step_candidate]
                            candidate_data={key_candidate:candidate_data[key_candidate].to('cuda') for key_candidate in candidate_data}
                            candidate_data={key_candidate:candidate_data[key_candidate].squeeze(0).to('cuda') for key_candidate in candidate_data }
                            candidate_output=model(inputs=candidate_data,mode='c')
                            candidate_ouput_list.append(candidate_output)
                        #   loss=loss_func(output_query,candidate_output,data['labels'])
                        loss=loss_func(output_query,candidate_ouput_list,data['labels'])
                if config.getboolean('train', 'fp16'):
                    # scaler.scale(loss['overall']).backward()
                    scaler.scale(loss).backward()
                else:
                    # loss['overall'].backward()
                    loss.backward()
                if (step + 1) % grad_accumulate == 0:
                    if config.getboolean('train', 'fp16'):
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                    exp_lr_scheduler.step()
    #       # 打印以及测试、保存模型等
            logging_step = config.getint('train', 'logging_step')
            # writer.add_scalar('loss',loss,step)
            loss_epoch.append(loss)
            if step % logging_step == 0:
                info = 'epoch: {}, step:{}, loss: {}'.format(current_epoch, step + 1, loss)
                with open(output_path + '/logging.txt', 'a') as f:
                    f.write(info + '\n')
                print(info)
            if step%save_step==0:
                test_data=config.get('data','valid_data_raw')
                # result=test(test_data_dir=test_data,encoder_query=model.module.encoder_query,encoder_candidate=model.module.encoder_candidate,config=config)
                result=test(test_data_dir=test_data,model=model.module,struct_test=False,config=config)
                print(result['sent_avg'])
                print('test------------------------------------------------------------------------')
                # result1=test(test_data_dir=config.get('data','test_data'),model=model.module,struct_test=False,config=config)
                # print(result1['sent_avg'])
                # writer.add_scalars('metric',result['sent_avg'],step)
                print('\n')
                print(result['sent_avg']['avg'])
                running_loss = defaultdict(int)
                if float(result['sent_avg']['R@50']) > float(best_metric):
                    best_model=model
                    print(f"Saving best model (Loss: {loss:.4f})...")
                    best_metric = float(result['sent_avg']['R@50'])
                    checkpoint(
                    os.path.join(output_path, "{}-{}k.pkl".format(current_epoch, step )),
                    model,
                    optimizer,
                    current_epoch,
                    config,
                    global_step,
                    config.get('train', 'sub_batch_size_gpu'))
            
            # if step %1==0:
        # writer.add_scalar('epoch_loss',sum(loss_epoch)/len(loss_epoch),current_epoch)
        test_data=config.get('data','test_data_raw')
        result=test(test_data_dir=test_data,model=best_model.module,struct_test=False,config=config)
        print('test------test')
        print(result['sent_avg'])
        for key,value in result['sent_avg'].items():
            result_epoch[key].append(value)
        # writer.add_scalars('epoch_metric',result['sent_avg'],current_epoch)
        # print(result['sent_avg'])
    # length=[]
    # for _,value in result_epoch.items():
    #     length.append(len(value))
    # if len(set(length))==1:
    #     df=pd.DataFrame(result_epoch)
    #     df.to_json("result_lawformers_Lecard.json", orient='records', force_ascii=False,lines=True)
    # writer.close()      

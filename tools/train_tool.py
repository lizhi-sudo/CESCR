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


def train(parameters, config, gpu_list, local_rank=-1):
    loss_func=ContrastiveLossLecard(config)
    struct_key=config.get('train', 'struct_key')
    epoch = config.getint('train', 'epoch')
    trained_epoch = parameters['trained_epoch'] + 1
    model = parameters['model']
    optimizer = parameters['optimizer']
    dataset = parameters['train_dataset']
    global_step = parameters['global_step']
    step_size = config.getint('train', 'step_size')
    save_step = config.getint('train', 'save_step')
    gamma = config.getfloat('train', 'lr_multiplier')
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    grad_accumulate = config.getint('train', 'grad_accumulate')
    logger.info('Training start ... ')
    output_path='./parameters_fact_075_struct_025'
    os.makedirs(output_path, exist_ok=True)
    best_metric = 0
    result_epoch=defaultdict(list)
    for current_epoch in tqdm(range(trained_epoch, epoch)):
        model.train()
        model=model.to('cuda')
        scaler = torch.amp.GradScaler('cuda')
        loss_epoch=[]
        for step, data in enumerate(dataset):
            candidate_ouput_list=[]
            with torch.amp.autocast('cuda'):
                for key in data:
                    if key == 'inputs_query':
                        query_data=data[key]
                        query_data={key_query:query_data[key_query].unsqueeze(0).to('cuda') for key_query in query_data}
                        model=model.to('cuda:0')
                        output_query=model(inputs=query_data,mode='q')
                    elif key == struct_key:
                        candidate_list_length=len(data[key])
                        for step_candidate in range(candidate_list_length):
                            candidate_data=data[key][step_candidate]
                            candidate_data={key_candidate:candidate_data[key_candidate].to('cuda') for key_candidate in candidate_data}
                            candidate_data={key_candidate:candidate_data[key_candidate].squeeze(0).to('cuda') for key_candidate in candidate_data }
                            candidate_output=model(inputs=candidate_data,mode='c')
                            candidate_ouput_list.append(candidate_output)
                        loss=loss_func(output_query,candidate_ouput_list,data['labels'])
                if config.getboolean('train', 'fp16'):
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % grad_accumulate == 0:
                    if config.getboolean('train', 'fp16'):
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                    exp_lr_scheduler.step()
            logging_step = config.getint('train', 'logging_step')
            loss_epoch.append(loss)
            if step % logging_step == 0:
                info = 'epoch: {}, step:{}, loss: {}'.format(current_epoch, step + 1, loss)
                with open(output_path + '/logging.txt', 'a') as f:
                    f.write(info + '\n')
                print(info)
            if step%save_step==0:
                test_data=config.get('data','valid_data_raw')
                result=test(test_data_dir=test_data,model=model.module,struct_test=False,config=config)
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
        test_data=config.get('data','test_data_raw')
        result=test(test_data_dir=test_data,model=best_model.module,struct_test=False,config=config)
        print('test')
        print(result['sent_avg'])
        for key,value in result['sent_avg'].items():
            result_epoch[key].append(value)  

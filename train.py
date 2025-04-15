import argparse
from config_parser import create_config
import os,torch
from tools.init_tool import init_all
from tools.train_tool import train
from model.CrossCaseCL import CrossCaseCL
from tools.test_tool_lecard import test
import pandas as pd
def test_main(config,checkpoint,test_data):
    model=CrossCaseCL(config)
    existing = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(existing['model'])
    model=model.cuda()
    result=test(test_data_dir=test_data,model=model,struct_test=False,llm_second=False,config=config)
    print(result['sent_avg'])
    return result['sent_avg']
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", default='./config/template.config',required=False)
    parser.add_argument('--gpu_id',  default='0,1,2,3' ,help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--local_rank')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_list = []
    if args.gpu_id is not None: 
        gpu_list = list(range(0, len(args.gpu_id.split(','))))
    config = create_config(args.config)
    parameters = init_all(config, args.local_rank, gpu_list, args.checkpoint, args.seed, mode='train')
    train(parameters, config, gpu_list)


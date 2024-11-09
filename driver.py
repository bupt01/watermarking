import itertools
import math
import subprocess
from datasets import load_dataset, Dataset
import watermarkImpl
from argparse import Namespace

# 超参数的设置
gamma = [0.5]
delta = [0.5, 1, 2, 5, 10]
max_new_tokens = [200]
sample = [8]


# 将所有超参数的列表组合在一起
all_combinations = list(itertools.product(gamma, delta, max_new_tokens, sample))
# index=0
# 依次执行每一种超参数组合

# dataset_name, dataset_config_name = "c4", "realnewslike"
    

dataset = load_dataset("allenai/c4", "realnewslike",split='train',trust_remote_code=True, cache_dir="/project/lm-watermarking-main/c4/cache/")
def add_idx(example, idx):
    example.update({"idx": idx})
    return example
indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)
arg_dict = {
    'run_gradio': False,
    'demo_public': False,
    # 'model_name_or_path': 'facebook/opt-125m',
    # 'model_name_or_path': '/project/lm-watermarking-main/',
    # 'model_name_or_path': 'facebook/opt-2.7b',
    'model_name_or_path': 'facebook/opt-1.3b',
    # 'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16': False,
    'prompt_max_length': None,
    'max_new_tokens': 200, #params[2],
    'generation_seed': 123,
    'use_sampling': True,#params[3]==1,
    'n_beams': 1,#params[3],
    'sampling_temp': 0.7,
    'use_gpu': True,
    'seeding_scheme': 'simple_1',
    'gamma': 0.25,#params[0],
    'delta': 1,#params[1],
    'normalizers': '',
    'ignore_repeated_bigrams': False,
    'detection_z_threshold': 4.0,
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'min_sample_tokens':200,#params[2],#200,
    'original_watermark':True,
    'shuffle_dataset':False,
    'output_dir':'./output',
    'shuffle_buffer_size':10_000,
    'shuffle_seed':1234,
    'min_prompt_tokens':50,
    'oracle_model_name':None,#'facebook/opt-2.7b',
    'file_name':'gamma',#str(params[0])+'delta'+str(params[1])+'tokens'+str(params[2])+'search'+str(params[3])
}
   
args = Namespace()

   
args.__dict__.update(arg_dict)


model, tokenizer, device ,oracle_model,oracle_tokenizer,oracle_device= watermarkImpl.load_model(args)

shuffled_dataset = indexed_dataset       

for params in all_combinations[1:]:
    # 生成命令参数列表
    # command = ["python", script_name] + [str(param) for param in params]
    # print("Executing command:", " ".join(command))

    # from argparse import Namespace

    # args = Namespace()
    print(params)
    arg_dict['max_new_tokens']=params[2]
    arg_dict['use_sampling']=(params[3]==1)
    arg_dict['n_beams']=params[3]
    arg_dict['gamma']=params[0]
    arg_dict['delta']=params[1]
    arg_dict['min_sample_tokens']=params[2]
    arg_dict['file_name']='gamma'+str(params[0])+'delta'+str(params[1])+'tokens'+str(params[2])+'search'+str(params[3])
    
   
    args.__dict__.update(arg_dict)
    print(args)

    max_retries = 10000  # 设置最大重试次数
    retry_count = 0

    while retry_count < max_retries:
        try:
            result = watermarkImpl.main(args, shuffled_dataset, model, tokenizer, device, oracle_model, oracle_tokenizer, oracle_device)
            break  # 如果成功执行，跳出循环
        except Exception as e:
            print(f"出现异常: {e}，正在重试... (第 {retry_count + 1} 次)")
            retry_count += 1
            if retry_count >= max_retries:
                print("达到最大重试次数，操作失败")
                result = None  # 或者返回默认值、或其他处理方式

    # result=watermarkImpl.main(args,shuffled_dataset,model,tokenizer,device,oracle_model,oracle_tokenizer,oracle_device)



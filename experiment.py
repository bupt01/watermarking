import os
from functools import partial

from datasets import load_dataset, Dataset
# better bool flag type for argparse
# from submitit_utils import str2bool

# some file i/o helpers
# from io_utils import write_jsonlines, write_json, read_jsonlines, read_json
import watermarkImpl
import argparse






def main(args,delta):
    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)
        
    dataset_name, dataset_config_name = "c4", "realnewslike"
    

    dataset = load_dataset("allenai/c4", "realnewslike",split='train',trust_remote_code=True, cache_dir="/project/lm-watermarking-main/c4/cache/")
    
    def add_idx(example, idx):
        example.update({"idx": idx})
        return example
    
    arg_dict = {
        'run_gradio': False,
        'demo_public': False,
        # 'model_name_or_path': 'facebook/opt-125m',
         'model_name_or_path': 'facebook/opt-1.3b',
        # 'model_name_or_path': 'facebook/opt-2.7b',
        #'model_name_or_path': 'facebook/opt-6.7b',
        # 'model_name_or_path': 'facebook/opt-13b',
        # 'load_fp16' : True,
        'load_fp16': False,
        'prompt_max_length': None,
        'max_new_tokens': 200,
        'generation_seed': 123,
        'use_sampling': True,
        'n_beams': 1,
        'sampling_temp': 0.7,
        'use_gpu': True,
        'seeding_scheme': 'simple_1',
        'gamma': 0.5,
        'delta': delta, #2,
        'normalizers': '',
        'ignore_repeated_bigrams': False,
        'detection_z_threshold': 4.0,
        'select_green_tokens': True,
        'skip_model_load': False,
        'seed_separately': True,
        'oracle_model_name':None,#'facebook/opt-2.7b',
    }

        #  if "c4" in dataset_name:
        #     columns_to_remove = ["text","timestamp","url"]
        #  else:
        #     columns_to_remove = []
            
    indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)
    if args.shuffle_dataset:
        shuffled_dataset = indexed_dataset.shuffle(seed=args.shuffle_seed,
                                                   buffer_size=args.shuffle_buffer_size)
    else:
        shuffled_dataset = indexed_dataset
            
        
        #以下求证下
        #shuffled the first shuffle_buffer_size rows of the (streaming) dataset
   
  
    args.__dict__.update(arg_dict)
  
    result=watermarkImpl.main(args,shuffled_dataset)
    return

    # # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
    # tokenized_and_truncated_dataset = shuffled_dataset.map(tokenize_prompts,
    #                                                        batched=False,
    #                                                        with_indices=True)
    #
    # # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
    # input_length_filtered_dataset = tokenized_and_truncated_dataset.filter(input_check,
    #                                                                        batched=False,
    #                                                                        with_indices=True)
    #
    # # perform generation by calling the models
    # columns_to_remove += ["inputs", "untruncated_inputs"]  # these are now materialized and must be dropped externally
    # generations_dataset = input_length_filtered_dataset.map(gen_completions,
    #                                                         batched=False,
    #                                                         with_indices=True,
    #                                                         remove_columns=columns_to_remove)



if __name__ == "__main__":
      #开始做实验啦
    parser = argparse.ArgumentParser(description="Run watermarked huggingface LM generation pipeline")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=50, # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="The seed to use for dataset shuffle op.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="The buffer size to use for dataset shuffle op - takes n rows first, then shuffles those indices",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        type=bool,
        default=False,
        help="Shuffle or not",
    )
    parser.add_argument(
        "--original_watermark",
        type=bool,
        default=True,
        help="Original watermark or not",
    )
    parser.add_argument(
        "--signal_period",
        type=int,
        default=200,
        help="The number of tokens in a period",
    )
    
    parser.add_argument(
        "--min_sample_tokens",
        type=int,
        default=200, 
        help="The the minimum length of raw prompt samples to consider.",
    )
    
    

   
    args = parser.parse_args()
    #deltas=[1.5,2.5,3,4]
    # deltas=[3,4]
    # deltas=[3.5,4.5]
    deltas=[1,2,3,4,5,10]
    for delta in deltas:
        main(args,delta)

    # print(args)

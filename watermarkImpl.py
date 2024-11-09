# added by toby
# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial

import numpy  # for gradio hot reload
import gradio as gr
import json
import math
import pandas as pd
import csv

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector,MultiDimensionWatermarkLogitsProcessor,MultiWatermarkDetector,WatermarkDetectorMD

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)
def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(
        description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=False,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    args = parser.parse_args()
    return args


def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]])
    args.is_decoder_only_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16,
                                                         device_map='auto')
            
            
            oracle_model = None if(args.oracle_model_name==None) else AutoModelForCausalLM.from_pretrained(args.oracle_model_name, torch_dtype=torch.float16,
                                                         device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            oracle_model = None if(args.oracle_model_name==None) else AutoModelForCausalLM.from_pretrained(args.oracle_model_name, torch_dtype=torch.float16,
                                                         device_map='auto')
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")
    gpu_index = 0 # 例如，使用第二个GPU
    oracle_index=1
    oracle_device=None
    if args.use_gpu:
        device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        oracle_device=f"cuda:{oracle_index}" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
            if oracle_model!=None:
                oracle_model=oracle_model.to(oracle_device)
    else:
        device = "cpu"
    model.eval()
    if oracle_model!=None:
        oracle_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    oracle_tokenizer = None if(args.oracle_model_name==None) else AutoTokenizer.from_pretrained(args.oracle_model_name)

    return model, tokenizer, device,oracle_model,oracle_tokenizer, oracle_device



def generate(example, args, model=None, device=None, tokenizer=None,original_watermark=True,oracle_model=None,oracle_tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """

    # print(f"Generating with {args}")
    inputs = example["text"]
    
    original_watermark=args.original_watermark;
    
    
    if args.prompt_max_length:
        pass
    elif hasattr(model.config, "max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        args.prompt_max_length = 2048 - args.max_new_tokens

    # token_kwargs = dict(
    #     hf_model_name=args.model_name_or_path,
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    max_new_tokens = args.max_new_tokens #该参数用于将数据集的切断
   
    slice_length = min(len(inputs)-1, max_new_tokens)

    # tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
    #                        max_length=args.prompt_max_length).to(device)
    
    tokd_input = tokenizer(inputs, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=min(args.prompt_max_length, len(inputs)-slice_length)).to(device)
    
    # print(tokd_input)
    # print(len(tokd_input['input_ids'][0]))

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                   gamma=args.gamma,
                                                   delta=args.delta,
                                                   seeding_scheme=args.seeding_scheme,
                                                   select_green_tokens=args.select_green_tokens)
    
    
    water_multi_processor=MultiDimensionWatermarkLogitsProcessor(t=args.signal_period,prompt_len=len(tokd_input['input_ids'][0]),
                                                                 vocab=list(tokenizer.get_vocab().values()),
                                                   gamma=args.gamma,
                                                   delta=args.delta,
                                                   seeding_scheme=args.seeding_scheme,
                                                   select_green_tokens=args.select_green_tokens)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True,
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        **gen_kwargs
    )
    generate_with_multi_watermark= partial(
        model.generate,
        logits_processor=LogitsProcessorList([water_multi_processor]),
        **gen_kwargs
    )
   
    

        
    # example=truncate(example,tokd_input["input_ids"],completion_length=token_kwargs["max_new_tokens"],
    #                  prompt_length=token_kwargs["min_prompt_tokens"],
    #                  hf_model_name=token_kwargs["hf_model_name"],
    #                  model_max_seq_len=None)
    
    
    # truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input) 

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)
    output_with_multi_watermark = generate_with_multi_watermark(**tokd_input)
    output_with_watermark =generate_with_watermark(**tokd_input) if (original_watermark is True) else None
    # print(redecoded_input)
    # print('\n')
    # print(tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0])
    # print('\n') 
    # print(tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0])
    # print('\n')  
    # print(tokenizer.batch_decode(output_with_multi_watermark, skip_special_tokens=True)[0])
    # exit()

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:, tokd_input["input_ids"].shape[-1]:] if original_watermark else None 
        output_with_multi_watermark=output_with_multi_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0] if original_watermark else None
    output_with_multi_watermark = tokenizer.batch_decode(output_with_multi_watermark, skip_special_tokens=True)[0]
    
    without_loss,without_ppl=compute_ppl_single(redecoded_input+decoded_output_without_watermark,decoded_output_without_watermark,args.oracle_model_name,oracle_model,oracle_tokenizer)
    with_loss,with_ppl=compute_ppl_single(redecoded_input+decoded_output_with_watermark,decoded_output_with_watermark,args.oracle_model_name,oracle_model,oracle_tokenizer)


    return (redecoded_input,
            # int(truncation_warning),
            decoded_output_without_watermark,
            decoded_output_with_watermark,
            output_with_multi_watermark,
            without_ppl,
            without_loss,
            with_ppl,
            with_loss,
            args)
    # decoded_output_with_watermark)


def format_names(s):
    """Format names for the gradio demo interface"""
    s = s.replace("num_tokens_scored", "Tokens Counted (T)")
    s = s.replace("num_green_tokens", "# Tokens in Greenlist")
    s = s.replace("green_fraction", "Fraction of T in Greenlist")
    s = s.replace("z_score", "z-score")
    s = s.replace("p_value", "p value")
    s = s.replace("prediction", "Prediction")
    s = s.replace("confidence", "Confidence")
    return s


def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k, v in score_dict.items():
        if k == 'green_fraction':
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k == 'confidence':
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float):
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else:
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2, ["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1, ["z-score Threshold", f"{detection_threshold}"])
    return lst_2d


def detect(input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                           gamma=args.gamma,
                                           seeding_scheme=args.seeding_scheme,
                                           device=device,
                                           tokenizer=tokenizer,
                                           z_threshold=args.detection_z_threshold,
                                           normalizers=args.normalizers,
                                           ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                           select_green_tokens=args.select_green_tokens)
    if len(input_text) - 1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        # output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error", "string too short to compute metrics"]]
        output += [["", ""] for _ in range(6)]
        score_dict=None
    return score_dict


def detectMulti(input_text,args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = MultiWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                           gamma=args.gamma,
                                           seeding_scheme=args.seeding_scheme,
                                           device=device,
                                           tokenizer=tokenizer,
                                           z_threshold=args.detection_z_threshold,
                                           normalizers=args.normalizers,
                                           ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                           t=args.signal_period,
                                           select_green_tokens=args.select_green_tokens)
    
    # watermark_detector = WatermarkDetectorMD(vocab=list(tokenizer.get_vocab().values()),
    #                                        gamma=args.gamma,
    #                                        seeding_scheme=args.seeding_scheme,
    #                                        device=device,
    #                                        tokenizer=tokenizer,
    #                                        z_threshold=args.detection_z_threshold,
    #                                        mul_nums=2,
    #                                        normalizers=args.normalizers,
    #                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
    #                                     #    t=args.signal_period,
    #                                        select_green_tokens=args.select_green_tokens)
    
    #if len(input_text) - 1 > watermark_detector.min_prefix_len:
    score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        #output = list_format_scores(score_dict, watermark_detector.z_threshold)
    #else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        #output = [["Error", "string too short to compute metrics"]]
        #output += [["", ""] for _ in range(6)]
    # print(watermark_detector.min_prefix_len)
    # print("!!!")
    return score_dict



# def detectMulti(input_text,args,mul_nums=2, device=None, tokenizer=None):
#     """Instantiate the WatermarkDetection object and call detect on
#         the input text returning the scores and outcome of the test"""
#     # watermark_detector = MultiWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
#     #                                        gamma=args.gamma,
#     #                                        seeding_scheme=args.seeding_scheme,
#     #                                        device=device,
#     #                                        tokenizer=tokenizer,
#     #                                        z_threshold=args.detection_z_threshold,
#     #                                        normalizers=args.normalizers,
#     #                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
#     #                                        t=args.signal_period,
#     #                                        select_green_tokens=args.select_green_tokens)
    
#     watermark_detector = WatermarkDetectorMD(vocab=list(tokenizer.get_vocab().values()),
#                                            gamma=args.gamma,
#                                            seeding_scheme=args.seeding_scheme,
#                                            device=device,
#                                            tokenizer=tokenizer,
#                                            z_threshold=args.detection_z_threshold,
#                                            mul_nums=2,
#                                            normalizers=args.normalizers,
#                                            ignore_repeated_bigrams=args.ignore_repeated_bigrams,
#                                         #    t=args.signal_period,
#                                            select_green_tokens=args.select_green_tokens)
    
#     #if len(input_text) - 1 > watermark_detector.min_prefix_len:
#     score_dict = watermark_detector.detect(input_text)
#         # output = str_format_scores(score_dict, watermark_detector.z_threshold)
#         #output = list_format_scores(score_dict, watermark_detector.z_threshold)
#     #else:
#         # output = (f"Error: string not long enough to compute watermark presence.")
#         #output = [["Error", "string too short to compute metrics"]]
#         #output += [["", ""] for _ in range(6)]
#     # print(watermark_detector.min_prefix_len)
#     # print("!!!")
#     return score_dict



def run_gradio(args, model=None, device=None, tokenizer=None):
    """Define and launch the gradio demo interface"""
    generate_partial = partial(generate, model=model, device=device, tokenizer=tokenizer)
    detect_partial = partial(detect, device=device, tokenizer=tokenizer)

    with gr.Blocks() as demo:
        # Top section, greeting and instructions
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                    """
                    ## 💧 [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226) 🔍
                    """
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    [![](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/jwkirchenbauer/lm-watermarking)
                    """
                )
            # with gr.Column(scale=2):
            #     pass
            # ![visitor badge](https://visitor-badge.glitch.me/badge?page_id=tomg-group-umd_lm-watermarking) # buggy

        with gr.Accordion("Understanding the output metrics", open=False):
            gr.Markdown(
                """
                - `z-score threshold` : The cuttoff for the hypothesis test
                - `Tokens Counted (T)` : The number of tokens in the output that were counted by the detection algorithm. 
                    The first token is ommitted in the simple, single token seeding scheme since there is no way to generate
                    a greenlist for it as it has no prefix token(s). Under the "Ignore Bigram Repeats" detection algorithm, 
                    described in the bottom panel, this can be much less than the total number of tokens generated if there is a lot of repetition.
                - `# Tokens in Greenlist` : The number of tokens that were observed to fall in their respective greenlist
                - `Fraction of T in Greenlist` : The `# Tokens in Greenlist` / `T`. This is expected to be approximately `gamma` for human/unwatermarked text.
                - `z-score` : The test statistic for the detection hypothesis test. If larger than the `z-score threshold` 
                    we "reject the null hypothesis" that the text is human/unwatermarked, and conclude it is watermarked
                - `p value` : The likelihood of observing the computed `z-score` under the null hypothesis. This is the likelihood of 
                    observing the `Fraction of T in Greenlist` given that the text was generated without knowledge of the watermark procedure/greenlists.
                    If this is extremely _small_ we are confident that this many green tokens was not chosen by random chance.
                -  `prediction` : The outcome of the hypothesis test - whether the observed `z-score` was higher than the `z-score threshold`
                - `confidence` : If we reject the null hypothesis, and the `prediction` is "Watermarked", then we report 1-`p value` to represent 
                    the confidence of the detection based on the unlikeliness of this `z-score` observation.
                """
            )

        with gr.Accordion("A note on model capability", open=True):
            gr.Markdown(
                """
                This demo uses open-source language models that fit on a single GPU. These models are less powerful than proprietary commercial tools like ChatGPT, Claude, or Bard. 

                Importantly, we use a language model that is designed to "complete" your prompt, and not a model this is fine-tuned to follow instructions. 
                For best results, prompt the model with a few sentences that form the beginning of a paragraph, and then allow it to "continue" your paragraph. 
                Some examples include the opening paragraph of a wikipedia article, or the first few sentences of a story. 
                Longer prompts that end mid-sentence will result in more fluent generations.
                """
            )
        gr.Markdown(f"Language model: {args.model_name_or_path} {'(float16 mode)' if args.load_fp16 else ''}")

        # Construct state for parameters, define updates and toggles
        default_prompt = args.__dict__.pop("default_prompt")
        session_args = gr.State(value=args)

        with gr.Tab("Generate and Detect"):

            with gr.Row():
                prompt = gr.Textbox(label=f"Prompt", interactive=True, lines=10, max_lines=10, value=default_prompt)
            with gr.Row():
                generate_btn = gr.Button("Generate")
            with gr.Row():
                with gr.Column(scale=2):
                    output_without_watermark = gr.Textbox(label="Output Without Watermark", interactive=False, lines=14,
                                                          max_lines=14)
                with gr.Column(scale=1):
                    # without_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    without_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,
                                                                      row_count=7, col_count=2)
            with gr.Row():
                with gr.Column(scale=2):
                    output_with_watermark = gr.Textbox(label="Output With Watermark", interactive=False, lines=14,
                                                       max_lines=14)
                with gr.Column(scale=1):
                    # with_watermark_detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    with_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,
                                                                   row_count=7, col_count=2)

            redecoded_input = gr.Textbox(visible=False)
            truncation_warning = gr.Number(visible=False)

            def truncate_prompt(redecoded_input, truncation_warning, orig_prompt, args):
                if truncation_warning:
                    return redecoded_input + f"\n\n[Prompt was truncated before generation due to length...]", args
                else:
                    return orig_prompt, args

        with gr.Tab("Detector Only"):
            with gr.Row():
                with gr.Column(scale=2):
                    detection_input = gr.Textbox(label="Text to Analyze", interactive=True, lines=14, max_lines=14)
                with gr.Column(scale=1):
                    # detection_result = gr.Textbox(label="Detection Result", interactive=False,lines=14,max_lines=14)
                    detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False, row_count=7,
                                                    col_count=2)
            with gr.Row():
                detect_btn = gr.Button("Detect")

        # Parameter selection group
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"#### Generation Parameters")
                    with gr.Row():
                        decoding = gr.Radio(label="Decoding Method", choices=["multinomial", "greedy"],
                                            value=("multinomial" if args.use_sampling else "greedy"))
                    with gr.Row():
                        sampling_temp = gr.Slider(label="Sampling Temperature", minimum=0.1, maximum=1.0, step=0.1,
                                                  value=args.sampling_temp, visible=True)
                    with gr.Row():
                        generation_seed = gr.Number(label="Generation Seed", value=args.generation_seed,
                                                    interactive=True)
                    with gr.Row():
                        n_beams = gr.Dropdown(label="Number of Beams", choices=list(range(1, 11, 1)),
                                              value=args.n_beams, visible=(not args.use_sampling))
                    with gr.Row():
                        max_new_tokens = gr.Slider(label="Max Generated Tokens", minimum=10, maximum=1000, step=10,
                                                   value=args.max_new_tokens)

                with gr.Column(scale=1):
                    gr.Markdown(f"#### Watermark Parameters")
                    with gr.Row():
                        gamma = gr.Slider(label="gamma", minimum=0.1, maximum=0.9, step=0.05, value=args.gamma)
                    with gr.Row():
                        delta = gr.Slider(label="delta", minimum=0.0, maximum=10.0, step=0.1, value=args.delta)
                    gr.Markdown(f"#### Detector Parameters")
                    with gr.Row():
                        detection_z_threshold = gr.Slider(label="z-score threshold", minimum=0.0, maximum=10.0,
                                                          step=0.1, value=args.detection_z_threshold)
                    with gr.Row():
                        ignore_repeated_bigrams = gr.Checkbox(label="Ignore Bigram Repeats")
                    with gr.Row():
                        normalizers = gr.CheckboxGroup(label="Normalizations",
                                                       choices=["unicode", "homoglyphs", "truecase"],
                                                       value=args.normalizers)
            # with gr.Accordion("Actual submitted parameters:",open=False):
            with gr.Row():
                gr.Markdown(
                    f"_Note: sliders don't always update perfectly. Clicking on the bar or using the number window to the right can help. Window below shows the current settings._")
            with gr.Row():
                current_parameters = gr.Textbox(label="Current Parameters", value=args)
            with gr.Accordion("Legacy Settings", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed_separately = gr.Checkbox(label="Seed both generations separately",
                                                      value=args.seed_separately)
                    with gr.Column(scale=1):
                        select_green_tokens = gr.Checkbox(label="Select 'greenlist' from partition",
                                                          value=args.select_green_tokens)

        with gr.Accordion("Understanding the settings", open=False):
            gr.Markdown(
                """
                #### Generation Parameters:
    
                - Decoding Method : We can generate tokens from the model using either multinomial sampling or we can generate using greedy decoding.
                - Sampling Temperature : If using multinomial sampling we can set the temperature of the sampling distribution. 
                                    0.0 is equivalent to greedy decoding, and 1.0 is the maximum amount of variability/entropy in the next token distribution.
                                    0.7 strikes a nice balance between faithfulness to the model's estimate of top candidates while adding variety. Does not apply for greedy decoding.
                - Generation Seed : The integer to pass to the torch random number generator before running generation. Makes the multinomial sampling strategy
                                    outputs reproducible. Does not apply for greedy decoding.
                - Number of Beams : When using greedy decoding, we can also set the number of beams to > 1 to enable beam search. 
                                    This is not implemented/excluded from paper for multinomial sampling but may be added in future.
                - Max Generated Tokens : The `max_new_tokens` parameter passed to the generation method to stop the output at a certain number of new tokens. 
                                        Note that the model is free to generate fewer tokens depending on the prompt. 
                                        Implicitly this sets the maximum number of prompt tokens possible as the model's maximum input length minus `max_new_tokens`,
                                        and inputs will be truncated accordingly.
    
                #### Watermark Parameters:
    
                - gamma : The fraction of the vocabulary to be partitioned into the greenlist at each generation step. 
                         Smaller gamma values create a stronger watermark by enabling the watermarked model to achieve 
                         a greater differentiation from human/unwatermarked text because it is preferentially sampling 
                         from a smaller green set making those tokens less likely to occur by chance.
                - delta : The amount of positive bias to add to the logits of every token in the greenlist 
                            at each generation step before sampling/choosing the next token. Higher delta values 
                            mean that the greenlist tokens are more heavily preferred by the watermarked model
                            and as the bias becomes very large the watermark transitions from "soft" to "hard". 
                            For a hard watermark, nearly all tokens are green, but this can have a detrimental effect on
                            generation quality, especially when there is not a lot of flexibility in the distribution.
    
                #### Detector Parameters:
    
                - z-score threshold : the z-score cuttoff for the hypothesis test. Higher thresholds (such as 4.0) make
                                    _false positives_ (predicting that human/unwatermarked text is watermarked) very unlikely
                                    as a genuine human text with a significant number of tokens will almost never achieve 
                                    that high of a z-score. Lower thresholds will capture more _true positives_ as some watermarked
                                    texts will contain less green tokens and achive a lower z-score, but still pass the lower bar and 
                                    be flagged as "watermarked". However, a lowere threshold will increase the chance that human text 
                                    that contains a slightly higher than average number of green tokens is erroneously flagged. 
                                    4.0-5.0 offers extremely low false positive rates while still accurately catching most watermarked text.
                - Ignore Bigram Repeats : This alternate detection algorithm only considers the unique bigrams in the text during detection, 
                                        computing the greenlists based on the first in each pair and checking whether the second falls within the list.
                                        This means that `T` is now the unique number of bigrams in the text, which becomes less than the total
                                        number of tokens generated if the text contains a lot of repetition. See the paper for a more detailed discussion.
                - Normalizations : we implement a few basic normaliations to defend against various adversarial perturbations of the
                                    text analyzed during detection. Currently we support converting all chracters to unicode, 
                                    replacing homoglyphs with a canonical form, and standardizing the capitalization. 
                                    See the paper for a detailed discussion of input normalization. 
                """
            )

        gr.HTML("""
                <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. 
                    Follow the github link at the top and host the demo on your own GPU hardware to test out larger models.
                <br/>
                <a href="https://huggingface.co/spaces/tomg-group-umd/lm-watermarking?duplicate=true">
                <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                <p/>
                """)

        # Register main generation tab click, outputing generations as well as a the encoded+redecoded+potentially truncated prompt and flag
        generate_btn.click(fn=generate_partial, inputs=[prompt, session_args],
                           outputs=[redecoded_input, truncation_warning, output_without_watermark,
                                    output_with_watermark, session_args])
        # Show truncated version of prompt if truncation occurred
        redecoded_input.change(fn=truncate_prompt, inputs=[redecoded_input, truncation_warning, prompt, session_args],
                               outputs=[prompt, session_args])
        # Call detection when the outputs (of the generate function) are updated
        output_without_watermark.change(fn=detect_partial, inputs=[output_without_watermark, session_args],
                                        outputs=[without_watermark_detection_result, session_args])
        output_with_watermark.change(fn=detect_partial, inputs=[output_with_watermark, session_args],
                                     outputs=[with_watermark_detection_result, session_args])
        # Register main detection tab click
        detect_btn.click(fn=detect_partial, inputs=[detection_input, session_args],
                         outputs=[detection_result, session_args])

        # State management logic
        # update callbacks that change the state dict
        def update_sampling_temp(session_state, value):
            session_state.sampling_temp = float(value); return session_state

        def update_generation_seed(session_state, value):
            session_state.generation_seed = int(value); return session_state

        def update_gamma(session_state, value):
            session_state.gamma = float(value); return session_state

        def update_delta(session_state, value):
            session_state.delta = float(value); return session_state

        def update_detection_z_threshold(session_state, value):
            session_state.detection_z_threshold = float(value); return session_state

        def update_decoding(session_state, value):
            if value == "multinomial":
                session_state.use_sampling = True
            elif value == "greedy":
                session_state.use_sampling = False
            return session_state

        def toggle_sampling_vis(value):
            if value == "multinomial":
                return gr.update(visible=True)
            elif value == "greedy":
                return gr.update(visible=False)

        def toggle_sampling_vis_inv(value):
            if value == "multinomial":
                return gr.update(visible=False)
            elif value == "greedy":
                return gr.update(visible=True)

        def update_n_beams(session_state, value):
            session_state.n_beams = value; return session_state

        def update_max_new_tokens(session_state, value):
            session_state.max_new_tokens = int(value); return session_state

        def update_ignore_repeated_bigrams(session_state, value):
            session_state.ignore_repeated_bigrams = value; return session_state

        def update_normalizers(session_state, value):
            session_state.normalizers = value; return session_state

        def update_seed_separately(session_state, value):
            session_state.seed_separately = value; return session_state

        def update_select_green_tokens(session_state, value):
            session_state.select_green_tokens = value; return session_state

        # registering callbacks for toggling the visibilty of certain parameters
        decoding.change(toggle_sampling_vis, inputs=[decoding], outputs=[sampling_temp])
        decoding.change(toggle_sampling_vis, inputs=[decoding], outputs=[generation_seed])
        decoding.change(toggle_sampling_vis_inv, inputs=[decoding], outputs=[n_beams])
        # registering all state update callbacks
        decoding.change(update_decoding, inputs=[session_args, decoding], outputs=[session_args])
        sampling_temp.change(update_sampling_temp, inputs=[session_args, sampling_temp], outputs=[session_args])
        generation_seed.change(update_generation_seed, inputs=[session_args, generation_seed], outputs=[session_args])
        n_beams.change(update_n_beams, inputs=[session_args, n_beams], outputs=[session_args])
        max_new_tokens.change(update_max_new_tokens, inputs=[session_args, max_new_tokens], outputs=[session_args])
        gamma.change(update_gamma, inputs=[session_args, gamma], outputs=[session_args])
        delta.change(update_delta, inputs=[session_args, delta], outputs=[session_args])
        detection_z_threshold.change(update_detection_z_threshold, inputs=[session_args, detection_z_threshold],
                                     outputs=[session_args])
        ignore_repeated_bigrams.change(update_ignore_repeated_bigrams, inputs=[session_args, ignore_repeated_bigrams],
                                       outputs=[session_args])
        normalizers.change(update_normalizers, inputs=[session_args, normalizers], outputs=[session_args])
        seed_separately.change(update_seed_separately, inputs=[session_args, seed_separately], outputs=[session_args])
        select_green_tokens.change(update_select_green_tokens, inputs=[session_args, select_green_tokens],
                                   outputs=[session_args])
        # register additional callback on button clicks that updates the shown parameters window
        generate_btn.click(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        detect_btn.click(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        # When the parameters change, display the update and fire detection, since some detection params dont change the model output.
        gamma.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        gamma.change(fn=detect_partial, inputs=[output_without_watermark, session_args],
                     outputs=[without_watermark_detection_result, session_args])
        gamma.change(fn=detect_partial, inputs=[output_with_watermark, session_args],
                     outputs=[with_watermark_detection_result, session_args])
        gamma.change(fn=detect_partial, inputs=[detection_input, session_args],
                     outputs=[detection_result, session_args])
        detection_z_threshold.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        detection_z_threshold.change(fn=detect_partial, inputs=[output_without_watermark, session_args],
                                     outputs=[without_watermark_detection_result, session_args])
        detection_z_threshold.change(fn=detect_partial, inputs=[output_with_watermark, session_args],
                                     outputs=[with_watermark_detection_result, session_args])
        detection_z_threshold.change(fn=detect_partial, inputs=[detection_input, session_args],
                                     outputs=[detection_result, session_args])
        ignore_repeated_bigrams.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[output_without_watermark, session_args],
                                       outputs=[without_watermark_detection_result, session_args])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[output_with_watermark, session_args],
                                       outputs=[with_watermark_detection_result, session_args])
        ignore_repeated_bigrams.change(fn=detect_partial, inputs=[detection_input, session_args],
                                       outputs=[detection_result, session_args])
        normalizers.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        normalizers.change(fn=detect_partial, inputs=[output_without_watermark, session_args],
                           outputs=[without_watermark_detection_result, session_args])
        normalizers.change(fn=detect_partial, inputs=[output_with_watermark, session_args],
                           outputs=[with_watermark_detection_result, session_args])
        normalizers.change(fn=detect_partial, inputs=[detection_input, session_args],
                           outputs=[detection_result, session_args])
        select_green_tokens.change(lambda value: str(value), inputs=[session_args], outputs=[current_parameters])
        select_green_tokens.change(fn=detect_partial, inputs=[output_without_watermark, session_args],
                                   outputs=[without_watermark_detection_result, session_args])
        select_green_tokens.change(fn=detect_partial, inputs=[output_with_watermark, session_args],
                                   outputs=[with_watermark_detection_result, session_args])
        select_green_tokens.change(fn=detect_partial, inputs=[detection_input, session_args],
                                   outputs=[detection_result, session_args])

    demo.queue(concurrency_count=3)

    if args.demo_public:
        demo.launch(share=True)  # exposes app to the internet via randomly generated link
    else:
        demo.launch()

def tokenize_and_truncate(example: dict,
                          completion_length: int = None,
                          prompt_length: int = None,
                          hf_model_name: str = None,
                          tokenizer = None,
                          model_max_seq_len: int = 4096):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert "text" in example, "expects 'text' field to be present"
    # tokenize
    inputs = tokenizer.encode(example["text"], return_tensors="pt", truncation=True, max_length=model_max_seq_len)
    example.update({"untruncated_inputs": inputs})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs.shape[1]-1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs.shape[1]-1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError((f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                          f" but got completion_length:{completion_length},prompt_length:{prompt_length}"))

    # truncate
    inputs = inputs[:,:inputs.shape[1]-slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        inputs[0,-1] = 1
    # else: pass
    example.update({"inputs": inputs})
    return example

def check_input_lengths(example,idx, min_sample_len=0, min_prompt_len=0, min_completion_len=0):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["real_completion_length"]

    # breakpoint()

    conds = all([
        orig_sample_length >= min_sample_len,
        prompt_length >= min_prompt_len,
        real_completion_length >= min_completion_len,
    ])
    return conds

def tokenize_for_generation(example: dict,
                            idx: int,
                            tokenizer,
                            model,
                            max_new_tokens: int=None,
                            min_prompt_tokens: int=None,
                            hf_model_name : str=None
                           ):

    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    # preprocess for model generation/completion
    example = tokenize_and_truncate(example,
                                    completion_length=max_new_tokens,
                                    prompt_length=min_prompt_tokens,
                                    hf_model_name=hf_model_name,
                                    tokenizer=tokenizer,
                                    # model_max_seq_len=model.config.max_position_embeddings)
                                    model_max_seq_len=None)
    inputs = example["inputs"]
    # for calculating the baseline violation rate across the "gold" completion
    untruncated_inputs = example["untruncated_inputs"]

    # decode the preprocessed input to store for audit
    re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
    example.update({"truncated_input":re_decoded_input})

    # also decode the original suffix of the input for audit as the baseline
    decoded_untruncated_input = tokenizer.batch_decode(untruncated_inputs, skip_special_tokens=True)[0]
    example.update({"baseline_completion":decoded_untruncated_input.replace(re_decoded_input,"")})

    example.update({
        "orig_sample_length"            : untruncated_inputs.shape[1],
        "prompt_length"                 : inputs.shape[1],
        "real_completion_length"        : untruncated_inputs.shape[1] - inputs.shape[1],
    })
    return example



def compute_ppl_single(prefix_and_output_text = None,
                        output_text = None,
                        oracle_model_name = None,
                        oracle_model = None,
                        oracle_tokenizer = None):
    if oracle_model==None:
        return -1,-1

    with torch.no_grad():
        tokd_prefix = tokenize_and_truncate({"text":prefix_and_output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]
        tokd_inputs = tokd_prefix
        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenize_and_truncate({"text":output_text}, completion_length=0, hf_model_name=oracle_model_name, tokenizer=oracle_tokenizer, model_max_seq_len=oracle_model.config.max_position_embeddings)["inputs"]

        tokd_inputs = tokd_inputs.to(oracle_model.device)
        # make labels, mark if not including all positions
        tokd_labels = tokd_inputs.clone().detach()
        tokd_labels[:,:tokd_labels.shape[1]-tokd_suffix.shape[1]+1] = -100

        outputs = oracle_model(input_ids=tokd_inputs, labels=tokd_labels)
        loss = outputs.loss # avg CE loss all positions (except -100, TODO plz check that this is working correctly)
        ppl = torch.tensor(math.exp(loss))
    
    return loss.item(), ppl.item()

def main(args,dataset,model=None,tokenizer=None,device=None,oracle_model=None,oracle_tokenizer=None,oracle_device=None):
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    if model==None:
        model, tokenizer, device ,oracle_model,oracle_tokenizer,oracle_device= load_model(args)
   
    
    #result=[]

    input_check_kwargs = dict(
        # min_sample_len = min_prompt_tokens + max_new_tokens,
        min_sample_len = args.min_sample_tokens, # first line is a bug sometimes with large amounts
        min_prompt_len = 0,
        min_completion_len=args.max_new_tokens
    )
    if hasattr(args, 'file_name'):
        print("file_name 属性存在，值为:", args.file_name)
        file_name=args.file_name+'.csv'
     
    else:
        file_name= 'sub-1-gamma'+str(args.gamma)+'delta'+str(args.delta)+'tokens'+str(args.max_new_tokens)+'search'+str(args.n_beams)+'.csv'
        print("file_name 属性不存在"+"设置为："+file_name)
    file_path = os.path.join(args.output_dir, file_name)
         # 判断文件是否存在
    if os.path.exists(file_path):
        print(f"文件已存在：{file_path}")
        return None
    
   



    # input_check = partial(
    #     check_input_lengths,
    #     **input_check_kwargs
    # )

    # token_kwargs = dict(
    #     hf_model_name=args.model_name,
    #     tokenizer=tokenizer,
    #     model=model,
    # )

    # tokenize_prompts = partial(
    #     tokenize_for_generation,
    #     **token_kwargs
    # )



    # # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
    # tokenized_and_truncated_dataset = dataset.map(tokenize_prompts,
    #                                                        batched=False,
    #                                                        with_indices=True)

    # # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
    # input_length_filtered_dataset = tokenized_and_truncated_dataset.filter(input_check,
    #                                                                        batched=False,
    #                                                                        with_indices=True)


    #处理输入数据
    ds_iterator = iter(dataset)
    # monitor=dict()
    # monitor.update(T100=0)
    # monitor.update(T200=0)
    

    data_batch=[]

    #for example in ds_iterator:
    for index, example in enumerate(ds_iterator):
        # inputs = example["inputs"]

        # term_width = 80
        # print("#" * term_width)
        # print("Prompt:")
        # print(inputs)
        max_retries = 10000  # 设置最大重试次数
        retry_count = 0
        while retry_count < max_retries:
            try:
                prompt_text, decoded_output_without_watermark, decoded_output_with_watermark, output_with_multi_watermark,without_ppl,without_loss,with_ppl,with_loss_,_ = generate(example,
                                                                                            args,
                                                                                            model=model,
                                                                                            device=device,
                                                                                            tokenizer=tokenizer,
                                                                                            oracle_model=oracle_model,
                                                                                            oracle_tokenizer=oracle_tokenizer)
                                                                                      
            
                break  # 如果成功执行，跳出循环
            except Exception as e:
                print(f"生成时出现异常: {e}，正在重试... (第 {retry_count + 1} 次)")
                retry_count += 1
                if retry_count >= max_retries:
                    print("达到最大重试次数，操作失败")
                    break

        
        # prompt_text, decoded_output_without_watermark, decoded_output_with_watermark, without_ppl,without_loss,with_ppl,with_loss_,_ = generate(example,
        #                                                                                     args,
        #                                                                                     model=model,
        #                                                                                     device=device,
        #                                                                                     tokenizer=tokenizer,
        #                                                                                     oracle_model=oracle_model,
        #                                                                                     oracle_tokenizer=oracle_tokenizer
        #                                                                               )
        # print(example)
        # print(decoded_output_without_watermark)
        # print(decoded_output_with_watermark)
        
        # print("no water"+decoded_output_without_watermark)
        # print("with water"+decoded_output_with_watermark)
        # print("multi water"+output_with_multi_watermark)
        if decoded_output_without_watermark==None or decoded_output_with_watermark==None or output_with_multi_watermark==None:
            continue;
        
        
        without_watermark_detection_result = detect(decoded_output_without_watermark,
                                                    args,
                                                    device=device,
                                                    tokenizer=tokenizer)
        with_watermark_detection_result = detect(decoded_output_with_watermark,
                                                 args,
                                                 device=device,
                                                 tokenizer=tokenizer)
        
        
        
        
        # without_watermar_2d_detection_result = detectMulti(decoded_output_without_watermark,
        #                                             args,
        #                                             mul_nums=2,
        #                                             device=device,
        #                                             tokenizer=tokenizer)
        # with_watermark_2d_detection_result=detectMulti(decoded_output_with_watermark,
        #                                          args,
        #                                          mul_nums=2,
        #                                          device=device,
        #                                          tokenizer=tokenizer)
        
        
        
        without_watermar_multi_detection_result = detectMulti(decoded_output_without_watermark,
                                                    args,
                                                    device=device,
                                                    tokenizer=tokenizer)
        
        with_watermark_multi_detection_result=detectMulti(output_with_multi_watermark,
                                                 args,
                                                 device=device,
                                                 tokenizer=tokenizer)
        
        detect_to_save={}
        # print(without_watermark_detection_result)
        # print(with_watermark_detection_result)
        # print(without_watermar_multi_detection_result)
        # print(with_watermark_multi_detection_result)
        if without_watermark_detection_result!=None and with_watermark_detection_result!=None and without_watermar_multi_detection_result!=None and with_watermark_multi_detection_result !=None:
            if without_watermark_detection_result["gen_len"]==None or  with_watermark_detection_result["gen_len"]==None or with_watermark_multi_detection_result["gen_len"]==None:
                continue
            if ((args.min_sample_tokens-5)<without_watermark_detection_result["gen_len"]<=(args.min_sample_tokens+5)) and ((args.min_sample_tokens-5)<with_watermark_detection_result["gen_len"]<=(args.min_sample_tokens+5)
                                                                                                                           and ((args.min_sample_tokens-5)<with_watermark_multi_detection_result["gen_len"]<=(args.min_sample_tokens+5))):    
                print("正在存储第："+str(len(data_batch)))
                detect_to_save["without_len"]=without_watermark_detection_result["gen_len"]
                detect_to_save["without_num_tokens_scored"]=without_watermark_detection_result["num_tokens_scored"]
                detect_to_save["without_num_green_tokens"]=without_watermark_detection_result["num_green_tokens"]
        
        
                detect_to_save["with_len"]=with_watermark_detection_result["gen_len"]        
                detect_to_save["with_num_tokens_scored"]=with_watermark_detection_result["num_tokens_scored"]
                detect_to_save["with_num_green_tokens"]=with_watermark_detection_result["num_green_tokens"]
                
                
                
                
        

                detect_to_save["without_high_num_tokens_scored"]=without_watermar_multi_detection_result["high_num_tokens_scored"]
                detect_to_save["without_low_num_tokens_scored"]=without_watermar_multi_detection_result["low_num_tokens_scored"]  
                detect_to_save["without_high_num_green_tokens"]=without_watermar_multi_detection_result["high_num_green_tokens"]
                detect_to_save["without_low_num_green_tokens"]=without_watermar_multi_detection_result["low_num_green_tokens"]
                
                
                detect_to_save["with_multi_len"]=with_watermark_multi_detection_result["gen_len"]        
                detect_to_save["with_high_num_green_tokens"]=with_watermark_multi_detection_result["high_num_green_tokens"]
                detect_to_save["with_low_num_green_tokens"]=with_watermark_multi_detection_result["low_num_green_tokens"]        
                detect_to_save["with_high_num_tokens_scored"]=with_watermark_multi_detection_result["high_num_tokens_scored"]
                detect_to_save["with_low_num_tokens_scored"]=with_watermark_multi_detection_result["low_num_tokens_scored"]
                
                
                # new_without_watermar_multi_detection_result = {'without_'+key: value for key, value in without_watermar_2d_detection_result.items()}
                # new_with_watermar_multi_detection_result = {'with_'+key: value for key, value in with_watermark_2d_detection_result.items()}
                # detect_to_save.update(new_without_watermar_multi_detection_result)
                # detect_to_save.update(new_with_watermar_multi_detection_result)
                detect_to_save['text']=prompt_text
                detect_to_save['without_watermark']=decoded_output_without_watermark
                detect_to_save['with_watermark']=decoded_output_with_watermark
                detect_to_save['with_multi_watermark']=output_with_multi_watermark



        
                # exit()
                data_batch.append(detect_to_save)
                if len(data_batch) % 250 == 0:
                    print(f"Generating with {args}")
                    # 获取字段名（假设所有字典的键一致）
                    fieldnames = data_batch[0].keys()

                    # 将数据写入 CSV 文件
                    with open(os.path.join(args.output_dir, file_name), "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data_batch)
                    
                    print(f"已将数据写入 {os.path.join(args.output_dir, file_name)}")
        
                    # 清空数据批处理列表以准备下一组数据
                    # data_batch = []
                    # file_count += 1
                    return data_batch

                    #print(detect_to_save)
        
        # print(without_watermark_detection_result)
        # print(with_watermark_detection_result)
        
        # print(without_watermar_multi_detection_result)
        # print(with_watermark_multi_detection_result)

        

        # print("#" * term_width)
        # print("Output without watermark:")
        # print(decoded_output_without_watermark)
        # print("-" * term_width)
        # print(f"Detection result @ {args.detection_z_threshold}:")
        # pprint(without_watermark_detection_result)
        # print("-" * term_width)

        # print("#" * term_width)
        # print("Output with watermark:")
        # print(decoded_output_with_watermark)
        # print("-" * term_width)
        # print(f"Detection result @ {args.detection_z_threshold}:")
        # pprint(with_watermark_detection_result)
        # print("-" * term_width)

        #result.append(detect_to_save)

    
    
# 处理最后一批（如果有的话）
    if data_batch:
        # 获取字段名（假设所有字典的键一致）
        fieldnames = data_batch[0].keys()

        # 将数据写入 CSV 文件
        with open(os.path.join(args.output_dir, file_name), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_batch)        
            print(f"已将数据写入 {os.path.join(args.output_dir, file_name)}")
    return data_batch



if __name__ == "__main__":
    args = parse_args()
    print(args)

    # result=main(args)
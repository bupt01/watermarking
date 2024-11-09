import torch
import math
import os
import csv
import statistics
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)
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
    return ppl.item()


# Specify the directory containing .csv files
directory_path = './output/'  # Update with the path to your folder
result_path='./output/ppl_new.txt'




if __name__ == "__main__":
    oracle_model_name = 'facebook/opt-2.7b'
    oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name)
    gpu_index = 1 # 例如，使用第二个GPU
    device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
    oracle_model = oracle_model.to(device)
    oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
    # Iterate through each file in the directory
    for filename in os.listdir(directory_path): 
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)

            # Read and process the CSV file row by row
            ppls_withoutwater=[]
            ppls_withwater=[]
            ppls_withmultiwater=[]
            with open(file_path, mode='r', newline='') as infile:
                reader = csv.DictReader(infile)
                data = list(reader)  # 将内容读取到列表中
                torch.cuda.empty_cache()
                ppls_withmultiwater = [compute_ppl_single(row['text']+row['with_multi_watermark'],row['with_multi_watermark'],oracle_model_name,oracle_model=oracle_model,oracle_tokenizer=oracle_tokenizer) for row in data]  # Process each row
                ppls_withoutwater = [compute_ppl_single(row['text']+row['without_watermark'],row['without_watermark'],oracle_model_name,oracle_model=oracle_model,oracle_tokenizer=oracle_tokenizer) for row in data]  # Process each row
                ppls_withwater = [compute_ppl_single(row['text']+row['with_watermark'],row['with_watermark'],oracle_model_name,oracle_model=oracle_model,oracle_tokenizer=oracle_tokenizer) for row in data]  # Process each row

            # 打开文件并写入
            # with open('output-2.txt', 'w') as f:
            #     for a, b, c in zip(ppls_withoutwater, ppls_withwater, ppls_withmultiwater):
            #         # 将每一行的三个元素写入文件，使用空格分隔
            #         f.write(f"{a} {b} {c}\n")
            with open(result_path, 'a') as file:
                avg_withoutwater = sum(ppls_withoutwater)/len(ppls_withoutwater)
                avg_withwater = sum(ppls_withwater)/len(ppls_withwater)
                avg_withmultiwater = sum(ppls_withmultiwater)/len(ppls_withmultiwater)
                print(filename+' '+str(avg_withmultiwater)+' '+str(avg_withwater)+' '+str(avg_withoutwater))
                file.write(filename+'       '+str(avg_withoutwater)+'     '+str(avg_withwater)+'         '+str(avg_withmultiwater))
                file.write('\n')

                
            

print("All .csv files processed.")

    

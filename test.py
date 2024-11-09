# # from datasets import load_dataset, Dataset


# # dataset = load_dataset('c4', 'realnewslike ', split="train", streaming=True)
# # indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)
# import os
# import json

# # 目录路径
# directory_path = '/project/lm-watermarking-main/output/'  # 替换为你的目录路径

# # 存储所有数据的列表
# all_data = []

# # 遍历目录中的每个文件
# for filename in os.listdir(directory_path):
#     if filename.endswith('.json'):
#         file_path = os.path.join(directory_path, filename)
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 # 读取 JSON 数据
#                 data = json.load(file)
                
#                 # 假设每个 JSON 文件的内容都是一个字典，将字典的值添加到列表中
#                 all_data.append(data)
        
#         except json.JSONDecodeError:
#             print(f"Error decoding JSON from file: {file_path}")
#         except Exception as e:
#             print(f"An error occurred while processing file: {file_path} - {e}")

# # 打印所有数据
# print(all_data)

#
# import os
#
# # 替换为你的文件夹路径
# folder_path = 'F:\lz\data\learning fix\datasets-50-100\\50-100\source_code\\50-100\\buggy'
# # 替换为你的要检查的字符串列表
# string_list = ['atomic', 'violation', 'thread','concurrency','race']
#
# # 用于存储匹配的文件名
# matching_files = []
#
# # 遍历文件夹下的所有文件
# for filename in os.listdir(folder_path):
#     # 检查文件名是否包含列表中的任意一个字符串
#     if any(string in filename for string in string_list):
#         matching_files.append(filename)
#
# # 打印匹配的文件名
# print("Matching files:")
# for file in matching_files:
#     print(file)

import csv

# 假设这是实验数据，每个字典代表一行数据
data = [{'without_len': 50,
'without_num_tokens_scored': 49,
'without_num_green_tokens': 8,
'with_len': 50,
'with_num_tokens_scored': 49,
'with_num_green_tokens': 15,
'without_num_tokens_scored_0': -1,
'without_num_green_tokens_0': 0,
'without_num_tokens_scored_1': 24,
'without_num_green_tokens_1': 2,
'without_gen_len': 50, 'with_num_tokens_scored_0': -1,
'with_num_green_tokens_0': 0,
'with_num_tokens_scored_1': 24,
'with_num_green_tokens_1': 3,
'with_gen_len': 50,
'text': 'After the martyrdom of St. Boniface, Vergilius was made Bishop of Salzburg (766 or 767) and laboured successfully for the upbuilding of his diocese as well as for the spread of the Faith in neighbouring heathen countries, especially in Carinthia. He died at Salzburg, 27 November, 789. In 1233 he was canonized by Gregory IX. His doctrine that the earth is a sphere was derived from the teaching of ancient geographers, and his belief in the existence of the antipodes was probably influenced by the accounts which the ancient Irish voyagers gave of their journeys. This, at least, is the opinion of Rettberg ("Kirchengesch. Deutschlands", II, 236).',
'without_watermark': '\n\nThe monastery of St. Boniface on Salzburg Hill was founded in 1035 (see Salzburg monastery and church). The first Benedictine abbot was Heinrich, Bishop of Salzburg (1035-1040',
'with_watermark': '\n\nThe most important works of Saint Vergilius, are the following:\n\n1. "De Eusebii et Nilamensium contra ab dico et bono fidei"; "On the Faith and Doctrine of the'}]

# 获取字段名（假设所有字典的键一致）
fieldnames = data[0].keys()

# 将数据写入 CSV 文件
with open("output.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print("数据已存储到 output.csv 文件中")

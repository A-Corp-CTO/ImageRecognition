from groupN2_mask_creation import create_mask
import random 
import os 
from matplotlib import pyplot as plt 
import json 
from joblib import Parallel, delayed
import multiprocessing
import time 
import pandas as pd 
import skimage
import numpy as np
from math import floor

file_list = [i for i in os.walk("../data/test_data")][0][2]
test_mask_list = [i for i in os.walk("../data/mask_tests")][0][2]


tmp = []
for i in test_mask_list:
    tmp.append(i.split("_")[1])
tmp_list ={}
for i in tmp: 
    if i in tmp_list.keys():
        tmp_list[i] += 1 
    else: 
        tmp_list[i] = 1
# for i in tmp_list:
#     print(tmp_list[i])
test_mask_list = [i for i in test_mask_list if tmp_list[i.split("_")[1]] == 540]
test_mask_unique = []
for i in test_mask_list:
    if i.split("_")[1] not in test_mask_unique:
        test_mask_unique.append(i.split("_")[1])


test_results = {
    "image": [],
    "segments": [],
    "disk_size": [],
    "slic_sigma": [],
    "compactness": [],
    "cut_off": [],
    "x_dim": [],
    "y_dim": [],
    "pic_area": [],
    "false_pos": [],
    "false_neg": [],
    "true_mask_area": [],
    "common_area": [],
    "area": [],
    "area_diff": [],
    "attribute_set": []
}


true_masks = {}
attr_sets = {}
for i in range(1,541): 
    with open(f"../data/test_data/{i}.json","r") as infile: 
        attr_sets[i] = json.load(infile)
for i in test_mask_unique:
    true_masks[i] = plt.imread(f"../data/example_segmentation/ISIC_{i}_segmentation.png")


base_path = "../data/mask_tests/"

num_cores = floor(multiprocessing.cpu_count()/2)
inputs = sorted(test_mask_list)
def processInput(im): 
    # print(im)
    start_time = time.time()
    tmp = im[:-4].split("_")
    
    test_mask = plt.imread(f"{base_path}{im}")
    tmp_results = {
    "image": f"{tmp[0]}_{tmp[1]}",
    "segments": 0,
    "disk_size": 0,
    "slic_sigma": 0,
    "compactness": 0,
    "cut_off": 0,
    "x_dim": len(test_mask[0]),
    "y_dim": len(test_mask),
    "pic_area": len(test_mask[0]) * len(test_mask),
    "false_pos": 0,
    "false_neg": 0,
    "area": 0,
    "true_mask_area": 0,
    "common_area": 0,
    "area_diff": 0,
    "attribute_set": 0
    }
    for i in attr_sets[int(tmp[2])]: 
        tmp_results[i] = attr_sets[int(tmp[2])][i]
    
    # tes_mask = np.zeros((len(test_mask),len(test_mask[0])))
    # for i in range(len(test_mask)):
    #     for j in range(len(test_mask[0])):
    #         if test_mask[i,j,0] == 1:
    #             tmp_mask[i,j] = 1
    tmp_mask = np.dot(test_mask[...,:3], [1, 0, 0])
    for i in range(tmp_results["y_dim"]):
        for j in range(tmp_results["x_dim"]):

            if tmp_mask[i,j] == 1 and true_masks[tmp[1]][i,j] == 0:
                tmp_results["false_pos"] += 1 
                tmp_results["area"] += 1 
            elif tmp_mask[i,j] == 0 and true_masks[tmp[1]][i,j] == 1:
                tmp_results["false_neg"] += 1
                tmp_results["true_mask_area"] += 1
            elif tmp_mask[i,j] == 0 and true_masks[tmp[1]][i,j] == 0:
                pass 
            else: 
                tmp_results["true_mask_area"] += 1
                tmp_results["common_area"] += 1 
                tmp_results["area"]
    tmp_results["area_diff"] = tmp_results["false_neg"] + tmp_results["false_pos"] 
    print(f"{im} - {start_time - time.time()}")
    with open(f"./tmp_data/{im[:-4]}.json", "w") as outfile:
        json.dump(tmp_results,outfile,indent=4)
    return 
Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
# for i in results:
#     for j in i:
#         test_results[j].append(i[j])

# with open("test_results_json.json", "w") as outfile: 
#     json.dump(test_results,outfile,indent=4)

# tmp_df = pd.DataFrame(test_results)
# tmp_df.to_csv("./test_results.csv",sep=";",index=False)

# for im in sorted(test_mask_list):
#     start_time = time.time()
#     tmp = im[:-4].split("_")
    
#     test_mask = plt.imread(f"{base_path}{im}")
#     tmp_results = {
#     "image": f"{tmp[0]}_{tmp[1]}",
#     "segments": 0,
#     "disk_size": 0,
#     "slic_sigma": 0,
#     "compactness": 0,
#     "cut_off": 0,
#     "x_dim": len(test_mask[0]),
#     "y_dim": len(test_mask),
#     "pic_area": len(test_mask[0]) * len(test_mask),
#     "false_pos": 0,
#     "false_neg": 0,
#     "area": 0,
#     "true_mask_area": 0,
#     "common_area": 0,
#     "area_diff": 0,
#     "attribute_set": 0
#     }
#     for i in attr_sets[int(tmp[2])]: 
#         tmp_results[i] = attr_sets[int(tmp[2])][i]
    
#     # tes_mask = np.zeros((len(test_mask),len(test_mask[0])))
#     # for i in range(len(test_mask)):
#     #     for j in range(len(test_mask[0])):
#     #         if test_mask[i,j,0] == 1:
#     #             tmp_mask[i,j] = 1
#     tmp_mask = np.dot(test_mask[...,:3], [1, 0, 0])
#     for i in range(tmp_results["y_dim"]):
#         for j in range(tmp_results["x_dim"]):

#             if tmp_mask[i,j] == 1 and true_masks[tmp[1]][i,j] == 0:
#                 tmp_results["false_pos"] += 1 
#                 tmp_results["area"] += 1 
#             elif tmp_mask[i,j] == 0 and true_masks[tmp[1]][i,j] == 1:
#                 tmp_results["false_neg"] += 1
#                 tmp_results["true_mask_area"] += 1
#             elif tmp_mask[i,j] == 0 and true_masks[tmp[1]][i,j] == 0:
#                 pass 
#             else: 
#                 tmp_results["true_mask_area"] += 1
#                 tmp_results["common_area"] += 1 
#                 tmp_results["area"]
#     tmp_results["area_diff"] = tmp_results["false_neg"] + tmp_results["false_pos"] 
#     for i in tmp_results:
#         test_results[i].append(tmp_results[i])     
#     # with open("tmp_results.json","w") as outfile:
#     #     json.dump(test_results,outfile,indent=4)
#     print(f"{im} - {start_time - time.time()}")
    # print(sum(sum(mask-tmp_mask)))

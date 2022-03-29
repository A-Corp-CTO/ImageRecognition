from groupN2_mask_creation import create_mask
import random 
import os 
from matplotlib import pyplot as plt 
import json 
from joblib import Parallel, delayed
import multiprocessing
import time 
segments = [i for i in range(0,500,50)][1:]
disk_size = [i for i in range(0,5)][1:]
slic_sigma = [1,2,3]
compactness = [0.01,0.1,1,10,100]
cut_off = [0.6]

attr = [[i,j,k,l,m] for i in segments 
                    for j in disk_size 
                    for k in slic_sigma
                    for l in compactness
                    for m in cut_off]
attribute_set = 1
for i in attr: 
    # print(attribute_set)
    tmp_dict = {
        "segments": i[0],
        "disk_size": i[1], 
        "slic_sigma": i[2], 
        "compactness": i[3], 
        "cut_off": i[4],
        "attribute_set": attribute_set
    }
    with open(f"../data/test_data/{attribute_set}.json", "w") as outfile:
        json.dump(tmp_dict,outfile)
    attribute_set += 1


def find_images(path, number, seed,): 
    im_list = os.walk(path)
    im_list = [i for i in im_list][0][2]
    random.seed(seed)
    random_list = []
    while len(random_list) != number: 
        tmp_random = random.randint(0,len(im_list)-1)
        if im_list[tmp_random] in random_list:
            continue 
        else: 
            random_list.append(im_list[tmp_random])
    return random_list

def generate_mask(in_path, out_path, attributes,im): 
    im_name = f"{in_path}/{im}"
    mask = create_mask(im_name, attributes['segments'],attributes['disk_size'],attributes['slic_sigma'],attributes['compactness'],attributes['cut_off'])
    out_name = im.split(".")
    out_name = f"{out_path}/{out_name[0]}_{attributes['attribute_set']}.png"
    plt.imsave(out_name,mask,cmap="gray")


def processInput(i,in_path,out_path,attr_path,im_list):
    with open(f"{attr_path}/{i}","r") as infile: 
        tmp_json = json.load(infile)
    
    for j in im_list:
        print(f"{tmp_json['attribute_set']} - {j}")
        generate_mask(in_path,out_path,tmp_json,j)


def test_all_attributes(attr_path,in_path,out_path,number,seed):
    start_time = time.time()        
    attr_list = os.walk(attr_path)
    attr_list = sorted([i for i in attr_list][0][2])
    im_list = find_images(in_path,number,seed)
    print(len(attr_list))
    print(len(im_list))
    # num_cores = multiprocessing.cpu_count()
    # Parallel(n_jobs=num_cores/2)(delayed(processInput)(i,in_path,out_path,attr_path,number,seed) for i in attr_list)
    for i in attr_list:
        with open(f"{attr_path}/{i}","r") as infile: 
            tmp_json = json.load(infile)
        
        for j in im_list:
            print(f"{tmp_json['attribute_set']} - {j} - {time.time() - start_time:5.3f}")
            generate_mask(in_path,out_path,tmp_json,j)
        
        
def test_all_attributes_parallel(attr_path,in_path,out_path,number,seed):
    attr_list = os.walk(attr_path)
    attr_list = sorted([i for i in attr_list][0][2])
    im_list = find_images(in_path,number,seed)
    print(len(attr_list))
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=6)(delayed(processInput)(i,in_path,out_path,attr_path,im_list) for i in attr_list)
    # for i in attr_list:
    #     with open(f"{attr_path}/{i}","r") as infile: 
    #         tmp_json = json.load(infile)
        
    #     for j in im_list:
    #         print(f"{tmp_json['attribute_set']} - {j}")
    #         generate_mask(in_path,out_path,tmp_json,j)        


test_all_attributes('../data/test_data','../data/example_image','../data/mask_tests',10,7)
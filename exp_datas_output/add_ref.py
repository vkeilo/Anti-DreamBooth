import os
# import sys 
fsmg_exp_out_path = "/root/xinglin-data/github/Anti-DreamBooth/exp_datas_output/fsmg_VGGFace2_random50v1_r6_copy"
data_path = "/root/xinglin-data/github/data/VGGFace2-clean"

for dir in os.listdir(fsmg_exp_out_path):
    id = dir.split('-')[1][2:]
    print(id)
    os.system(f"mkdir {fsmg_exp_out_path}/{dir}/image_clean")
    os.system(f"cp {data_path}/{id}/set_A/* {fsmg_exp_out_path}/{dir}/image_clean -r" )
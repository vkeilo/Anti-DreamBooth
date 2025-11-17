import os
import json
import time
import re
import argparse
import numpy as np

def set_env_with_expansion(key, value):
    """
    Replace ${VAR_NAME} in the environment variable value with the corresponding environment variable value and set the new environment variable.
    """
    # 使用正则表达式找到 ${VAR_NAME} 并替换为对应的环境变量值
    pattern = re.compile(r'\$\{([^}]+)\}')
    expanded_value = pattern.sub(lambda match: os.getenv(match.group(1), match.group(0)), value)
    
    # 设置环境变量
    os.environ[key] = expanded_value
    print(f"Environment variable '{key}' set to: {expanded_value}")

def test_one_args(args,test_lable):
    for k,v in args.items():
        if "$" in str(v):
            set_env_with_expansion(k,v)
        else:
            os.environ[k] = str(v)
    # os.chdir("..")
    # bash run : nohup bash script/gen_and_eval_vk.sh > output_MAT-1000-200-6-6-x1x1-radius11-allSGLD-rubust0.log 2>&1
    os.environ["test_timestamp"] = str(int(time.time()))
    test_timestamp = os.getenv("test_timestamp")
    run_name = f"-id{os.getenv('data_id')}-r{os.getenv('r')}-steps{os.getenv('attack_steps')}-{os.getenv('test_timestamp')}"
    # 唯一的tmp目录标识
    unique_tmp_dir = f"tmp_ADVERSARIAL_{test_timestamp}"  
    os.environ["UNIQUE_TMP_DIR"] = unique_tmp_dir 
    if os.getenv('attack_mode') in ["aspl"]:
        run_name = "ASPL" + run_name
    elif os.getenv('attack_mode') in ["fsmg"]:
        run_name = "FSGM" + run_name
    else:
        exit(f"attack_mode {os.getenv('attack_mode')} not support")
    os.environ["wandb_run_name"] = run_name  
    #export test_timestamp=$(date +%s)
    print(f"run_name: {run_name}")
    if os.getenv('attack_mode') in ["aspl"]:
        os.system(f"bash run_in_batch/attack_with_aspl.sh > output_{run_name}.log 2>&1")
    if os.getenv('attack_mode') in ["fsmg"]:
        os.system(f"bash run_in_batch/attack_with_fsmg.sh > output_{run_name}.log 2>&1")
    check_file_for_pattern(f"output_{run_name}.log","exp finished")
    # rename dir exp_data to exp_data_{run_name}
    # check dir exp_datas_output exist
    if not os.path.exists("exp_datas_output"):
        os.mkdir("exp_datas_output")
    if not os.path.exists(f"exp_datas_output/{test_lable}"):
        os.mkdir(f"exp_datas_output/{test_lable}")
    if not os.path.exists("logs_output"):
        os.mkdir("logs_output")
    if not os.path.exists(f"logs_output/{test_lable}"):
        os.mkdir(f"logs_output/{test_lable}")
    if os.getenv('attack_mode') in ["aspl"]:
        os.system(f"mv outputs/ASPL/{unique_tmp_dir} exp_datas_output/{test_lable}/exp_data_{run_name}")
        # os.system(f"cp -r outputs/ASPL/{unique_tmp_dir} exp_datas_output/{test_lable}/exp_data_{run_name}")
        # os.system(f"rm -r outputs/ASPL/{unique_tmp_dir}")
    if os.getenv('attack_mode') in ["fsmg"]:
        os.system(f"mv outputs/FSMG/{unique_tmp_dir}/checkpoint* outputs/FSMG/")
        os.system(f"mv outputs/FSMG/{unique_tmp_dir} exp_datas_output/{test_lable}/exp_data_{run_name}")
        os.system(f"mkdir outputs/FSMG/{unique_tmp_dir}")
        os.system(f"mv outputs/FSMG/checkpoint* outputs/FSMG/{unique_tmp_dir}/")
    os.system(f"mv output_{run_name}.log logs_output/{test_lable}/output_{run_name}.log")
    return run_name

def update_finished_json(finished_log_json_path, run_name):
    if not os.path.exists(finished_log_json_path):
        with open(finished_log_json_path, "w") as f:
            json.dump({}, f)
    finished_file = json.load(open(finished_log_json_path))
    # if json is empty, add key finished_args_list and value []
    if "finished_args_list" not in finished_file:
        finished_file["finished_args_list"] = []
    finished_file["finished_args_list"].append(run_name)
    json.dump(finished_file, open(finished_log_json_path, "w"))

def update_untest_json(untest_args_json_path):
    json_dict = json.load(open(untest_args_json_path))
    json_dict["untest_args_list"].pop(0)
    json.dump(json_dict, open(untest_args_json_path, "w"))

def check_file_for_pattern(file_path, pattern="find function last"):
    while True:
        try:
            # 打开文件并读取最后一行
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()  # 获取最后一行并去除两边空白
                    print(f"Last line detected:{last_line}")
                    # 检查最后一行是否以指定模式开头
                    if last_line.startswith(pattern):
                        print("Find a matching line and exit the test.")
                        return last_line
        except Exception as e:
            print(f"No matching row found, wait 3 minutes and recheck...(task error, please check task settings:{e})")
        
        # 等待 3 分钟（180 秒）
        print("")
        time.sleep(180)

if __name__ == "__main__":
    print("batch test start...")
    # run in dir MetaCloak
    ADB_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    Pro_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.environ["ADB_PROJECT_ROOT"] = ADB_path
    os.environ["PYTHONPATH"] = str(os.getenv("PYTHONPATH")) + ":" + ADB_path + ":" + Pro_path
    
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="untest.json")
    parser.add_argument("--save_path", type=str, default="finished.json")
    parser.add_argument("--device_n", type=str, default="0")

    os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().device_n
    untest_args_json_path = "run_in_batch/"+ parser.parse_args().target
    finished_log_json_path = "run_in_batch/" + parser.parse_args().save_path
    untest_file_con = json.load(open(untest_args_json_path))
    untest_args_list = untest_file_con["untest_args_list"].copy()
    test_lable = untest_file_con["test_lable"]
    for args in untest_args_list:
        print(f"start run :{args}")
        finished_name = test_one_args(args,test_lable)
        print(f"finished run :{finished_name}")
        update_untest_json(untest_args_json_path)
        update_finished_json(finished_log_json_path, finished_name)
    if not os.path.exists("finished_test"):
        os.mkdir("finished_test")
    os.system(f"mv {untest_args_json_path} finished_test")



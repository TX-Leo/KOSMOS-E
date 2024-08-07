import json  
import os  
from tqdm import tqdm
import string
import pdb  
import re

def process_json_file(json_file_name, tsv_save_root_path):  
    with open(json_file_name, "r") as file:  
        dataset = json.load(file)  
  
    
    file_counter = 0  
    line_counter = 0  
    basename = os.path.basename(json_file_name)[:-5]
    save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.json")
    output_file = open(save_name, "w")  
    print(f"Writing to {save_name}")  
    for data in tqdm(dataset):  
        ins_string = data['instruction'].replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        inp_string = data['input'].replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        out_string = data['output'].replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        
        end_sym = [':', '.', '!', '?']
        # pdb.set_trace()
        if ins_string and ins_string[-1] not in string.punctuation:
            # print(f'Instruction string: {ins_string}')
            ins_string += "\n"
        if inp_string and inp_string[-1] not in string.punctuation:
            # print(f'Ins & Input string: {ins_string} {inp_string}')
            inp_string += "\n"
        # if out_string and out_string[-1] not in string.punctuation:
        #     print(f'Output string: {out_string}')
            
        reform_dict = {}
        reform_dict["input"] = ins_string + inp_string
        reform_dict["output"] = out_string
        
        # filter chinese or japanese
        # if bool(re.compile(r'[^\x00-\x7F]+').search(reform_dict["input"])):
        #     print("Non english string: ", reform_dict["input"], '\n')
        # if bool(re.compile(r'[^\x00-\x7F]+').search(reform_dict["output"])):
        #     print("Non english string: ", reform_dict["output"], '\n')
            
        output_file.write(f"{str(reform_dict)}\n")  
        # Check if the current file has reached 1000 lines  
        if line_counter == 999:  
            output_file.close()  
            line_counter = 0  
            file_counter += 1  
            save_name = os.path.join(tsv_save_root_path, f"{basename}_{file_counter}.json")
            output_file = open(f"{save_name}", "w")  
            print(f"Writing to {save_name}")  
        else:  
            line_counter += 1  
  
    output_file.close()  

  
  
def main():  
    json_file_name = '/mnt/msranlp/zliang/data/tuning/stanford_alpaca/alpaca_data.json'
    tsv_save_root_path = '/mnt/msranlp/shaohanh/exp/unigpt_exp/tuning/unna_inst/shard_data/'
 
    process_json_file(json_file_name, tsv_save_root_path)  
        
  
if __name__ == "__main__":  
    main()  

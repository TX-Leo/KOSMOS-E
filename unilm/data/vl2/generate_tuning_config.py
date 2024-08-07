import json  
import os  
from glob import glob

def mmc4():  

    # json_files = glob(f"/mnt/msranlp/wenwan/data/mmc4-core/tsvs/*.tsv")
    json_files = [f"/mnt/msranlp/wenwan/data/mmc4-core/tsvs/docs_shard_{i}_v3.tsv" for i in range(0, 23099)]
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        source_files.append(f"../tsvs/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "mmc4-core",  
        "weight": 1.0,  
        "name": "mmc4-core"  
    }
    
    with open("/mnt/msranlp/wenwan/data/mmc4-core/config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)

def llava():  
    json_files = glob(f"/mnt/msranlp/zliang/data/tuning/LLaVA-Instruct-150K/*.tsv")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        source_files.append(f"../LLaVA-Instruct-150K/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "LLaVA-Instruct-150K",  
        "weight": 1.0,  
        "name": "llava"  
    }
    
    with open("/mnt/msranlp/zliang/data/tuning/core_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)
  
def refcoco():  
    json_files = glob(f"/mnt/msranlp/zliang/data/tuning/refcoco/*.tsv")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        source_files.append(f"../refcoco/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "refcoco,refcoco+,refcocog",  
        "weight": 1.0,  
        "name": "refcoco"  
    }
    
    with open("/mnt/msranlp/zliang/data/tuning/core_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)

def lvis():  
    json_files = glob(f"/mnt/msranlp/zliang/data/tuning/lvis/*.tsv")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        source_files.append(f"../lvis/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "lvis",  
        "weight": 1.0,  
        "name": "lvis"  
    }
    
    with open("/mnt/msranlp/zliang/data/tuning/core_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)

def flickr():  
    json_files = glob(f"/mnt/msranlp/zliang/data/tuning/flickr/*.tsv")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        source_files.append(f"../flickr/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "flickr",  
        "weight": 1.0,  
        "name": "flickr"  
    }
    
    with open("/mnt/msranlp/zliang/data/tuning/core_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)

def clevr_ref():  
    json_files = glob(f"/mnt/msranlp/zliang/data/tuning/clevr_ref/*.tsv")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        source_files.append(f"../clevr_ref/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "clevr_ref",  
        "weight": 1.0,  
        "name": "clevr_ref"  
    }
    
    with open("/mnt/msranlp/zliang/data/tuning/clevr_ref_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)
        
def merge_all():  
    json_files = []
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/LLaVA-Instruct-150K/*.tsv")
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/refcoco/*.tsv")
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/lvis/*.tsv")
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/flickr/*.tsv")
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/cocotext/*.tsv")
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/totaltext/*.tsv")
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/vcr/*.tsv")
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/clevr_ref/*.tsv")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        dirname = json_file_name.split('/')[-2]
        source_files.append(f"../{dirname}/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "llava,refcoco,refcoco+,refcocog,lvis,flickr,cocotext,totaltext,vcr,clevr_ref",  
        "weight": 1.0,  
        "name": "instruction tuning"  
    }
    
    with open("/mnt/msranlp/zliang/data/tuning/core_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)
    
    print("Writing to /mnt/msranlp/zliang/data/tuning/core_config/json/train.json")

def llava_filtered():  
    json_files = []
    json_files += glob(f"/mnt/msranlp/zliang/data/tuning/LLaVA-Instruct-150K-filtered/*.tsv")
    # json_files += glob(f"/mnt/msranlp/zliang/data/tuning/refcoco/*.tsv")
    # json_files += glob(f"/mnt/msranlp/zliang/data/tuning/lvis/*.tsv")
    # json_files += glob(f"/mnt/msranlp/zliang/data/tuning/flickr/*.tsv")
    # json_files += glob(f"/mnt/msranlp/zliang/data/tuning/cocotext/*.tsv")
    # json_files += glob(f"/mnt/msranlp/zliang/data/tuning/totaltext/*.tsv")
    # json_files += glob(f"/mnt/msranlp/zliang/data/tuning/vcr/*.tsv")
    # json_files += glob(f"/mnt/msranlp/zliang/data/tuning/clevr_ref/*.tsv")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        dirname = json_file_name.split('/')[-2]
        source_files.append(f"../{dirname}/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "llava",  
        "weight": 1.0,  
        "name": "instruction tuning"  
    }
    
    with open("/mnt/msranlp/zliang/data/tuning/llava_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)
    
    print("Writing to /mnt/msranlp/zliang/data/tuning/llava_config/json/train.json")
    
# text data
def alpaca():  
    json_files = []
    json_files += glob(f"/mnt/msranlp/shaohanh/exp/unigpt_exp/tuning/unna_inst/shard_data/alpaca_data*.json")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        dirname = json_file_name.split('/')[-2]
        source_files.append(f"../{dirname}/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "alpaca",  
        "weight": 1.0,  
        "name": "instruction tuning text data"  
    }
    
    with open("/mnt/msranlp/shaohanh/exp/unigpt_exp/tuning/unna_inst/alpaca_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)
    
    print("Writing to /mnt/msranlp/shaohanh/exp/unigpt_exp/tuning/unna_inst/alpaca_config/json/train.json")
    
def merge_text():  
    json_files = []
    json_files += glob(f"/mnt/msranlp/shaohanh/exp/unigpt_exp/tuning/unna_inst/shard_data/alpaca_data*.json")
    json_files += glob(f"/mnt/msranlp/shaohanh/exp/unigpt_exp/tuning/unna_inst/shard_data/core_*.json")
    
    source_files = []
    for json_file_name in json_files:  
        basename = os.path.basename(json_file_name)
        dirname = json_file_name.split('/')[-2]
        source_files.append(f"../{dirname}/{basename}")
                
    file_list = {  
        "source": source_files,  
        "source_lang": "alpaca, core",  
        "weight": 1.0,  
        "name": "instruction tuning text data"  
    }
    
    with open("/mnt/msranlp/shaohanh/exp/unigpt_exp/tuning/unna_inst/alpaca_core_config/json/train.json", "w") as file_list_file:  
        json.dump([file_list], file_list_file, indent=4)
    
    print("Writing to /mnt/msranlp/shaohanh/exp/unigpt_exp/tuning/unna_inst/alpaca_core_config/json/train.json")
    
if __name__ == "__main__":  
    # mmc4()  
    # llava()
    # refcoco()
    # lvis()
    # clevr_ref()
    # merge_all()
    llava_filtered()
    
    # alpaca()
    # merge_text()
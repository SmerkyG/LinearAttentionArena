import datasets

dataset = datasets.load_dataset("JeanKaddour/minipile", split="validation")        
dataset.to_json("minipile_validation.jsonl") 

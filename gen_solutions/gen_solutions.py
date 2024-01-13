import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq, get_scheduler
import torch
from peft import LoraConfig, PeftConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
from prompt_template import Prompter

dataset = load_dataset("longhoang06/Vi-GSM8K", split = "train")
gen_dataset = dataset.select(range(7500ss)) 

model_path = "mistralai/Mistral-7B-v0.1"
peft_path = "checkpoint/generator"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto", torch_dtype = torch.bfloat16)
model = PeftModel.from_pretrained(model, peft_path)
model = model.merge_and_unload()

def generator_and_save(dataset):
    def get_answer(text):
        return text.split("Đáp án:")[-1].strip()
    
    prompter = Prompter()
    progress_bar = tqdm(range(len(dataset)))
    results = []
    
    for d in dataset:
        question = d["question"]
        answer = d["answer"]
        prompt = prompter.generate_prompt(instruction = question)
        
        input_ids = tokenizer(prompt, return_tensors = "pt").input_ids
        with torch.no_grad():
            outputs = model.generate(
                input_ids = input_ids.to("cuda"),
                max_new_tokens = 512,
                temperature = 0.7,
                top_k = 50,
                top_p = 1,
                num_return_sequences = 50,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = tokenizer.eos_token_id,
                do_sample = True,
            ) 
        outputs.to("cpu")
        texts = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        texts = [prompter.get_response(text) for text in texts]
        batch = [
            {"question": question,
             "answer": answer,
             "candidate": candidate,
             "label": True if get_answer(candidate) == get_answer(answer) else False
            } for candidate in texts
        ]
        results.extend(batch)
        del batch
        del input_ids
        progress_bar.update(1)
    
    new_dataset = Dataset.from_dict(
        {"question": [d["question"] for d in results],
         "answer": [d["answer"] for d in results],
         "candidate": [d["candidate"] for d in results],
         "label": [d["label"] for d in results],
        }
    )
    
    new_dataset.save_to_disk("solutions_genereted")

if __name__ == "__main__":
    generator_and_save(gen_dataset)
                
        


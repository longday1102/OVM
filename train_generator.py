from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq, get_scheduler
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from torch.distributed import  destroy_process_group, init_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import SequentialSampler
import os
from prompt_template import Prompter

backend = "nccl"
init_process_group(backend = backend)
local_rank = int(os.environ["LOCAL_RANK"])

dataset = load_dataset("longhoang06/Vi-GSM8K", split = "train")
train_size = 7500
train_dataset = dataset.select(range(train_size))

model_path = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

prompter = Prompter() 
max_length = 640
train_dataset = train_dataset.filter(lambda x: len(tokenizer(prompter.generate_prompt(instruction = x["question"], response = x["answer"])).input_ids) + 1 <= max_length)

def tokenize_fn(prompt: str):
    result = tokenizer(
        prompt,
        truncation = True,
        max_length = max_length,
        padding = False,
        return_tensors = None)
    
    if (   
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_length
    ):    
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
        
    result["labels"] = result["input_ids"].copy()
    return result

def get_items(dataset):
    full_prompt = prompter.generate_prompt(
        dataset["question"],
        dataset["answer"],
    )
        
    tokenized_full_prompt = tokenize_fn(full_prompt)
    return tokenized_full_prompt

train_dataset = train_dataset.map(get_items, num_proc = 16)
train_dataset = train_dataset.remove_columns(column_names = ["question", "answer"])

train_dataloader = DataLoader(
    train_dataset,
    batch_size = 1,
    sampler = DistributedSampler(train_dataset),
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer = tokenizer,
        padding = True,
        return_tensors = "pt",
    ),
    pin_memory = True,
)                    

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = False,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
)
        
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config = bnb_config,
    device_map = {"": torch.device(f"cuda:{local_rank}")},
    torch_dtype = torch.bfloat16,
)

r = 64
lora_alpha = 16
lora_dropout = 0.1

lora_config = LoraConfig(
    r = r,
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = [
         "q_proj" , "k_proj" , "v_proj", "o_proj", "gate_proj" , "up_proj" ,"down_proj", "lm_head",
    ]
)

model = get_peft_model(model, lora_config)

model = model.to(f"cuda:{local_rank}")
model = DDP(model, device_ids = [local_rank])

def train():
    
    epochs = 3
    lr = 2e-4
    max_norm_value = 0.3
    num_update_steps_per_epoch = len(train_dataloader)
    num_steps = num_update_steps_per_epoch * epochs
    warmup_ratio = 0.03
    num_warmup_steps = int(warmup_ratio * num_steps)
    optimizer = AdamW(model.parameters(), lr = lr, weight_decay = 0.001)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer = optimizer,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_steps,
    )

    def is_master_process():
        ddp_rank = int(os.environ['RANK'])
        return ddp_rank == 0
        
    logging_steps = 100
    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)
        total_loss = 0
        cur_steps = 0
        model.train()
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(local_rank) for k, v in batch.items()}        
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm_value)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            cur_steps += 1

            if cur_steps % logging_steps == 0 and is_master_process():
                print(f"Epoch: {epoch + 1} -- cur_steps: {cur_steps} -- train_loss: {total_loss/cur_steps} -- lr: {optimizer.param_groups[0]['lr']}")
                
    if is_master_process():
        print("SAVING......................................................................")
        model.module.save_pretrained(f"checkpoint/generator")
        print(f"----------------------------- END OF TRAINING -----------------------------")

if __name__ == "__main__":
    train()
    destroy_process_group()
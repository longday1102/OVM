import torch
import transformers
from datasets import load_dataset, load_from_disk
from prompt_template import Prompter

class VerifierDataset:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer = None,
        data_path: str = None,
        max_length: int = None,
        load_data_method: str = None,
        mapping: bool = False,
    ):
        self.tokenizer = tokenizer
        self.prompter = Prompter()
        self.max_length = max_length
        
        if load_data_method == "hf_hub":
            dataset = load_dataset(data_path)
        elif load_data_method == "local":
            dataset = load_from_disk(data_path)
        else:
            raise NotImplementedError        
        self.dataset = dataset["train"].filter(
            lambda x: len(self.tokenizer(self.prompter.generate_prompt(instruction = x["question"], response = x["answer"])).input_ids) + 1 <= self.max_length
        )           
        if mapping:
            self.dataset = self.dataset.map(self.get_items, num_proc = 16)
            self.dataset = self.dataset.remove_columns(["question", "answer", "candidate", "label"])
            
    def left_padding(
        self,
        input_ids: list,
        attention_mask: list,
        labels: list,
        v_labels: list,
        padding_value: int = -100,
    ):
        pad_length = self.max_length - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id]*pad_length + input_ids
        attention_mask = [0]*pad_length + attention_mask
        labels = [padding_value]*pad_length + labels
        v_labels = [padding_value]*pad_length + v_labels
        
        return input_ids, attention_mask, labels, v_labels
               
    def get_items(self, dataset, IGNORE_INDEX : int = -100):
        prompt = self.prompter.generate_prompt(instruction = dataset["question"], response = dataset["candidate"])
        question = self.prompter.generate_prompt(instruction = dataset["question"])
        len_question = len(self.tokenizer(question).input_ids)
        label = dataset["label"]
        
        result = self.tokenizer(
            prompt,
            truncation = True,
            max_length = self.max_length,
            padding = False,
            return_tensors = None,
        )
        
        if (   
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
        ):    
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["attention_mask"][:len_question] = [0]*len_question
        
        result["labels"] = result["input_ids"].copy()
        result["labels"] = [token if mask != 0 else IGNORE_INDEX for token, mask in zip(result["labels"], result["attention_mask"])]
        
        v_labels = [int(label)] * len(result["input_ids"])
        result["v_labels"] = [token if mask != 0 else IGNORE_INDEX for token, mask in zip(v_labels, result["attention_mask"])]
        
        result["input_ids"], result["attention_mask"], result["labels"], result["v_labels"] = self.left_padding(
            result["input_ids"], result["attention_mask"], result["labels"], result["v_labels"], padding_value = IGNORE_INDEX)
        
        return result
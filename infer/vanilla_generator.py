from prompt_template import Prompter
from datasets import load_dataset
from dataclasses import dataclass
from tqdm.auto import tqdm
import json
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

class VanillaSampling:
    def __init__(
        self,
        generator: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.generator = generator
        self.tokenizer = tokenizer
        self.prompter = Prompter()
        
    def generate(
        self,
        dataset,
        generation_config: GenerationConfig,
        output_dir: str,
    ):
        results = []
        progress_bar = tqdm(range(len(dataset)))
        self.generator.eval()
        for data in dataset:
            input_prompt = self.prompter.generate_prompt(
                instruction = data["question"]
            )
            input_ids = self.tokenizer(input_prompt, return_tensors = "pt").input_ids
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids = input_ids.to("cuda"),
                    generation_config = generation_config,
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
            response = self.prompter.get_response(text)
            batch = {
                "question": data["question"],
                "answer": data["answer"],
                "candidate": response,
            }
            results.append(batch)
            progress_bar.update(1)
            torch.cuda.empty_cache()
        
        with open(output_dir, "w") as f:
            json.dump({"data": results}, f, indent = 4, ensure_ascii = False)
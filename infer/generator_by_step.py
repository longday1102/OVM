from prompt_template import Prompter
from transformers import PreTrainedTokenizer, GenerationConfig
import torch
import json
from tqdm.auto import tqdm

class ValueGuideBeamSearch:
    def __init__(
        self,
        verifier,
        tokenizer: PreTrainedTokenizer,
    ):
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.prompter = Prompter()
    
    def _compute_batch_vscore(
        self,
        input_ids: torch.LongTensor,
        mask_len: int,
        batch_size: int,
    ):
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, :mask_len] = 0
        input_ids = input_ids.split(batch_size, dim = 0)
        attention_mask = attention_mask.split(batch_size, dim = 0)
        batches = [
            {
                "input_ids": input_ids_,
                "attention_mask": attention_mask_,
            } for input_ids_, attention_mask_ in zip(input_ids, attention_mask)
        ]
        v_scores = []
        for batch in batches:
            with torch.no_grad():
                batch = {k:v.to("cuda") for k, v in batch.items()}
                outputs = self.verifier(
                    input_ids = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                )
                v_scores.append(outputs.v_scores.squeeze(-1).to("cpu"))
        
        assert all(
            tensor.shape[1] == v_scores[0].shape[1] for tensor in v_scores
        )   
        v_scores = torch.cat(v_scores, dim = 0) 
        input_ids = torch.cat(input_ids, dim = 0)
        assert input_ids.shape == v_scores.shape
        find_n_eos_tokens = input_ids.eq(self.tokenizer.eos_token_id).sum(dim = 1)
        
        if (find_n_eos_tokens == 0).all():
            return v_scores[:,-1] 
        else:
            indices = torch.nonzero(find_n_eos_tokens, as_tuple = False).view(-1)
        
            selected_vscores = []
            for batch_id in range(v_scores.shape[0]):
                if batch_id in indices:
                    # Find the position where the eos token id appears first
                    idx = (input_ids[batch_id] == self.tokenizer.eos_token_id).nonzero(as_tuple = False).view(-1)[0]
                    selected_vscores.append(v_scores[batch_id, idx - 1])
                else:
                    selected_vscores.append(v_scores[batch_id, -1])
            
            selected_vscores = torch.stack(selected_vscores, dim = 0)
            return selected_vscores
            
    def _prepare_for_generate(
        self,
        input_text: str,
        num_beams: int,
        batch_size_per_sequence: int
    ):
        items = self.tokenizer(input_text, return_tensors = "pt")
        input_ids = items.input_ids
        assert input_ids.ndim == 2
        
        input_ids = input_ids.repeat_interleave(num_beams, dim = 0)
        batches = input_ids.split(batch_size_per_sequence, dim = 0)
        return batches
    
    def _padding_to_concat(
        self,
        tensor: torch.LongTensor,
        max_length: int,
        padding_value: int,
    ):
        cur_size = tensor.size()
        padding_size = max_length - cur_size[1]
        padding_tensor = torch.full((cur_size[0], padding_size), padding_value)
        return torch.cat((tensor, padding_tensor), dim = 1)
        
    def generate(
        self,
        dataset,
        output_dir: str,
        generation_config: GenerationConfig,
        num_beams: int = 20,
        top_k_per_step: int = 1,
        max_steps: int = 10,
        max_new_tokens_per_step: int = 32,
        batch_size_per_sequence: int = 2,
        **kwargs        
    ):
        results = []
        progress_bar = tqdm(range(len(dataset)))
        n_sampling_per_step = int(num_beams / top_k_per_step)
        max_new_tokens = generation_config.max_new_tokens
        assert num_beams % top_k_per_step == 0
        
        self.verifier.eval()
        for data in dataset:
            input_prompt = self.prompter.generate_prompt(instruction = data["question"])
            batches = self._prepare_for_generate(
                input_text = input_prompt,
                num_beams = num_beams,
                batch_size_per_sequence = batch_size_per_sequence,
            )
            mask_len = len(self.tokenizer(input_prompt).input_ids)
            cur_step = 0
            cur_tokens_length = 0
            result = None
            while cur_step <= max_steps and cur_tokens_length <= max_new_tokens:
                sequences = []
                for input_ids in batches:
                    with torch.no_grad():
                        outputs = self.verifier.backbone.generate(
                            input_ids = input_ids.to("cuda"),
                            max_new_tokens = max_new_tokens_per_step,
                            generation_config = generation_config,
                        )
                        sequences.append(outputs.to("cpu"))
                    
                if all(
                    tensor.shape[1] == sequences[0].shape[1] for tensor in sequences
                ):
                    sequences = torch.cat(sequences, dim = 0)
                else:
                    max_seqlen = max([tensor.shape[1] for tensor in sequences])
                    sequences = torch.cat(
                        [self._padding_to_concat(tensor, max_seqlen, self.tokenizer.eos_token_id)
                         for tensor in sequences], dim = 0
                    )
                v_scores = self._compute_batch_vscore(
                    input_ids = sequences,
                    mask_len = mask_len,
                    batch_size = batch_size_per_sequence,
                )              
                _, indices = torch.topk(
                    v_scores, k = top_k_per_step, largest = True,
                )
                selected_sequences = sequences.index_select(0, indices)
                selected_scores = v_scores.index_select(0, indices)
                
                assert len(selected_sequences) == top_k_per_step
                
                if (selected_sequences == self.tokenizer.eos_token_id).any(dim = 1).all():
                    _, final_idx = torch.topk(
                    selected_scores, k = 1, largest = True,
                    )
                    selected_sequence = selected_sequences[final_idx.item()]     # [length,]
                    eos_indices = (selected_sequence == self.tokenizer.eos_token_id).nonzero(as_tuple = False).view(-1)[0]
                    final_sequence = selected_sequence[:eos_indices]
                    
                    response = self.tokenizer.decode(final_sequence, skip_special_tokens = True)
                    candidate = self.prompter.get_response(response)
                    result = {
                        "question": data["question"],
                        "answer": data["answer"],
                        "candidate": candidate,
                    }
                    break

                else:
                    input_ids = selected_sequences.repeat_interleave(n_sampling_per_step, dim = 0)
                    batches = input_ids.split(batch_size_per_sequence, dim = 0)
                    assert input_ids.shape[0] == num_beams
                    cur_step += 1
                    cur_tokens_length += max_new_tokens_per_step
            
            if result is None:
                result = {
                    "question": data["question"],
                    "answer": data["answer"],
                    "candidate": "",
                }
            results.append(result)    
            progress_bar.update(1)
        
        with open(output_dir, "w") as f:
            json.dump({"data": results}, f, indent = 4, ensure_ascii = False)

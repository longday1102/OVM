from eval_sampling.generator_by_step import ValueGuideBeamSearch
from eval_sampling.vanilla_generator import VanillaSampling
from transformers import PreTrainedModel, GenerationConfig, PreTrainedTokenizer
import torch

class SimpleSampling(VanillaSampling):
    def __init__(
        self,
        generator: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        super(SimpleSampling, self).__init__()
    
    def generate(
        self,
        input_text: str,
        generation_config: GenerationConfig,
    ):
        input_prompt = self.prompter.generate_prompt(
                instruction = input_text
            )
        input_ids = self.tokenizer(input_prompt, return_tensors = "pt").input_ids
        with torch.no_grad():
            outputs = self.generator.generate(
                input_ids = input_ids.to("cuda"),
                generation_config = generation_config,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
        response = self.prompter.get_response(text)
        torch.cuda.empty_cache()
        return response
    
class OVMSampling(ValueGuideBeamSearch):    
    def __init__(
        self,
        verifier,
        tokenizer: PreTrainedTokenizer,
        **kwargs
    ):
        super(OVMSampling, self).__init__(verifier, tokenizer, **kwargs)
        
    def generate(
        self,
        input_text: str,
        generation_config: GenerationConfig,
        num_beams: int = 20,
        top_k_per_step: int = 1,
        max_steps: int = 10,
        max_new_tokens_per_step: int = 32,
        batch_size_per_sequence: int = 2,
        **kwargs
    ):
        self.verifier.eval()
        input_prompt = self.prompter.generate_prompt(instruction = input_text)
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
                result = self.prompter.get_response(response)
                break
            else:
                input_ids = selected_sequences.repeat_interleave(n_sampling_per_step, dim = 0)
                batches = input_ids.split(batch_size_per_sequence, dim = 0)
                assert input_ids.shape[0] == num_beams
                cur_step += 1
                cur_tokens_length += max_new_tokens_per_step
                
        torch.cuda.empty_cache()
        if result is not None:
            return result
        else:
            return "Tôi không thể giải được bài toán này."

class GetResponse:
    def __init__(
        self,
        generator: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        verifier_weigth_path: str = None,
        **kwargs,
    ):
        generator.eval()
        self.simple_sampling = SimpleSampling(
            generator = generator,
            tokenizer = tokenizer,
        )
        self.verifier = VerifierModel(
            backbone = generator,
            checkpoint_dir = verifier_weight_path,
        )
        self.verifier.eval()
        self.ovm_sampling = OVMSampling(
            verifier = self.verifier,
            tokenizer = tokenizer,
        )
        ovm_generation_cf = {
            "ovm_num_beams": 20,
            "top_k_per_step": 4,
            "max_steps": 10,
            "max_new_tokens_per_step": 100,
            "batch_size_per_sequence": 16,
        }
        self.ovm_generation_cf = kwargs.get("ovm_generation_cf", ovm_generation_cf)   
        
    def generate_respone(
        self,
        input_text: str,
        ovm_mode: bool = False
    ):
        generation_config = GenerationConfig(
                max_new_tokens = 512,
                do_sample = True if ovm_mode else False,
                top_k = 50,
                top_p = 1.0,
                temperature = 0.7,
                length_penalty = 1.0,
                repetition_penalty = 1.0,
        )
        if ovm_mode:
            response = self.ovm_sampling.generate(
                input_text = input_text,
                generation_config = self.generation_config,
                **self.ovm_generation_cf,
            )
        else:
            response = self.simple_sampling.generate(
                input_text = input_text,
                generation_config = self.generation_config,
            )
        return response

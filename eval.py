from infer.vanilla_generator import VanillaSampling
from infer.generator_by_step import ValueGuideBeamSearch
from build_verifier import VerifierModel, load_generator_and_tokenizer
from transformers import GenerationConfig
import torch
from datasets import load_dataset
from argparse import ArgumentParser

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ovm_mode", required=True, default=False)
    parser.add_argument("--verifier_weight_path", default=None, type=str)
    parser.add_argument("--generator_path", default="mistralai/Mistral-7B-v0.1", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--load_k_bit", required=True, default=False)
    parser.add_argument("--max_new_tokens", required=True, type=int)
    parser.add_argument("--do_sample", required=True, default=False)
    parser.add_argument("--top_k", default=None, type=int)
    parser.add_argument("--top_p", default=None, type=float)
    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--length_penalty", default=None, type=float)
    parser.add_argument("--repetition_penalty", default=None, type=float)
    parser.add_argument("--ovm_num_beams", required=False, type=int)
    parser.add_argument("--top_k_per_step", required=False, type=int)
    parser.add_argument("--max_steps", required=False, type=int)
    parser.add_argument("--max_new_tokens_per_step", required=False, type=int)
    parser.add_argument("--batch_size_per_sequence", required=False, type=int)
    args = parser.parse_args()
    print(vars(args))
    
    generator, tokenizer = load_generator_and_tokenizer(
            generator_path = args.generator_path,
            load_k_bit = args.load_k_bit,
            local_rank = None,
    )
    
    generation_config = GenerationConfig(
        max_new_tokens = args.max_new_tokens,
        do_sample = args.do_sample,
        top_k = args.top_k,
        top_p = args.top_p,
        temperature = args.temperature,
        length_penalty = args.length_penalty,
        num_beams = 1,
        repetition_penalty = args.repetition_penalty,
        eos_token_id = tokenizer.eos_token_id,
    )
    dataset = load_dataset("longhoang06/Vi-GSM8K", split = "train")
    test_dataset = dataset.select(range(7500, len(dataset)))
    
    if args.ovm_mode:
        verifier = VerifierModel(
            backbone = generator,
            checkpoint_dir = args.verifier_weight_path,
        )
        infer = ValueGuideBeamSearch(
            verifier = verifier,
            tokenizer = tokenizer
        )
        infer.generate(
            dataset = test_dataset,
            output_dir = args.output_dir,
            generation_config = generation_config,
            num_beams = args.ovm_num_beams,
            top_k_per_step = args.top_k_per_step,
            max_steps = args.max_steps,
            max_new_tokens_per_step = args.max_new_tokens_per_step,
            batch_size_per_sequence = args.batch_size_per_sequence,
        )
        
    else:
        infer = VanillaSampling(
            generator = generator,
            tokneizer = tokenizer,
        )
        infer.generate(
            dataset = test_dataset,
            generation_config = generation_config,
            output_idr = args.output_dir,
        )

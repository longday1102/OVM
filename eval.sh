#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

python eval.py \
--ovm_mode True \
--verifier_weight_path "checkpoint/verifier/verifier.pt" \
--generator_path "longhoang06/OVM-generator" \
--output_dir "eval_results.json" \
--max_new_tokens 512 \
--do_sample True \
--top_k 50 \
--top_p 1.0 \
--temperature 0.7 \
--length_penalty 1.0 \
--repetition_penalty 1.0 \
--ovm_num_beams 20 \
--top_k_per_step 4 \
--max_steps 10 \
--max_new_tokens_per_step 100 \
--batch_size_per_sequence 16 

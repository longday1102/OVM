from transformers.generation.utils import ModelOutput
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import json

@dataclass
class VerifierModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    v_scores: torch.FloatTensor = None
    all_losses: Optional[Dict[str, torch.FloatTensor]] = None

class VerifierModel(nn.Module):
    def __init__(self, backbone, checkpoint_dir: str = None):
        super(VerifierModel, self).__init__()
        self.backbone = backbone    # GENRATOR
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        device = self.backbone.device
        dtype = self.backbone.dtype
        
        self.dropout = nn.Dropout(p = 0.2)
        self.vscore_head = nn.Linear(
            self.backbone.get_input_embeddings().embedding_dim, 1, bias = False, device = device, dtype = dtype
        )
        
        if checkpoint_dir:    # For inference
            vrf_params = torch.load(checkpoint_dir)
            self.gain = vrf_params["gain"].to(device)
            self.bias = vrf_params["bias"].to(device)
            self.vscore_head.load_state_dict(vrf_params["vscore_head"])
            
            torch.cuda.empty_cache()
        
        else:
            self.init_head_params()
            self.gain = nn.Parameter(
                torch.randn((1,), device = device, dtype = dtype)
            )
            self.bias = nn.Parameter(
                torch.randn((1,), device = device, dtype = dtype)
            )
            
        print("All parameters of vscore_head aren't frozen: ", 
              all(param.requires_grad == True for param in self.vscore_head.parameters())
             )
        print("All parameters of backbone are frozen: ",
              all(param.requires_grad == False for param in self.backbone.parameters())
             )
        self.pad_token_id = backbone.config.pad_token_id
    
    def init_head_params(self):
        output_embeddings = self.backbone.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings.mean(dim = 0, keepdim = True)

        self.vscore_head.weight = nn.Parameter(output_embeddings_avg)

    def loss_fct(self, v_scores: torch.FloatTensor, v_labels: torch.LongTensor):
        return self.mse_loss_with_mask(v_scores.squeeze(), v_labels.type_as(v_scores))
    
    def transform(self, last_hidden_states):
        return self.gain * last_hidden_states + self.bias
    
    def forward(self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        v_labels: Optional[torch.LongTensor] = None,
        output_all_losses: Optional[bool] = None,
    ):
        outputs = self.backbone(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            labels = labels, 
            use_cache = False,
            output_hidden_states = True, 
            return_dict = True,
        )
        llm_logits = outputs.logits
        llm_loss = outputs.loss
        llm_hidden_states = outputs.hidden_states

        v_hidden_states = self.transform(llm_hidden_states[-1])
        v_scores = self.vscore_head(self.dropout(v_hidden_states))

        v_loss, loss = None, None
        if v_labels is not None:
            v_loss = self.loss_fct(v_scores, v_labels)
            loss = v_loss + (llm_loss if labels is not None else 0)

        all_losses = None
        if output_all_losses:
            all_losses = {'llm_loss': llm_loss, 'v_loss': v_loss}

        return VerifierModelOutput(
            loss = loss,
            v_scores = v_scores,
            all_losses = all_losses,
        )

    def mse_loss_with_mask(self, scores: torch.FloatTensor, labels: torch.FloatTensor, IGNORE_INDEX: int = -100):
        scores = torch.where(labels.ne(IGNORE_INDEX), scores, 0)
        labels = torch.where(labels.ne(IGNORE_INDEX), labels, 0)
        return F.mse_loss(scores, labels, reduction = 'sum') / scores.shape[0]
    
def save_verifier(verifier, output_dir: str = None):
    # Saving when training with DDP  
    torch.save(
        {
            "gain": verifier.module.gain,
            "bias": verifier.module.bias,
            "vscore_head": verifier.module.vscore_head.state_dict(),
        },
        output_dir
    )
    torch.cuda.empty_cache()
        
def load_generator_and_tokenizer(generator_path: str, load_k_bit: bool = False, local_rank: int = None): 
    if load_k_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = False,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
        )
    else:
        bnb_config = None
      
    config = PeftConfig.from_pretrained(generator_path)
    base_model = config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        device_map = {"": torch.device(f"cuda:{local_rank}")} if local_rank is not None else "auto",
        torch_dtype = torch.bfloat16,
    )
    
    generator = PeftModel.from_pretrained(model, generator_path)
    torch.cuda.empty_cache()
    
    return generator, tokenizer
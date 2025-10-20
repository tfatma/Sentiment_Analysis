import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import Dict, Optional


class LoRASentimentModel(nn.Module):

    def __init__(self, model_name="roberta-base", num_labels=3, lora_config=None):

        super().__init__()
        self.num_labels= num_labels
        self.model_name= model_name

        if lora_config is None:
            lora_config= {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["query", "value", "key", "dense"],
                "bias": "none",
                "task_type": TaskType.SEQ_CLS
            }
        
        self.base_model= AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            return_dict=True
        )

        self.peft_model= LoraConfig(**lora_config)

        self.model= get_peft_model(self.base_model, self.peft_config)

        print(f"LoRA Model created with {self.get_trainable_parameters()} trainable parameters"
              f"out of {self.get_total_parameters()} total parameters "
              f"({100 * self.get_trainable_parameters()/ self.get_total_parameters():.2f}%)")
        
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs= self.model(
            input_ids= input_ids,
            attention_mask= attention_mask,
            labels= labels,
            return_dict= True
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": getattr(outputs, "hidden_states", None)
        }
    
    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
    
    def load_adapter(self, adapter_path):
        self.model= PeftModel.from_pretrained(self.base_model, adapter_path)

    def merge_and_unload(self):
        return self.model.merge_and_unload()
    

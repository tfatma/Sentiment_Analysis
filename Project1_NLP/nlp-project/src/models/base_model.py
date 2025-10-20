import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from typing import Optional, Dict

class BaseSentimentModel(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=3, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.num_labels= num_labels
        self.model_name= model_name
        self.config= AutoConfig.from_pretrained(model_name)
        self.config.num_labels= num_labels
        self.encoder= AutoModel.from_pretrained(model_name, config=self.config)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        hidden_size= self.config.hidden_size
        self.dropout= nn.Dropout(dropout)
        self.classifier=nn.Linear(hidden_size, num_labels)

        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        
    def forward(self, input_ids, attention_mask, labels, **kwargs):
        outputs=self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        pooled_output= outputs.last_hidden_state[:,0]
        pooled_output= self.dropout(pooled_output)

        logits= self.classifier(pooled_output)

        loss= None
        if labels is not None:
            loss_fn= nn.CrossEntropyLoss()
            loss= loss_fn(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.last_hidden_state,
            "pooled_output": pooled_output
        }
    
    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        return sum(p.numel() for p in self.parameters())

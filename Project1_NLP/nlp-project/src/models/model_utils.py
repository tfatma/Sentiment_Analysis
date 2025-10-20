import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any, Optional
import json
import os

class ModelUtils:

    @staticmethod
    def count_parameters(model):
        total_params= sum(p.numel() for p in  model.parameters())
        trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
            "trainable_percent": 100 * trainable_params/ total_params
        }
    
    @staticmethod
    def freeze_layers(model, layers_to_freeze):
        for name, param  in model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_freeze):
                param.requires_grad=False

    @staticmethod
    def unfreeze_layers(model, layers_to_unfreeze):
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_unfreeze):
                param.requires_grad=True

    @staticmethod
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size+= param.nelement() * param.element_size()

        buffer_size =0
        for buffer in model.buffers():
            buffer_size+= buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2

        return size_mb
    
    @staticmethod
    def save_model_info(model, save_path, additional_info):
        param_info =  ModelUtils.count_parameters(model)
        model_size =  ModelUtils.get_model_size(model)

        info = {
            "parameters" : param_info,
            "size_mb": model_size,
            "model_type": type(model).__name__
        }

        if additional_info:
            info.update(additional_info)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(info, f, indent=2)

    
    @staticmethod
    def compare_models(models):
        comparison = {}
        for name,  model in models.items():
            comparison[name]= {
                **ModelUtils.count_parameters(model),
                "size_mb" : ModelUtils.get_model_size(model)
            } 
        return comparison

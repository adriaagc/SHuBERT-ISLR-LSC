import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from fairseq.models import register_model
# Importamos la configuración y clase base del archivo shubert.py
from .shubert import SHubertConfig, SHubertModel

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

logger = logging.getLogger(__name__)


@dataclass
class SHubertLoraConfig(SHubertConfig):
    _name: str = field(default="shubert_lora")
    
    # Parámetros específicos de LoRA
    lora_r: int = field(
        default=1, 
        metadata={"help": "Rank of the LoRA matrices"}
    )
    lora_alpha: int = field(
        default=1, 
        metadata={"help": "LoRA's scaling factor"}
    )
    lora_dropout: float = field(
        default=0.1, 
        metadata={"help": "Dropout applied to the layers of LoRA"}
    )
    lora_target_modules: str = field(
        default="k_proj,v_proj,q_proj,out_proj,fc1,fc2,post_extract_proj,face_proj,left_hand_proj,right_hand_proj,body_posture_proj",
        metadata={"help": "Comma separated list of the layers of the SHuBERT model for LoRA to be applied"}
    )
    lora_modules_to_save: str = field(
        default="head", 
        metadata={"help": "Comma separated list of the layers of the SHuBERT model for LoRA to not be applied"}
    )

@register_model("shubert_lora", dataclass=SHubertLoraConfig)
class SHubertLoraModel(SHubertModel):
    def __init__(self, cfg: SHubertLoraConfig):
        super().__init__(cfg)
        self.cfg = cfg

        if not HAS_PEFT:
            raise ImportError(
                "'peft' library from Hugging Face needed to use SHubertLoraModel. "
                "Install with: pip install peft"
            )

        base_model = SHubertModel(cfg)
        
        target_modules = [m.strip() for m in cfg.lora_target_modules.split(",") if m.strip()]
        save_modules = [m.strip() for m in cfg.lora_modules_to_save.split(",") if m.strip()]
        
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=target_modules,
            modules_to_save=save_modules,
        )

        logger.info(
            f"Applying LoRA to base SHuBERT model with rank={cfg.lora_r}, "
            f"alpha={cfg.lora_alpha}, lora_dropout={cfg.lora_dropout},target_modules={target_modules}, save_models={save_modules}"
        )

        # Wrap the whole model with PEFT
        self.model = get_peft_model(base_model, lora_config)

    
    def forward(self, *args, **kwargs):
        # Delegar el paso hacia adelante al envoltorio PeftModel
        return self.model(*args, **kwargs)


    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """
        Adapta el state_dict original cargado desde el checkpoint para que coincida 
        con la estructura de llaves interna generada por PEFT (self.model).
        """
        # Delegar la carga de pesos directamente al contenedor PEFT interno (self.model)
        # Esto redirige la inyección de parámetros sin exigir el prefijo 'model.' superior.
        return self.model.load_state_dict(state_dict, strict=strict)


    @classmethod
    def build_model(cls, cfg: SHubertLoraConfig, task=None):
        """Build a new model instance."""
        return cls(cfg)
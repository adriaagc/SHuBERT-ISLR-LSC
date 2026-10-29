# fairseq/optim/adam_groups.py
# 
# Adam w/ diff. LR for LoRA and head, no need to touch trainer.py nor lr_schedulers.
# 
# How it works:
#   - At train.py -> main() each trainable param is labeled w/ `p.param_group = "lora" | "head"`.
#   - Aquí los parámetros se agrupan por esa etiqueta y cada grupo guarda un `lr_scale`.
#   - `set_lr()` (called by lr_schedulers) applies lr * lr_scale to each group, so that
#     both follow the same curve (warmup / decay) but scaled.
#
# At .yaml:
#   optimizer:
#     _name: adam_groups
#     lora_lr_mult: 0.1

import logging
from dataclasses import dataclass, field

from fairseq.logging import metrics
from fairseq.optim import register_optimizer
from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig

logger = logging.getLogger(__name__)


@dataclass
class FairseqAdamGroupsConfig(FairseqAdamConfig):
    lora_lr_mult: float = field(
        default=0.1,
        metadata={
            "help": "multiplier over the LR from the scheduler for the LoRA params "
            "(head uses LR w/o scaling)"
        },
    )


@register_optimizer("adam_groups", dataclass=FairseqAdamGroupsConfig)
class FairseqAdamGroups(FairseqAdam):
    def __init__(self, cfg, params):
        # Fairseq delivers `params` as a flat tensors list: regroup by label.
        # Params w/o label (e.g. criterion) go to base group (head).
        buckets = {} # dict w/ 2 lists of params (lora and head) --> {"lora": [p1,p2,...], "head": [p1,p2,...]}
        for p in params:
            buckets.setdefault(getattr(p, "param_group", "head"), []).append(p) # head is default

        # if cfg.lora_lr_mult != 1.0 and "lora" not in buckets:
            # raise ValueError(
            #     "adam_groups: no param has param_group='lora'. Check they are "
            #     "labeled at main() before the optimizer is built. With fp16, "
            #     "set common.fp16_no_flatten_grads=true (otherwise, Fairseq flattens all the "
            #     "params in a single tensor and groups cannot be formed)."
            # )

        if "lora" not in buckets:
            logger.warning(
                "adam_groups: no 'lora' params were detected. "
                "Default FairseqAdam is initialized."
            )
            super().__init__(cfg, params)
            return

        scales = {"lora": cfg.lora_lr_mult}
        groups = [
            {"params": ps, "name": name, "lr_scale": scales.get(name, 1.0)}
            # 'head' goes first because get_lr() from Fairseq reads the 1st group
            for name, ps in sorted(buckets.items(), key=lambda kv: kv[0] != "head")
        ] # list of {"params":ps, "name":"head" or "lora", lr_scale:0.1 or 1.0}, where "head" goes 1st

        # FairseqAdam but with defined groups
        super().__init__(cfg, groups)  # the parent class will assign to the self.para_groups attribute the list of dictionaries (groups)
        self.set_lr(self.optimizer_config["lr"])

        # print a summary to the .out:
        for g in self.param_groups:
            n = sum(p.numel() for p in g["params"])
            logger.info(
                f"adam_groups | {g['name']:<5} | {len(g['params']):>4} tensors | "
                f"{n:>12,} params | lr_scale={g['lr_scale']:g}"
            )

    def set_lr(self, lr):
        # called by lr_schedulers at each step; otherwise the same scaling factor would go to every LR
        for g in self.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0) # scale the lr of each group of params
        
            # Show both lr:
            group_name = g.get("name", "default")
            metrics.log_scalar(f"lr_{group_name}", g["lr"], priority=300, round=8)
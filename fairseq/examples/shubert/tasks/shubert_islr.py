import os
import logging
from dataclasses import dataclass

import torch

from fairseq.tasks import register_task
from fairseq.tasks.audio_pretraining import AudioPretrainingTask, AudioPretrainingConfig
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset

logger = logging.getLogger(__name__)


class ShubertISLRDataset(FileAudioDataset):
    """The pretraining 4-stream dataset + ONE gloss id per clip.
    Reuses FileAudioDataset's manifest parsing & .npy stream loading;
    replaces the per-frame k-means targets with a per-clip gloss label,
    and pads (instead of cropping) so no frames of a sign are lost."""

    def __init__(self, manifest_path, gloss_label_path, **kwargs):
        kwargs["kmeans_label_paths"] = {}            # no cluster targets for ISLR
        super().__init__(manifest_path, **kwargs)    # fills self.fnames, self.sizes, self.skipped_indices

        with open(gloss_label_path) as f:
            gloss = [int(x) for x in f if x.strip()]
        if self.skipped_indices:                     # keep alignment if any short clip was dropped
            gloss = [g for i, g in enumerate(gloss) if i not in self.skipped_indices]
        assert len(gloss) == len(self.fnames), f"{len(gloss)} labels != {len(self.fnames)} clips"
        self.gloss_labels = gloss

    def __getitem__(self, index):
        item = super().__getitem__(index)            # {"id","source":{4 streams},"kmeans_labels":{}}
        item["gloss"] = self.gloss_labels[index]
        return item

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]       # filters the null samples
        if not samples:
            return {}

        parts = ["face", "left_hand", "right_hand", "body_posture"]
        lengths = [len(s["source"]["face"]) for s in samples]       # number of frames of every sample
        Tmax, B = max(lengths), len(samples)        # max. number of frames and batch size.

        source = {}
        for p in parts:                              # pad each stream to (B, Tmax, D)
            D = samples[0]["source"][p].shape[1]     # D is the number of features ()
            buf = samples[0]["source"][p].new_zeros(B, Tmax, D)
            for i, s in enumerate(samples):
                buf[i, : lengths[i]] = s["source"][p]
            source[p] = buf

        padding_mask = torch.ones(B, Tmax, dtype=torch.bool)   # True = padding
        for i, t in enumerate(lengths):
            padding_mask[i, :t] = False     # sets to false the real frames. 

        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {                           # → model(**net_input)
                "source": source,
                "padding_mask": padding_mask,
                "gloss_labels": torch.LongTensor([s["gloss"] for s in samples]),
            },
        }


@dataclass
class ShubertISLRConfig(AudioPretrainingConfig):
    pass   # reuses `data` (dir with {split}.tsv + {split}.gloss), max_sample_size, normalize, sample_rate, …


@register_task("shubert_islr", dataclass=ShubertISLRConfig)
class ShubertISLRTask(AudioPretrainingTask):
    def load_dataset(self, split, task_cfg=None, **kwargs):
        task_cfg = task_cfg or self.cfg
        d = self.cfg.data
        self.datasets[split] = ShubertISLRDataset(
            manifest_path=os.path.join(d, f"{split}.tsv"),
            gloss_label_path=os.path.join(d, f"{split}.gloss"),
            sample_rate=task_cfg.sample_rate,
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=False,
            normalize=task_cfg.normalize,
            num_buckets=self.cfg.num_batch_buckets,
        )
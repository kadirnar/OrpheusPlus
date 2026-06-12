"""Finetune the pretrained Qwen3-0.6B + Mimi(32-cb) TTS model on a single voice.

Continues from a pretrained checkpoint (NOT from scratch) on the
ElevenLabs-EN single-speaker set encoded by encode_finetune.py. Pure TTS:
no text-QA mixing, no ratio schedule - just the encoded `input_ids`.

Launch with:  accelerate launch --config_file configs/accelerate_config.yaml finetune.py

Config hardcoded - no CLI args.
"""
import os

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType)
import wandb
from accelerate import Accelerator
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# --------------------------------------------------------------------------
# Config (hardcoded on purpose - no CLI args)
# --------------------------------------------------------------------------
PRETRAINED_CKPT = "/home/microway/kadirnar/github/OrpheusPlus/checkpoints/checkpoint-171622"
FT_DATASET = "/scratch/kadirnar/elevenlabs-en-ft-mimi32/data"
OUTPUT_DIR = "/home/microway/kadirnar/github/OrpheusPlus/checkpoints_ft_elevenlabs_en"
PROJECT_NAME = "qwen3-ja-en-tts_finetune"
RUN_NAME = "ft-elevenlabs-en-iP95p4"

EPOCHS = 10                 # 2000 clips is tiny; more passes for voice adaptation
BATCH_SIZE = 8              # per device
LEARNING_RATE = 1.0e-5     # low: adapt, don't wreck pretrained knowledge
WARMUP_RATIO = 0.03
SAVE_STEPS = 100
PAD_TOKEN = 151643         # Qwen3 <|endoftext|> (collator pad value)


def data_collator(features):
    input_ids = [f["input_ids"] for f in features]

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1] * len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i, dtype=torch.long) for i in input_ids],
        batch_first=True, padding_value=PAD_TOKEN)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(m, dtype=torch.long) for m in attention_mask],
        batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(l, dtype=torch.long) for l in labels],
        batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels}


class FSDPTrainer(Trainer):
    """Trainer with FSDP-aware full-state-dict checkpoint saving."""

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=cpu_state_dict)


wandb.init(project=PROJECT_NAME, name=RUN_NAME)

accelerator = Accelerator()
device = accelerator.device

# Load FROM the pretrained checkpoint. Its config already carries the resized
# vocab (Qwen3 + Mimi tokens), so NO add_tokens / resize_token_embeddings here.
model = AutoLigerKernelForCausalLM.from_pretrained(
    PRETRAINED_CKPT,
    attn_implementation="kernels-community/flash-attn2",
    torch_dtype=torch.bfloat16,
)
if accelerator.is_local_main_process:
    print(f"Loaded pretrained checkpoint on {device}: {PRETRAINED_CKPT}")
model = model.to(device)

train_dataset = load_dataset(
    "parquet", data_files=os.path.join(FT_DATASET, "*.parquet"), split="train")
train_dataset = train_dataset.shuffle(seed=42)
if accelerator.is_local_main_process:
    print(f"Finetune rows: {len(train_dataset)}")

training_args = TrainingArguments(
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=1,
    bf16=True,
    output_dir=OUTPUT_DIR,
    report_to="wandb",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    remove_unused_columns=True,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    average_tokens_across_devices=False,
)

trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

if accelerator.is_local_main_process:
    print(f"Finetuning {EPOCHS} epochs, lr={LEARNING_RATE}, bs={BATCH_SIZE}/device")

# Finetune (continues from PRETRAINED_CKPT weights; not a resume of its optimizer).
trainer.train()
trainer.save_model(OUTPUT_DIR)

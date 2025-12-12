import torch
import json
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field

from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image


def is_main_process():
    """분산 학습 환경에서 main process인지 확인"""
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            print(f"{name} --> {param.requires_grad}")

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}%"
        )


@dataclass
class DotsOCRConfig:
    model_path: str = "./weights/DotsOCR"
    output_dir: str = "./outputs/dotsocr-ft-lora"
    use_lora: bool = True

    lora_config: dict = field(default_factory=lambda: {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "task_type": TaskType.CAUSAL_LM,
    })

    training_config: dict = field(default_factory=lambda: {
        "do_train": True,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "dataloader_num_workers": 8,
        
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "adam_epsilon": 1e-6,
        "max_grad_norm": 1.0,
        
        "bf16": True,
        "fp16": False,

        "deepspeed": {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "train_micro_batch_size_per_gpu": "auto",
            "train_batch_size": "auto",
            "gradient_accumulation_steps": "auto",
            # # LoRA와의 호환성을 위한 설정
            # "zero_allow_untested_optimizer": True,
            # "zero_force_ds_cpu_optimizer": False,
        },
        
        "num_train_epochs": 3,
        "eval_strategy": "steps",
        "eval_steps": 400,
        "save_strategy": "steps",
        "save_steps": 400,
        "save_total_limit": 5,
        "logging_strategy": "steps",
        "logging_steps": 20,
        
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "gradient_checkpointing": True,
        "report_to": None,
    })


class PubTabNetDataset(Dataset):
    """PubTabNetDataset"""
    data_path = "./datasets/pubtabnet/raw/pubtabnet"

    def __init__(self, processor=None, is_train: bool = True):
        self.processor = processor
        self.is_train = is_train

        self.data = self._load_data()
        print(f"PubTabNet | {len(self.data)}")

    def _load_data(self) -> List[Dict]:
        data = []
        data_path = Path(self.data_path)

        if self.is_train:
            image_dir = data_path / "train"
            annotation_path = data_path / "PubTabNet_2.0.0_revised_train.jsonl"
            split_name = "train"
        else:
            image_dir = data_path / "val"
            annotation_path = data_path / "PubTabNet_2.0.0_revised_val.jsonl"
            split_name = "val"

        error_count = 0
        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())

                filename = item["filename"]
                image_path = image_dir / filename

                if not image_path.exists():
                    error_count += 1
                    continue

                html_string = item["html_string"]

                data_item = {
                    "filename": filename,
                    "image_path": str(image_path),
                    "html_string": html_string,
                    "split": split_name,
                }
                data.append(data_item)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        filename = item["filename"]
        image_path = item["image_path"]
        html_string = item["html_string"]

        image = fetch_image(image_path)
        image = get_image_by_fitz_doc(image_path, target_dpi=200)
        image = fetch_image(image, min_pixels=None, max_pixels=None)

        prompt = (
            """Extract and describe the layout and content of this document image."""
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": html_string},
        ]
        user_messages = [msg for msg in messages if msg["role"] == "user"]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        encoded_inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        encoded_prompt_inputs = self.processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        prompt_length = len(encoded_prompt_inputs["input_ids"][0])

        return {
            "input_ids": encoded_inputs["input_ids"][0],
            "attention_mask": encoded_inputs["attention_mask"][0],
            "prompt_length": prompt_length,
            "pixel_values": (
                encoded_inputs["pixel_values"].detach()
                if "pixel_values" in encoded_inputs
                else None
            ),
            "image_grid_thw": (
                encoded_inputs["image_grid_thw"].detach()
                if "image_grid_thw" in encoded_inputs
                else None
            ),
        }


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        prompt_length = [item["prompt_length"] for item in batch]
        pixel_values = [item["pixel_values"] for item in batch]
        image_grid_thw = [item["image_grid_thw"] for item in batch]

        features = [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(input_ids, attention_mask)
        ]

        processed = super().__call__(features)

        labels = processed["input_ids"].clone()
        for idx, length in enumerate(prompt_length):
            labels[idx, :length] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        processed["labels"] = labels

        # 이미지 텐서 배치 처리
        if pixel_values:
            try:
                # 이미지 텐서들을 연결하고 gradient 계산이 필요하지 않음을 명시
                concatenated_pixel_values = torch.cat(pixel_values, dim=0)
                # 이미지 데이터는 gradient가 필요하지 않으므로 detach
                processed["pixel_values"] = concatenated_pixel_values.detach()
            except RuntimeError:
                # 크기가 다른 경우 리스트로 유지하고 각각 detach
                processed["pixel_values"] = [
                    pv.detach() if isinstance(pv, torch.Tensor) else pv
                    for pv in pixel_values
                ]

        if image_grid_thw:
            try:
                # 이미지 그리드 정보도 gradient가 필요하지 않음
                concatenated_grid_thw = torch.cat(image_grid_thw, dim=0)
                processed["image_grid_thw"] = concatenated_grid_thw.detach()
            except RuntimeError:
                # 크기가 다른 경우 리스트로 유지하고 각각 detach
                processed["image_grid_thw"] = [
                    thw.detach() if isinstance(thw, torch.Tensor) else thw
                    for thw in image_grid_thw
                ]

        return processed


def main():
    # # 분산 학습 환경 초기화
    # if "LOCAL_RANK" in os.environ:
    #     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    config = DotsOCRConfig()
    if is_main_process():
        print("=" * 50)
        print("DotsOCR Training Configuration")
        print("=" * 50)
        print(f"Model Path: {config.model_path}")
        print(f"Output Directory: {config.output_dir}")
        print(f"Use LoRA: {config.use_lora}")
        print()
        
        print("LoRA Configuration:")
        for key, value in config.lora_config.items():
            print(f"  {key}: {value}")
        print()
        
        print("Training Configuration:")
        for key, value in config.training_config.items():
            if key == "deepspeed":
                print(f"  {key}:")
                for ds_key, ds_value in value.items():
                    if isinstance(ds_value, dict):
                        print(f"    {ds_key}:")
                        for nested_key, nested_value in ds_value.items():
                            print(f"      {nested_key}: {nested_value}")
                    else:
                        print(f"    {ds_key}: {ds_value}")
            else:
                print(f"  {key}: {value}")
        print("=" * 50)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Processor
    processor = AutoProcessor.from_pretrained(
        config.model_path, trust_remote_code=True, use_fast=True
    )

    if config.use_lora:
        # LoRA 설정
        lora_config = LoraConfig(**config.lora_config)
        model = get_peft_model(model, lora_config)

    # 모델을 학습 모드로 설정
    model.train()

    if is_main_process():
        print(model)
        print_trainable_parameters(model)

    # Gradient 계산을 위한 필수 설정들
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Dataset
    train_dataset = PubTabNetDataset(processor=processor, is_train=True)
    eval_dataset = PubTabNetDataset(processor=processor, is_train=False)

    # Data Collator
    data_collator = CustomDataCollator(tokenizer=processor.tokenizer)

    # Trainer
    training_args = TrainingArguments(config.output_dir, **config.training_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
    )
    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(config.output_dir + "/trainer")
    model.save_pretrained(config.output_dir + "/model")
    if is_main_process():
        print(f"Training completed. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()

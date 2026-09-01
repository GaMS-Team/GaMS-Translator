from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
import torch

from argparse import ArgumentParser
import os


def use_lora(rank=128):
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=rank,  # Rank of the LoRA adapter (e.g. 128, 256, 512)
        lora_alpha=2 * rank,  # Scaling factor for LoRA updates (e.g. rank or 2x rank)
        lora_dropout=0.1,  # Optional dropout probability for LoRA layers
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Specify the target modules for LoRA
        bias="none",
        task_type="CAUSAL_LM"
    )

    return lora_config


def run_training(experiment_dir, model_input_path, tokenizer_path, run_name, lora_rank, warmup_steps, learning_rate,
                 min_lr, dpo_beta, resume_from_checkpoint=None):
    # Load the datasets from JSONL files
    train_dataset = load_dataset("json", data_files=f"/data/training.jsonl")["train"]
    val_dataset = load_dataset("json", data_files=f"/data/validation.jsonl")["train"]
    # Maximum length of the whole sequence (prompt + completion)
    MAX_LENGTH = 8192

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Per-device batch size, adjust based on your hardware
    micro_batch_size = 1
    # Global batch size
    batch_size = 8

    world_size = int(os.environ["WORLD_SIZE"])
    gradient_accumulation_steps = batch_size // (world_size * micro_batch_size)

    assert gradient_accumulation_steps >= 1, (
        f"Global batch size {batch_size} is smaller than world_size * micro_batch_size "
        f"({world_size} * {micro_batch_size}). Increase batch_size or use fewer GPUs."
    )

    # Get process rank from the environment variable
    process_rank = int(os.environ.get("RANK", 0))

    # Print only on rank 0
    if process_rank == 0:
        print("Train data size:", len(train_dataset))
        print("Val data size:", len(val_dataset))
        print("Micro batch size:", micro_batch_size)
        print("Effective batch size:", batch_size)
        print("Gradient accumulation steps:", gradient_accumulation_steps)
        print("World size:", world_size)
        print("LoRA rank:", lora_rank)
        print("Warmup steps:", warmup_steps)
        print("Learning rate:", learning_rate)
        print("Min learning rate:", min_lr)
        print("DPO beta:", dpo_beta)

    num_epochs = 3
    steps_per_epoch = len(train_dataset) // batch_size
    eval_steps = int(1 / 4 * steps_per_epoch)  # Evaluate 4 times per epoch
    save_steps = int(1 / 4 * steps_per_epoch)  # Save 4 times per epoch

    if process_rank == 0:
        print("--------------------------------")
        print("Training parameters:")
        print("--------------------------------")
        print(f"Run name: '{run_name}'")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Evaluate each {eval_steps} steps ({eval_steps / steps_per_epoch:.2f} epochs)")
        print(f"Save each {save_steps} steps ({save_steps / steps_per_epoch:.2f} epochs)")
        print(f"Warmup steps: {warmup_steps} steps ({warmup_steps / steps_per_epoch:.2f} epochs)")
        print("--------------------------------")

    if process_rank == 0:
        print("Initializing training args")
    # DPO configuration (this is before loading the model because it is required by deepspeed)
    training_args = DPOConfig(
        output_dir=experiment_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=8,
        max_length=MAX_LENGTH,
        max_prompt_length=None,
        max_completion_length=None,
        beta=dpo_beta,
        data_seed=5,

        eval_on_start=True,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=5,
        push_to_hub=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=False,
        greater_is_better=False,

        logging_strategy="steps",
        logging_dir=f"{experiment_dir}/logs",
        logging_first_step=True,
        logging_steps=10,
        report_to="wandb",
        run_name=run_name,

        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-5,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": min_lr},
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        deepspeed="/script/deepspeed_config.json"
    )
    if process_rank == 0:
        print("Training args initialized")

    if process_rank == 0:
        print("Initializing model")
    model = AutoModelForCausalLM.from_pretrained(
        model_input_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    if process_rank == 0:
        print("Model initialized")

    lora_config = use_lora(rank=lora_rank)

    if process_rank == 0:
        print("LoRA config initialized")

    # Define Trainer.
    # ref_model=None together with peft_config makes DPOTrainer use the base model
    # with the adapters disabled as the implicit reference model, so no second copy
    # of the weights is kept in memory.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config
    )

    if process_rank == 0:
        print("Trainer initialized")
        print("Train dataset:", trainer.train_dataset)
        print("First example:", trainer.train_dataset[0])

    # Resume logic
    if resume_from_checkpoint:
        if process_rank == 0:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        if process_rank == 0:
            print("Starting training from scratch")
        trainer.train()

    if process_rank == 0:
        print("Training completed")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the dir where logs and checkopints will be stored."
    )
    parser.add_argument(
        "--model_input_path",
        type=str,
        required=True,
        help="Name of the input model."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the dir with tokenizer."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="Rank of the LoRA updates"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO KL beta"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.experiment_dir, args.model_input_path, args.tokenizer_path, args.run_name, args.lora_rank,
                 args.warmup_steps, args.learning_rate, args.min_lr, args.beta, args.resume_from_checkpoint)

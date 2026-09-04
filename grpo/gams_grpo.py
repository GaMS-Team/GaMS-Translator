from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
import torch

from argparse import ArgumentParser
import os

from reward_functions import PROMPT_TEMPLATE, bleu_score, comet_score, language_score, length_score


def has_expected_instruction(example):
    """Checks that the example's prompt starts with PROMPT_TEMPLATE.

    extract_src in reward_functions.py recovers the English source by stripping
    exactly that instruction. An example carrying a different instruction would
    keep it inside its source text and skew the COMET and length rewards, so such
    examples are dropped instead.
    """
    prompt = example["prompt"]

    return bool(prompt) and prompt[0]["content"].startswith(PROMPT_TEMPLATE)


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
                 min_lr, grpo_beta, vllm_url, resume_from_checkpoint=None):
    # Load the datasets from the parquet shards of cjvt/GaMS-Translator-GRPO-Training.
    # Globs are used so the dataset can be re-sharded without touching this script.
    dataset = load_dataset(
        "parquet",
        data_files={
            "training": "/data/training-*.parquet",
            "validation": "/data/validation-*.parquet",
        }
    )

    # Keep only the examples whose prompt uses PROMPT_TEMPLATE
    raw_sizes = {split: len(split_dataset) for split, split_dataset in dataset.items()}
    dataset = dataset.filter(has_expected_instruction)

    train_dataset = dataset["training"]
    val_dataset = dataset["validation"]
    MAX_PROMPT_LENGTH = 4096
    MAX_COMPLETION_LENGTH = 4096

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Per-device batch size, adjust based on your hardware
    micro_batch_size = 1
    # Global batch size, in prompts
    batch_size = 64
    # Number of prompts sent to vLLM per generation call
    generation_batch_examples = 16
    # Number of completions sampled per prompt
    num_generations = 4

    world_size = int(os.environ["WORLD_SIZE"])
    gradient_accumulation_steps = batch_size * num_generations // (world_size * micro_batch_size)
    eval_batch_size = max(1, num_generations // (world_size * micro_batch_size))

    assert gradient_accumulation_steps >= 1, (
        f"Global batch size {batch_size} x {num_generations} generations is smaller than "
        f"world_size * micro_batch_size ({world_size} * {micro_batch_size}). "
        f"Increase batch_size or use fewer GPUs."
    )

    # Get process rank from the environment variable
    process_rank = int(os.environ.get("RANK", 0))

    # Print only on rank 0
    if process_rank == 0:
        print("Train data size:", len(train_dataset),
              f"(dropped {raw_sizes['training'] - len(train_dataset)} with unexpected instruction)")
        print("Val data size:", len(val_dataset),
              f"(dropped {raw_sizes['validation'] - len(val_dataset)} with unexpected instruction)")
        print("Micro batch size:", micro_batch_size)
        print("Effective batch size:", batch_size)
        print("Gradient accumulation steps:", gradient_accumulation_steps)
        print("World size:", world_size)
        print("Generations per prompt:", num_generations)
        print("LoRA rank:", lora_rank)
        print("Warmup steps:", warmup_steps)
        print("Learning rate:", learning_rate)
        print("Min learning rate:", min_lr)
        print("GRPO beta:", grpo_beta)
        print("vLLM server:", vllm_url)

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
    # GRPO configuration (this is before loading the model because it is required by deepspeed)
    training_args = GRPOConfig(
        # Data preprocessing parameters
        remove_unused_columns=False,
        num_generations=num_generations,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        shuffle_dataset=True,
        data_seed=5,

        # Generation parameters
        generation_batch_size=generation_batch_examples * num_generations,
        temperature=0.1,
        generation_kwargs={
            "stop_token_ids": [1, 106]
        },

        # vLLM parameters
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=vllm_url,

        # Reward parameters
        mask_truncated_completions=False,
        reward_weights=[1.0, 0.2, 0.2, 0.1],
        beta=grpo_beta,

        output_dir=experiment_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=8,

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

    # Define Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        reward_funcs=[
            comet_score,
            language_score,
            length_score,
            bleu_score,
        ]
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
        default=128,
        help="Rank of the LoRA updates"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
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
        default=5e-7,
        help="Minimum learning rate"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="GRPO KL beta"
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        required=True,
        help="Base URL of the vLLM generation server."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.experiment_dir, args.model_input_path, args.tokenizer_path, args.run_name, args.lora_rank,
                 args.warmup_steps, args.learning_rate, args.min_lr, args.beta, args.vllm_url,
                 args.resume_from_checkpoint)

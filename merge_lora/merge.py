import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--merged_model", type=str, required=True, help="Path to save the merged model")

    args = parser.parse_args()

    print(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"Loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter and base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.merged_model}...")
    model.save_pretrained(args.merged_model)
    tokenizer.save_pretrained(args.merged_model)

    print("Done!")

if __name__ == "__main__":
    main()

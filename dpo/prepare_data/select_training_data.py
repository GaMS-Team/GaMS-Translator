import json
import random
import argparse
from pathlib import Path

PROMPT_TEMPLATE = """\
Si profesionalni prevajalec iz angleščine v slovenščino. Tvoja naloga je ustvariti visoko kakovostne prevode, ki so naravni, tekoči in natančni. Izogibaj se dobesednim prevodom, ki zvenijo nenaravno. Upoštevaj kontekst, idiome in kulturne reference. Ne dodajaj nobenih dodatnih informacij ali komentarjev, samo prevod.

Prevedi podano angleško besedilo v slovenščino.\
"""


def process_file(file_path, comet_threshold, dataset_name):
    """
    Process a JSONL file, filter by COMET score, and format for DPO.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Check COMET score
            comet_score = item.get('comet_score', 0.0)
            if comet_score < comet_threshold:
                continue

            prompt = f"{PROMPT_TEMPLATE}\n\n{item['english']}"
            chosen_example = [{"role": "user", "content": prompt}, {"role": "assistant", "content": item["chosen"]}]
            rejected_example = [{"role": "user", "content": prompt}, {"role": "assistant", "content": item["rejected"]}]

            # Create DPO format
            dpo_item = {
                "id": f"{dataset_name}_{item['id']}",
                "chosen": chosen_example,
                "rejected": rejected_example,
                "comet_score": comet_score
            }
            data.append(dpo_item)

    return data


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO training data")
    parser.add_argument("--wikipedia_input", type=str, help="Path to wikipedia JSONL file")
    parser.add_argument("--ccnews_input", type=str, help="Path to CC news JSONL file")
    parser.add_argument("--output_dir", type=str, help="Output directory for train/eval splits", default=".")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--wikipedia_comet_threshold",
        type=float,
        default=0.65,
        help="Minimum COMET score for a Wikipedia example to be retained."
    )
    parser.add_argument(
        "--ccnews_comet_threshold",
        type=float,
        default=0.7,
        help="Minimum COMET score for a CC News example to be retained."
    )

    args = parser.parse_args()

    train_data = []
    eval_data = []

    if args.wikipedia_input:
        print(f"Processing Wikipedia data from {args.wikipedia_input} "
              f"(COMET threshold: {args.wikipedia_comet_threshold})...")
        wiki_data = process_file(
            args.wikipedia_input,
            comet_threshold=args.wikipedia_comet_threshold,
            dataset_name="wikipedia"
        )
        print(f"Retained {len(wiki_data)} Wikipedia examples.")
        split_idx = int(len(wiki_data) * 0.9)
        train_data.extend(wiki_data[:split_idx])
        eval_data.extend(wiki_data[split_idx:])

    if args.ccnews_input:
        print(f"Processing CC News data from {args.ccnews_input} "
              f"(COMET threshold: {args.ccnews_comet_threshold})...")
        ccnews_data = process_file(
            args.ccnews_input,
            comet_threshold=args.ccnews_comet_threshold,
            dataset_name="ccnews"
        )
        print(f"Retained {len(ccnews_data)} CC News examples.")
        split_idx = int(len(ccnews_data) * 0.9)
        train_data.extend(ccnews_data[:split_idx])
        eval_data.extend(ccnews_data[split_idx:])

    if not train_data and not eval_data:
        print("No data processed. Please provide valid input files using --wikipedia_input and/or --ccnews_input.")
        return

    # Shuffle merged data
    print("Shuffling merged data...")
    random.seed(args.seed)
    random.shuffle(train_data)
    random.shuffle(eval_data)

    print(f"Train set size: {len(train_data)}")
    print(f"Eval set size: {len(eval_data)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the splits
    train_path = output_dir / "training.jsonl"
    eval_path = output_dir / "validation.jsonl"

    print(f"Saving train data to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saving eval data to {eval_path}...")
    with open(eval_path, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("Done!")


if __name__ == "__main__":
    main()

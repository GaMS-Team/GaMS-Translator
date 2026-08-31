import argparse
import json
from tqdm import tqdm
import os

from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """\
Si profesionalni prevajalec iz angleščine v slovenščino. Tvoja naloga je ustvariti visoko kakovostne prevode, ki so naravni, tekoči in natančni. Izogibaj se dobesednim prevodom, ki zvenijo nenaravno. Upoštevaj kontekst, idiome in kulturne reference. Ne dodajaj nobenih dodatnih informacij ali komentarjev, samo prevod.

Prevedi podano angleško besedilo v slovenščino.\
"""


def read_jsonl(path):
    """Read a JSONL file and return a list of parsed JSON objects (one per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def process_batch(batch, model, sampling_params):
    prompts = [f"{PROMPT_TEMPLATE}\n\n{en_text.strip()}" for en_text in batch]
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    response = model.chat(messages, sampling_params, use_tqdm=False)
    predictions = [x.outputs[0].text for x in response]

    return predictions


def main(args):
    dataset = read_jsonl(args.input_path)

    model = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=2 * args.max_seq_len
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_seq_len)

    os.makedirs(args.output_path, exist_ok=True)

    # Process the dataset in batches
    num_examples = len(dataset)
    for batch_start in tqdm(range(0, num_examples, args.batch_size), desc="Translating batches"):
        batch_end = min(batch_start + args.batch_size, num_examples)
        batch = [example["text"] for example in dataset[batch_start:batch_end]]
        batch_predictions = process_batch(
            batch,
            model,
            sampling_params
        )

        # Write results immediately
        for j, (en_text, translation) in enumerate(zip(batch, batch_predictions)):
            idx = batch_start + j
            output_file_path = f"{args.output_path}/{idx}"
            with open(output_file_path + "_en.txt", "w") as f_out:
                f_out.write(en_text)
            with open(output_file_path + "_sl.txt", "w") as f_out:
                f_out.write(translation)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the JSONL file containing the English dataset. Each line must be a JSON object with a 'text' field."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the directory where translated dataset will be saved."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name. Either path to the local dir or HuggingFace ID."
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        required=True,
        help="Tensor parallel size of the model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Number of examples to translate at once."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help="Maximum number of input tokens accepted by the model."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

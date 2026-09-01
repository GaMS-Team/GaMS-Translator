#!/usr/bin/env python3
"""
Prepare DPO training data from English Wikipedia documents and their Slovene translations.

The translation model makes two systematic errors:
  1. Double newlines (paragraph separators) are collapsed to single newlines.
  2. The model does not stop at the end of the document and keeps generating.

For each (en, sl) pair this script produces a DPO sample:
  - "rejected": the raw Slovene translation (with formatting errors).
  - "chosen":   the Slovene translation reformatted to match the English original:
                   * single newlines → double newlines (restore paragraph separators)
                   * truncated to the same number of paragraphs as the English source.

Output: a JSONL file where each line is a JSON object with:
  english, rejected, chosen, id, has_formatting_errors, en_paragraphs, sl_raw_paragraphs
"""

import os
import re
import json
import argparse


def read_file(path: str) -> str:
    """Read a text file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_paragraphs(text):
    """
    Split text into paragraphs separated by single newlines.

    This is used for Slovene translations where the model collapsed
    double newlines into single newlines, so each line is a paragraph.
    """
    normalized = text.replace("\r\n", "\n").strip()
    paragraphs = [p.strip() for p in normalized.split("\n")]
    return paragraphs


def format_paragraphs(paragraphs):
    """Join paragraphs with single newlines."""
    return "\n".join(paragraphs) + "\n"


def insert_double_newlines(en_paras, sl_paras):
    """Insert double newlines into text."""
    sl_iterator = iter([p for p in sl_paras if p.strip() != ""])
    sl_corrected = []
    for p in en_paras:
        if p.strip() == "":
            sl_corrected.append("")
        else:
            sl_corrected.append(next(sl_iterator))

    return sl_corrected


def is_match(sl_paras, en_paras):
    for sl_p, en_p in zip(sl_paras, en_paras):
        if sl_p.strip() == "" or en_p.strip() == "":
            if not (sl_p.strip() == "" and en_p.strip() == ""):
                return False

    return True


def check_formatting(en_text, sl_text):
    """
    Analyse the Slovene translation for formatting errors.

    Returns a dict with:
      - has_double_newlines: whether the Slovene file uses \\n\\n separators
      - en_para_count: number of paragraphs in the English source
      - sl_para_count: number of paragraphs detected in the Slovene translation
      - has_extra_content: whether the Slovene has more paragraphs than English
      - is_correct: True only if formatting AND paragraph count match
    """
    en_paras = get_paragraphs(en_text)
    sl_paras = get_paragraphs(sl_text)

    en_double = len([p for p in en_paras if p.strip() == ""])
    sl_double = len([p for p in sl_paras if p.strip() == ""])

    en_para_count = len(en_paras) - en_double
    sl_para_count = len(sl_paras) - sl_double

    correct_title = sl_paras[0].startswith("# ")
    if not correct_title:
        sl_paras[0] = f"# {sl_paras[0]}"

    if sl_double >= en_double:
        return {
            "has_double_newlines": True,
            "correct_title": correct_title,
            "en_para_count": en_para_count,
            "sl_para_count": sl_para_count,
            "has_extra_content": sl_para_count > en_para_count,
            "is_truncated": sl_para_count < en_para_count,
            "is_correct": len(sl_paras) == len(en_paras) and is_match(sl_paras, en_paras),
            "en_paras": en_paras,
            "sl_paras": sl_paras,
        }

    return {
        "has_double_newlines": False,
        "correct_title": correct_title,
        "en_para_count": en_para_count,
        "sl_para_count": sl_para_count,
        "has_extra_content": sl_para_count > en_para_count,
        "is_truncated": sl_para_count < en_para_count,
        "is_correct": False,
        "en_paras": en_paras,
        "sl_paras": sl_paras,
    }


def build_chosen(en_paras, sl_paras):
    """
    Build the 'chosen' version of the Slovene translation:
      1. Keep only as many paragraphs as the English original has.
      2. Join with double newlines.
    """
    sl_corrected = insert_double_newlines(en_paras, sl_paras)
    truncated = sl_corrected[: len(en_paras)]
    return format_paragraphs(truncated)


def process_pair(en_path, sl_path):
    """
    Process a single (English, Slovene) document pair.

    Returns a DPO-ready dict, or None if the Slovene file is missing.
    """
    en_text = read_file(en_path)
    sl_text = read_file(sl_path)

    info = check_formatting(en_text, sl_text)

    if info["is_correct"]:
        return {}

    if info["is_truncated"]:
        return None

    # The rejected sample is always the raw Slovene translation
    rejected = sl_text

    # The chosen sample is the reformatted, truncated Slovene translation
    chosen = build_chosen(info["en_paras"], info["sl_paras"])

    # Extract the numeric id from the filename (e.g., "42_en.txt" -> 42)
    doc_id = int(os.path.split(en_path)[-1].replace("_en.txt", ""))

    return {
        "id": doc_id,
        "english": en_text.replace("\r\n", "\n").strip() + "\n",
        "rejected": rejected,
        "chosen": chosen,
        "has_formatting_errors": not info["is_correct"],
        "missing_double_newlines": not info["has_double_newlines"],
        "has_wrong_title": not info["correct_title"],
        "has_extra_content": info["has_extra_content"],
        "en_paragraphs": info["en_para_count"],
        "sl_raw_paragraphs": info["sl_para_count"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DPO training data from English Wikipedia and Slovene translations."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing paired *_en.txt and *_sl.txt files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output_path

    # Discover all English files
    en_files = sorted(
        [file for file in os.listdir(input_dir) if file.endswith("_en.txt")],
        key=lambda p: int(p.replace("_en.txt", ""))
    )
    print(f"Found {len(en_files)} English files in {input_dir}")

    results = []
    stats = {
        "total": 0,
        "formatting_errors": 0,
        "has_wrong_title": 0,
        "missing_double_newlines": 0,
        "extra_content": 0,
        "truncated": 0
    }

    for en_file in en_files:
        sl_file = os.path.join(input_dir, en_file.replace("_en.txt", "_sl.txt"))
        if not os.path.exists(sl_file):
            print(f"  WARNING: no Slovene counterpart for {en_file}, skipping.")
            continue

        result = process_pair(os.path.join(input_dir, en_file), sl_file)
        stats["total"] += 1
        if result is None:
            stats["truncated"] += 1
            continue
        if result == {}:
            continue

        results.append(result)
        if result["has_formatting_errors"]:
            stats["formatting_errors"] += 1
        if result["has_wrong_title"]:
            stats["has_wrong_title"] += 1
        if result["missing_double_newlines"]:
            stats["missing_double_newlines"] += 1
        if result["has_extra_content"]:
            stats["extra_content"] += 1

    # Write output
    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"DPO Data Preparation Summary")
    print(f"{'='*60}")
    print(f"  Total pairs processed:           {stats['total']}")
    print(f"  Pairs with formatting errors:    {stats['formatting_errors']}")
    print(f"    - Missing double newlines:     {stats['missing_double_newlines']}")
    print(f"    - Wrong title:                 {stats['has_wrong_title']}")
    print(f"    - Extra content (didn't stop): {stats['extra_content']}")
    print(f"    - Truncated (no pair):         {stats['truncated']}")
    print(f"  Pairs with correct formatting:   {stats['total'] - stats['formatting_errors'] - stats['truncated']}")
    print(f"{'='*60}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare DPO training data from English CCNews documents and their Slovene translations.

The translation model makes three systematic errors:
  1. Double newlines (paragraph separators) are collapsed to single newlines.
  2. The model does not stop at the end of the document and keeps generating.
  3. The model keeps the date in English format

For each (en, sl) pair this script produces a DPO sample:
  - "rejected": the raw Slovene translation (with formatting errors).
  - "chosen":   the Slovene translation reformatted to match the English original:
                   * single newlines → double newlines (restore paragraph separators)
                   * truncated to the same number of paragraphs as the English source.
                   * date is converted to Slovene format

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


def compare_header(en_paras, sl_paras):
    """
    Compare the header of the English and Slovene texts.

    Returns True if the headers match, False otherwise and the corrected Slovene header.
    """
    assert en_paras[1].startswith("*"), f"English title has more than one line {en_paras}"

    correct_format = True
    sl_correct = sl_paras.copy()
    if not sl_correct[0].startswith("#"):
        correct_format = False
        sl_correct[0] = f"# {sl_correct[0]}"

    if not sl_correct[1].startswith("*"):
        correct_format = False
        if len(sl_correct[1]) == 0:
            first_char = ""
        else:
            first_char = sl_correct[1].strip()[0]
        if first_char.isdigit():
            sl_correct[1] = convert_date(en_paras[1])
        else:
            sl_correct = [sl_correct[0], convert_date(en_paras[1]), *sl_correct[1:]]

    else:
        correct_date = convert_date(en_paras[1])
        if sl_correct[1] != correct_date:
            correct_format = False
            sl_correct[1] = correct_date

    return correct_format, sl_correct


def convert_date(en_date: str):
    en_date = en_date.replace("*", "").strip()

    # Convert YYYY-MM-DD to D. M. Y
    year, month, day = en_date.split('-')

    # Convert to integers to remove leading zeros and return in D. M. Y format
    return f"*{int(day)}. {int(month)}. {year}*"


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

    if len(sl_paras) < 2:
        return {
            "has_double_newlines": False,
            "en_para_count": len(en_paras),
            "sl_para_count": len(sl_paras),
            "has_extra_content": False,
            "has_wrong_date_format": False,
            "is_truncated": True,
            "is_correct": False,
            "en_paras": en_paras,
            "sl_paras": sl_paras
        }

    en_double = len([p for p in en_paras if p.strip() == ""])
    sl_double = len([p for p in sl_paras if p.strip() == ""])

    en_para_count = len(en_paras) - en_double
    sl_para_count = len(sl_paras) - sl_double

    correct_format, sl_correct = compare_header(en_paras, sl_paras)

    if sl_double >= en_double and correct_format:
        return {
            "has_double_newlines": True,
            "en_para_count": en_para_count,
            "sl_para_count": sl_para_count,
            "has_extra_content": sl_para_count > en_para_count,
            "has_wrong_date_format": not correct_format,
            "is_truncated": sl_para_count < en_para_count,
            "is_correct": len(sl_paras) == len(en_paras) and is_match(sl_paras, en_paras) and correct_format,
            "en_paras": en_paras,
            "sl_paras": sl_correct
        }

    return {
        "has_double_newlines": False,
        "en_para_count": en_para_count,
        "sl_para_count": sl_para_count,
        "has_extra_content": sl_para_count > en_para_count,
        "has_wrong_date_format": not correct_format,
        "is_truncated": sl_para_count < en_para_count,
        "is_correct": False,
        "en_paras": en_paras,
        "sl_paras": sl_correct
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
        "has_extra_content": info["has_extra_content"],
        "has_wrong_date_format": info["has_wrong_date_format"],
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
        "missing_double_newlines": 0,
        "extra_content": 0,
        "truncated": 0,
        "wrong_date_format": 0
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
        if result["missing_double_newlines"]:
            stats["missing_double_newlines"] += 1
        if result["has_extra_content"]:
            stats["extra_content"] += 1
        if result["has_wrong_date_format"]:
            stats["wrong_date_format"] += 1

    # Write output
    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"DPO Data Preparation Summary")
    print(f"{'=' * 60}")
    print(f"  Total pairs processed:           {stats['total']}")
    print(f"  Pairs with formatting errors:    {stats['formatting_errors']}")
    print(f"    - Missing double newlines:     {stats['missing_double_newlines']}")
    print(f"    - Extra content (didn't stop): {stats['extra_content']}")
    print(f"    - Truncated (no pair):         {stats['truncated']}")
    print(f"    - Wrong date format:           {stats['wrong_date_format']}")
    print(f"  Pairs with correct formatting:   {stats['total'] - stats['formatting_errors'] - stats['truncated']}")
    print(f"{'=' * 60}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()

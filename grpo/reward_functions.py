import os
import re
from functools import lru_cache
from typing import List

import comet
import fasttext
from comet import download_model, load_from_checkpoint
#from nltk.tokenize import word_tokenize
from sacrebleu import corpus_bleu


PROMPT_TEMPLATE = """\
Si profesionalni prevajalec iz angleščine v slovenščino. Tvoja naloga je ustvariti visoko kakovostne prevode, ki so naravni, tekoči in natančni. Izogibaj se dobesednim prevodom, ki zvenijo nenaravno. Upoštevaj kontekst, idiome in kulturne reference. Ne dodajaj nobenih dodatnih informacij ali komentarjev, samo prevod.

Prevedi podano angleško besedilo v slovenščino.\
"""


def compute_bleu(ref: List, pred: List) -> float:
    """Computes bleu score.

    Args:
        ref: reference translation
        pred: generated output by the model

    Returns:
        Bleu score.
    """

    tokenize = "13a"

    refs = [[ref[0]["content"]]]
    sys = [pred[0]["content"]]

    bleu_score = corpus_bleu(sys, refs, tokenize=tokenize).score
    return float(bleu_score) / 100.0


def extract_src(prompt: str) -> str:
    """Extracts the actual source text from a translation-style prompt.

    Args:
        prompt: Input to the model

    Returns:
        cleaned prompt
    """
    source = prompt.removeprefix(PROMPT_TEMPLATE)

    return source.strip()


@lru_cache(maxsize=1)
def get_comet_model() -> comet.models.CometModel:
    """Retrieves a comet model.

    Returns:
        Cached comet model.
    """
    model_path = "/reward_models/wmt22-cometkiwi-da/checkpoints/model.ckpt"

    model = load_from_checkpoint(model_path, reload_hparams=True, local_files_only=True)
    return model


def comet_score(completions: List, prompts: List, chosen: List, **kwargs) -> List:
    """Computes comet scores.

    Args:
        completions: the generated outputs by the model
        prompts: the prompts given to the model
        chosen: the reference translations

    Returns:
        Comet scores.
    """

    model = get_comet_model()

    samples = [
        {
            "src": extract_src(prompt[0]["content"]),
            "mt": completion[0]["content"],
        }
        for prompt, completion in zip(prompts, completions)
    ]
    model_output = model.predict(
        samples=samples,
        batch_size=8,
        gpus=1,
        num_workers=0,
        accelerator="cuda",
        devices=[int(os.environ.get("LOCAL_RANK", "0"))],
    )

    return model_output[0]


def _normalize_for_lid(s: str) -> str:
    """Normalize text for language identification or text preprocessing.

     Args:
        s: an input string

    Returns:
        A processed string
    """
    # converts non-strings to strings, removing tabs and new lines etc.
    _whitespace = re.compile(r"\s+")
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    # Remove newlines/tabs and collapse spaces
    s = _whitespace.sub(" ", s).strip()
    return s


@lru_cache(maxsize=1)
def get_fasttext_model() -> fasttext.FastText._FastText:
    """Retrieves a fasttext model.

    Returns:
        Cached fastettext model.
    """

    model = fasttext.load_model(
        "/reward_models/lid.176.bin"
    )
    return model


def bleu_score(completions: List, prompts: List, chosen: List, **kwargs) -> List:
    """Computes bleu scores.

    Args:
        completions: the generated outputs by the model
        prompts: the prompts given to the model
        chosen: the reference translations

    Returns:
        Bleu scores.
    """

    scores = []
    for i in range(len(completions)):
        score = compute_bleu(pred=completions[i], ref=chosen[i])
        scores.append(score)
    return scores


def language_score(completions: List, prompts: List, chosen: List, **kwargs) -> List:
    """Computes language scores.

    Given that the completions are longer than 3 characters, the language score for
    each entry in the completions list is the probability returned from the fasttext
    model of the lanaguge being slovene.

    Args:
        completions: the generated outputs by the model
        prompts: the prompts given to the model
        chosen: the reference translations

    Returns:
        Language scores (ranges from 0 to 1).
    """

    texts = [_normalize_for_lid(c[0]["content"]) for c in completions]
    texts_lengths = [len(t) for t in texts]
    model = get_fasttext_model()
    scores = []

    for t, l in zip(texts, texts_lengths):

        if l < 3:
            # very short strings are unreliable; treat as not Slovene
            scores.append(0.0)
        else:

            labels, probs = model.predict(t, k=176)

            target = "__label__sl"

            if target in labels:
                idx = labels.index(target)
                sl_prob = probs[idx]
            else:
                sl_prob = 0.0  # or None, depending on what you want

            scores.append(sl_prob)
    return scores


def length_score(completions: List, prompts: List, chosen: List, **kwargs) -> List:
    """Computes length-based scores using word counts.

    Compares the number of words in each generated completion against the
    number of words in its english source text and assigns a score between 0 and 1.

    The score is based on the completion/source length ratio:
    - ratio < 1.0: score = ratio
    - 1.0 <= ratio < 2.0: score = 2.0 - ratio
    - ratio >= 2.0: score = 0.0

    This gives the highest score to outputs whose word count is close to the
    source, while penalizing outputs that are much shorter or much longer.

    Args:
        completions: Generated outputs by the model.
        prompts: Prompts given to the model.
        chosen: Reference translations. Included for interface compatibility,
            but not used in this function.

    Returns:
        A list of length-based scores, each ranging from 0 to 1.
    """

    solution_str = [c[0]["content"] for c in completions]
    source_str = [extract_src(p[0]["content"]) for p in prompts]

    #solutions_len = [
    #    len(word_tokenize(text=s, language="slovene")) for s in solution_str
    #]
    #source_len = [len(word_tokenize(text=s, language="english")) for s in source_str]

    solutions_len = [len(s.split()) for s in solution_str]
    source_len = [len(s.split()) for s in source_str]

    scores = []
    for sol, source in zip(solutions_len, source_len):
        ratio = float(sol / source)
        if 1.0 < ratio < 2.0:
            score = 2.0 - ratio
        elif ratio >= 2.0:
            score = 0.0
        else:
            score = ratio
        scores.append(score)
    return scores

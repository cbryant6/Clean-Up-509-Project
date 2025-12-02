# models.py
"""
Simple melody generator based on a bigram (note-to-note) model.

Main pieces:
- load_melodies: read training melodies from a text file
- build_bigram_model: turn melodies into bigram transition counts
- generate_melody: sample a new melody from the model
- save_model / load_model: store and restore a trained model as JSON

The model format is:
    dict[current_note] -> dict[next_note] = count
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import random
from typing import Dict, List, Iterable, Optional


START_TOKEN = "^"
END_TOKEN = "$"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_melodies(path: str | Path) -> List[List[str]]:
    """
    Load melodies from a text file.

    Each line in the file represents one melody and contains
    space-separated tokens (e.g., note names or pitch IDs).

    Example line:
        C4 D4 E4 F4 G4

    Returns:
        melodies: list of list-of-tokens
    """
    path = Path(path)
    melodies: List[List[str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            melody = stripped.split()
            melodies.append(melody)

    return melodies


# ---------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------
def add_start_end_tokens(
    melody: List[str],
    start: str = START_TOKEN,
    end: str = END_TOKEN,
) -> List[str]:
    """
    Wrap a melody with start and end tokens so the model
    can learn when to begin and stop.
    """
    return [start] + melody + [end]


def build_bigram_model(
    melodies: Iterable[List[str]],
    use_start_end: bool = True,
    start: str = START_TOKEN,
    end: str = END_TOKEN,
) -> Dict[str, Dict[str, int]]:
    """
    Build a bigram model from a list of melodies.

    melodies:
        iterable of list-of-notes.

    Returns:
        bigram_counts: dict[current_note] -> dict[next_note] = count
    """
    bigram_counts: Dict[str, Counter] = defaultdict(Counter)

    for melody in melodies:
        if not melody:
            continue

        if use_start_end:
            seq = add_start_end_tokens(melody, start=start, end=end)
        else:
            seq = melody

        # count all (current, next) pairs
        for curr, nxt in zip(seq[:-1], seq[1:]):
            bigram_counts[curr][nxt] += 1

    # convert nested Counters to plain dicts for JSON-friendliness
    return {curr: dict(nexts) for curr, nexts in bigram_counts.items()}


# ---------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------
def _sample_from_counts(
    counts: Dict[str, int],
    rng: Optional[random.Random] = None,
) -> str:
    """
    Sample one key from a dict of counts, treating them as weights.
    """
    if rng is None:
        rng = random

    if not counts:
        raise ValueError("Cannot sample from empty counts dictionary.")

    notes = list(counts.keys())
    weights = list(counts.values())
    total = sum(weights)

    # simple manual weighted sampling to avoid numpy dependency
    r = rng.uniform(0, total)
    cumsum = 0.0
    for note, w in zip(notes, weights):
        cumsum += w
        if r <= cumsum:
            return note

    # Fallback (should not be reached)
    return notes[-1]


def generate_melody(
    model: Dict[str, Dict[str, int]],
    max_length: int = 32,
    start: str = START_TOKEN,
    end: str = END_TOKEN,
    forbid_triple_repeat: bool = True,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Generate one melody using the bigram model.

    Algorithm:
    - Start from the start token.
    - At each step, sample a next note from the learned counts.
    - Stop if we hit the end token or reach max_length.

    Args:
        model: dict[current] -> dict[next] = count
        max_length: maximum number of notes (excluding start/end tokens)
        start / end: start and end tokens used when training
        forbid_triple_repeat:
            if True, avoid sampling a note that would create
            three identical notes in a row (AAA).
        rng: optional random.Random instance for reproducibility

    Returns:
        melody: list of note tokens (WITHOUT start/end tokens)
    """
    if rng is None:
        rng = random

    current = start
    melody: List[str] = []

    for _ in range(max_length):
        if current not in model:
            break

        candidates = dict(model[current])
        if not candidates:
            break

        # Optionally avoid AAA patterns
        if forbid_triple_repeat and len(melody) >= 2:
            last1 = melody[-1]
            last2 = melody[-2]
            if last1 == last2 and last1 in candidates:
                # temporarily remove that note to avoid a triple
                candidates = {n: c for n, c in candidates.items() if n != last1}
                if not candidates:
                    # if nothing left, allow it anyway
                    candidates = dict(model[current])

        nxt = _sample_from_counts(candidates, rng=rng)

        if nxt == end:
            break

        melody.append(nxt)
        current = nxt

    return melody


# ---------------------------------------------------------------------
# Save / load utilities
# ---------------------------------------------------------------------
def save_model(
    model: Dict[str, Dict[str, int]],
    path: str | Path,
) -> None:
    """
    Save a bigram model as JSON to the given path.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, sort_keys=True)


def load_model(path: str | Path) -> Dict[str, Dict[str, int]]:
    """
    Load a bigram model from a JSON file created by save_model.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    # Keys are already strings, values are dict[str,int]; JSON loads ints as int
    return model


# ---------------------------------------------------------------------
# Convenience "train" function
# ---------------------------------------------------------------------
def train_and_save(
    data_path: str | Path,
    model_path: str | Path,
    use_start_end: bool = True,
) -> Dict[str, Dict[str, int]]:
    """
    Convenience helper:

    - Load melodies from data_path
    - Build bigram model
    - Save model to model_path

    Returns the trained model.
    """
    melodies = load_melodies(data_path)
    model = build_bigram_model(melodies, use_start_end=use_start_end)
    save_model(model, model_path)
    return model


if __name__ == "__main__":
    # Example manual run:
    # python models.py
    data_file = Path("data/melodies.txt")
    model_file = Path("data/bigram_model.json")

    if data_file.exists():
        print(f"Training model from {data_file} ...")
        model = train_and_save(data_file, model_file)
        print(f"Saved model to {model_file}")

        print("Example generated melody:")
        rng = random.Random(509)  # fixed seed for reproducible demo
        melody = generate_melody(model, max_length=32, rng=rng)
        print(" ".join(melody))
    else:
        print("data/melodies.txt not found; please add training data.")

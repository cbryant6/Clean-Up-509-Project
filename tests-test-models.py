# tests/test_models.py

import unittest
import random

from models import (
    add_start_end_tokens,
    build_bigram_model,
    generate_melody,
)


class TestMelodyModel(unittest.TestCase):

    def setUp(self):
        # Very small toy dataset for deterministic tests
        self.melodies = [
            ["C4", "D4", "E4"],
            ["C4", "E4", "G4"],
        ]

    def test_add_start_end_tokens(self):
        mel = ["A", "B"]
        wrapped = add_start_end_tokens(mel, start="^", end="$")
        self.assertEqual(wrapped, ["^", "A", "B", "$"])

    def test_build_bigram_model_counts(self):
        model = build_bigram_model(self.melodies, use_start_end=True,
                                   start="^", end="$")

        # Check some expected transitions and counts
        # ^ -> C4 should appear twice (two melodies)
        self.assertIn("^", model)
        self.assertIn("C4", model["^"])
        self.assertEqual(model["^"]["C4"], 2)

        # C4 -> D4 once, C4 -> E4 once
        self.assertEqual(model["C4"]["D4"], 1)
        self.assertEqual(model["C4"]["E4"], 1)

        # Check that end token exists as possible next state
        self.assertIn("$", model["E4"])

    def test_generate_melody_stops_on_end_or_maxlen(self):
        model = build_bigram_model(self.melodies, use_start_end=True)

        rng = random.Random(0)
        melody = generate_melody(
            model,
            max_length=10,
            rng=rng,
        )

        # Melody should be non-empty and at most max_length
        self.assertGreater(len(melody), 0)
        self.assertLessEqual(len(melody), 10)

        # All notes must be from the vocabulary
        vocab = {note for mel in self.melodies for note in mel}
        for note in melody:
            self.assertIn(note, vocab)

    def test_forbid_triple_repeat(self):
        # Craft a tiny model that would otherwise produce AAAA...
        model = {
            "^": {"A": 10},
            "A": {"A": 10, "$": 1},
        }

        rng = random.Random(42)
        melody = generate_melody(
            model,
            max_length=10,
            forbid_triple_repeat=True,
            rng=rng,
            start="^",
            end="$",
        )

        # Check there is no 'AAA' substring
        for i in range(len(melody) - 2):
            self.assertFalse(
                melody[i] == melody[i+1] == melody[i+2],
                msg=f"Triple repeat found at positions {i}-{i+2}",
            )


if __name__ == "__main__":
    unittest.main()

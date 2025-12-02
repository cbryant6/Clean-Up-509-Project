# TECHIN 509 – Melody Generator Clean Up

This project implements a simple music melody generator using a **bigram
(note-to-note) probabilistic model**.  
The model is trained from a text file of melodies and can then generate
new melodies by sampling from learned transition probabilities.

The repository is fully self-contained and reproducible.

---

## 📁 Project Structure


Music Generator File/
├── data/
│   ├── melodies.txt          # training data (one melody per line)
│   └── bigram_model.json     # trained model (auto-generated)
├── models.py                 # training, sampling, save/load logic
└── tests-test-models.py     # unit tests

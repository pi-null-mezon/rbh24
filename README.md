RBH24
===

This is SystemFailure solution for Russian Biometric Hackaton 2024

### Installation

```bash
python3.10 -m virtualenv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Repo structure

[./templatesgen](./templatesgen) - scripts to make photo + biometric templates pairs, read [manual of how to use here](./templatesgen/README.md)

[./train_naive_decoder](./train_naive_decoder) - scripts to train template to photo decoder (loss does not contain cosine similarity with insightface/buffalo_l templates)

[./researches](./researches) - some researches we have done along the way
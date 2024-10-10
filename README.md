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

1. [./templatesgen](./templatesgen) - scripts to make photo + biometric templates pairs

2. [./train_naive_decoder](./train_naive_decoder) - tools to train template to photo decoder

3. [./train_adapter_for_instantid](./train_adapter_for_instantid) - tools to train adapter for instantid

4. [./researches](./researches) - some researches we have done along the way
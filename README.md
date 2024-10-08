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

[./templatesgen](./templatesgen) - scripts to make photo + biometric templates pairs

[./train_naive_decoder](./train_naive_decoder) - tools to train template to photo decoder

[./train_adapter_for_instantid](./train_adapter_for_instantid) - tools to train adapter for instantid control-net (stable diffusion for face from vector reconstruction)

[./researches](./researches) - some researches we have done along the way
RBH24
===

This is SystemFailure solution for Russian Biometric Hackaton 2024

We have shown:

1. Biometric template could be reconstructed to a face photo with high cosine similarity score with original template. 
For two test sets, for our strongest reconstruction algorithm, we have measured positive pass rate to be 76.1 % and 43.4 % respectively.

| Test set | Description                                                             | Number of IDs | Samples per ID | Demographics                 | Positive pass rate FMR=1E-6 (Our decoder) | Positive pass rate FMR=1E-6 (Our adapter for InstantID) |
|----------|-------------------------------------------------------------------------|---------------|----------------|------------------------------|-------------------------------------------|---------------------------------------------------------|
| valface  | Manually collected samples of unique persons from internet. Not famous. | 1143          | 1              | White, Black and Asian faces | 34.2%                                     | 76.1%                                                   |
| glint    | First 1K ids from glint dataset                                         | 1000          | 1              | White, Black and Asian faces | 25.3%                                     | 43.3%                                                   |

2. To make reconstruction algorithm relatively small data could be stollen from biometric system. Our strongest 
reconstruction algorithm was trained on only 4K high quality photo-template pairs!

3. protection - in progress...

### Examples

| Original photo               | Our adapter for InstantID                                     | Our decoder                                      |
|------------------------------|---------------------------------------------------------------|--------------------------------------------------|
| ![](./examples/crops/ik.jpg) | ![](./examples/adapters/adapterHQ4K/ik_(cosine%200.7578).jpg) | ![](./examples/decoder/ik_(cosine%200.5390).png) |
|                              | Cosine: 0.7578                                                | Cosine: 0.5390                                   |
| ![](./examples/crops/ka.jpg) | ![](./examples/adapters/adapterHQ4K/ka_(cosine%200.8520).jpg) | ![](./examples/decoder/ka_(cosine%200.5383).png) |
|                              | Cosine: 0.8520                                                | Cosine: 0.5383                                   |
| ![](./examples/crops/kd.jpg) | ![](./examples/adapters/adapterHQ4K/kd_(cosine%200.7562).jpg) | ![](./examples/decoder/kd_(cosine%200.6774).png) |
|                              | Cosine: 0.7562                                                | Cosine: 0.6774                                   |
| ![](./examples/crops/at.jpg) | ![](./examples/adapters/adapterHQ4K/at_(cosine%200.7029).jpg) | ![](./examples/decoder/at_(cosine%200.6036).png) |
|                              | Cosine: 0.7029                                                | Cosine: 0.6036                                   |

### Installation

Prepare machine with CUDA12 compatible GPU and at least 24GB of VRAM (tested: RTX3090, RTX4090)

```bash
python3.10 -m virtualenv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Repo structure

1. [./templatesgen](./templatesgen) - scripts to make photo + biometric templates pairs

2. [./train_naive_decoder](./train_naive_decoder) - tools to train template to photo decoder (demo included)

3. [./train_adapter_for_instantid](./train_adapter_for_instantid) - tools to train adapter for instantid (demo included)

4. [./researches](./researches) - some researches we have done along the way

5. [./protection](./protection) - biometric templates protection research
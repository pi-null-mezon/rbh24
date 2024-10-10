# InstantID

Here we explore of how to train adapter for stable diffusion for face from biometric vector reconstruction.

[InstantID](https://github.com/instantX-research/InstantID) is control-net for stable diffusion image generator. It was 
trained specifically to condition image generation by face similarity score for `insightface/antelopev2`.

## Reasoning

1. there is a high quality and open source InstanceID model to generate faces from `insightface/antelopev2` feature vector
2. we have the restriction for attacked model (`insightface/buffalo_l`)
3. we can train `insightface/buffalo_l` to `insightface/antelopev2` feature mapper, and if it's quality will be high enough, we will be able to reconstruct similar face from such mapped feature vector

The visualization of the transition to a new basis in 3D:

![](./artifacts/figures/transition2new_basis.png)

## Key results

* We have used ordinary least squares (OLS) to find transformation matrix from `insightface/buffalo_l` features space to `insightface/antelopev2` features space  
* Single Linear layer is enough to make a transition from `insightface/buffalo_l` features space to `insightface/antelopev2` features space 
* 10k persons with single photo guarantees the generation of adapted vectors close to directly extracted vectors
* 1k persons with single photo is enough to generate some percent of adapted vectors close to directly extracted vectors

OSE (ordinal square error) on training data:

| N persons | N vectors | mean MSE | mean COS |
|-----------|-----------|----------|----------|
| 180k      | 8.6m      | 0.19     | 0.9      |
| 100k      | 100k      | 0.19     | 0.9      |
| 10k       | 10k       | 0.20     | 0.9      |
| 1k        | 1k        | 0.39     | 0.82     |
| 100       | 100       | 0.83     | 0.47     |


![](artifacts/figures/img.png)

## Validation of adapter_HQ_4000

This adapter was trained on only 4K of template pairs

![](./artifacts/adapter_HQ_4000_sample_0.png)      ![](./artifacts/adapter_HQ_4000_sample_1.png)      ![](./artifacts/adapter_HQ_4000_sample_2.png)
![](./artifacts/adapter_HQ_4000_sample_3.png)      ![](./artifacts/adapter_HQ_4000_sample_4.png)      ![](./artifacts/adapter_HQ_4000_sample_5.png)
![](./artifacts/adapter_HQ_4000_sample_6.png)      ![](./artifacts/adapter_HQ_4000_sample_7.png)      ![](./artifacts/adapter_HQ_4000_sample_8.png)
![](./artifacts/adapter_HQ_4000_sample_9.png)      ![](./artifacts/adapter_HQ_4000_sample_10.png)      ![](./artifacts/adapter_HQ_4000_sample_11.png)

```
STATISTICS ON 1143 TEST SAMPLES FROM 'valface':
 - COSINE MIN:    -0.0550
 - COSINE MEAN:   0.7127
 - COSINE MEDIAN: 0.7365
 - COSINE MAX:    0.8553
TOTAL: 870 of 1143 have cosine with genuine template greater than 0.661 >> it is 76.1 % of validation samples

STATISTICS ON 1000 TEST SAMPLES FROM 'glint':
 - COSINE MIN:    0.2915
 - COSINE MEAN:   0.6311
 - COSINE MEDIAN: 0.6424
 - COSINE MAX:    0.8479
TOTAL: 434 of 1000 have cosine with genuine template greater than 0.661 >> it is 43.4 % of validation samples
```

## Validation of adapter_100K

IN PROGRESS...

## How to run training

```bash
pip install jupyterlab
jupyter-lab ./buffalo2antelope_adapter.ipynb  
```

And press RUN button

## How to run validation

!Prepare machine with CUDA12 compatible GPU and at least 24GB of VRAM (tested: RTX3090, RTX4090)! 

1. Install [InstantID](https://github.com/instantX-research/InstantID) according to repo instructions
2. Copy `validate.py` and `tools.py` into root of InstantID installation folder
3. Run validation:

```bash 
python validate.py --set valface # it is database collected by SystemFailure (does not contain samples from glint nor webface)
python validate.py --set glint --max_ids 1000  # test part of glint dataset
```

## Demo

Demo allows you to reconstruct face photo from two types of input:

 - insighface/buffalo_l template saved as .pkl (numpy array with single (512,) vector of np.float) file or .b64 file (base64 encoded string)
 - photo of a face (insighface/buffalo_l will try to extract template from this photo before reconstruction)

```bash
python demo.py --input source_template.pkl --output ./output.jpg
```

So, reconstructed photo will be saved in --output value. Do not forget to copy `./artifacts` and `./models` to InstatID installation folder!

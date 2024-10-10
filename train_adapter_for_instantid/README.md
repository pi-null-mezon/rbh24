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

## Results preview for adapter_HQ_4000 (yes it was trained in 4000 ids - amazing!)

![](./artifacts/adapter_HQ_4000_sample_0.png)      ![](./artifacts/adapter_HQ_4000_sample_1.png)      ![](./artifacts/adapter_HQ_4000_sample_2.png)
![](./artifacts/adapter_HQ_4000_sample_3.png)      ![](./artifacts/adapter_HQ_4000_sample_4.png)      ![](./artifacts/adapter_HQ_4000_sample_5.png)
![](./artifacts/adapter_HQ_4000_sample_6.png)      ![](./artifacts/adapter_HQ_4000_sample_7.png)      ![](./artifacts/adapter_HQ_4000_sample_8.png)
![](./artifacts/adapter_HQ_4000_sample_9.png)      ![](./artifacts/adapter_HQ_4000_sample_10.png)      ![](./artifacts/adapter_HQ_4000_sample_11.png)

## How to run validation

1. Install [InstantID](https://github.com/instantX-research/InstantID) according to repo instructions
2. Copy `validate.py` and `tools.py` into root of InstantID installation folder
3. Run validation:

```bash 
python validate.py --set valface # it is database collected by SystemFailure (does not contain samples from glint nor webface)
python validate.py --set glint --max_ids 1000  # test part of glint dataset
```
# Train decoder_large

Log of the training session:
```
 - mean: [0.485, 0.456, 0.406]
 - std: [0.229, 0.224, 0.225]
 - swap_red_blue: True
 - train_in_fp16: False
 - crop_size: (128, 128)
 - batch_size: 64
 - grad_accum_batches: 2
 - num_epochs: 10
 - lr_scheduler: Cosine
 - perception_net_name: vgg11
 - latent_space_dims: 512
 - max_lr: 0.002
 - min_lr: 1e-05
 - max_grad_norm: 10.0
 - augment: False
 - normalize_templates: True
 - model_name: buffalo_decoder_large_on_vgg11
 - max_batches_per_train_epoch: -1
 - visualize_control_samples: 9
 - visualize_each_step: 32
 - perceptual_loss_weight: 11.0
 - laplace_loss_weight: 1000.0
 - mse_loss_weight: 1.0
 - model size: 94.100 MB
 - perceptor (vgg11) size: 35.173 MB
Train dataset:
  - unique samples: 7207803
Test dataset:
  - unique samples: 569866
TRAIN:
Epoch 0 >> loss 4.47211 | 7207744/7207803 ~ 100.0 % | 311 samples / s 
 - loss:  4.59519 - improvement
TEST:
 - loss:  9.01825 - improvement
Epoch 00001: adjusting learning rate of group 0 to 1.9513e-03.
TRAIN:
Epoch 1 >> loss 4.41544 | 7207744/7207803 ~ 100.0 % | 309 samples / s 
 - loss:  4.43517 - improvement
TEST:
 - loss:  8.94888 - improvement
...
```

Test set probes:

![](./artifacts/decoder_large_2nd_epoch_sample_0.png)      ![](./artifacts/decoder_large_2nd_epoch_sample_1.png)      ![](./artifacts/decoder_large_2nd_epoch_sample_2.png)
![](./artifacts/decoder_large_2nd_epoch_sample_3.png)      ![](./artifacts/decoder_large_2nd_epoch_sample_4.png)      ![](./artifacts/decoder_large_2nd_epoch_sample_5.png)
![](./artifacts/decoder_large_2nd_epoch_sample_6.png)      ![](./artifacts/decoder_large_2nd_epoch_sample_7.png)      ![](./artifacts/decoder_large_2nd_epoch_sample_8.png)

Test set similarity check:

```
STATISTICS ON 1000 TEST SAMPLES FROM GLINT:
 - COSINE MIN:    0.0928
 - COSINE MEAN:   0.5295
 - COSINE MEDIAN: 0.5415
 - COSINE MAX:    0.7490
```
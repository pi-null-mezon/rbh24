# InstantID

Here we explore of how to train adapter for stable diffusion for face from biometric vector reconstruction

Reasoning:

1. there is a high quality and open source InstanceID model to generate faces from antelope_v2 feature vector
2. we have the restriction for attacked model (buffalo_l)
3. we can train buffalo_l to antelope_v2 feature mapper, and if it's quality will be high enough, we will be able to reconstruct similar face from such mapped feature vector

The visualization of the transition to a new basis in 3D:

![img.png](images/transition2new_basis.png)

## Key results

* Single Linear layer is enough to make a transition from buffalo_l features space to antelope_v2 features space and achieve metrics average MSE= & average_cos = COS= 
* 10k persons with single photo is enough to generate adapted vectors close to directly extracted vectors 

![img.png](images/img.png)


## Results preview

Adapter `buffalo2atelope_adapler_analytical.onnx`

cosine: 0.673
![77de0ef.jpg](artifacts%2Fik%2Fportrait%2F77de0ef.jpg)   

cosine: 0.730
![80948f8.jpg](artifacts%2Fka%2Fportrait%2F80948f8.jpg)

cosine: 0.711
![079487b.jpg](artifacts%2Fkd%2Fportrait%2F079487b.jpg)

cosine: 0.745
![71fa9fe.jpg](artifacts%2Fat%2Fportrait%2F71fa9fe.jpg)

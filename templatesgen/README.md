Before run scripts in this directory, complete following steps:

1. Download weights of insightface/buffalo_l:

https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

2. Download weights of insightface/antelopev2:

https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing

3. Unzip archives with models files into:

```
../models
      /buffalo_l
      /antelopev2
```

4. Download glint dataset:

https://paperswithcode.com/dataset/glint360k

5. After this steps you could run templates generation script:

```bash
python -m templatesgen.run --images_path "local_path_to_glint_dataset" --output_file "where_to_save_pickle_file"
```
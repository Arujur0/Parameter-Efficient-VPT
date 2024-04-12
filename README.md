# Dynamic Rs-VPT

This repository contains the code for the Parameter Efficient Prompt Tuning of Vision Transformers project.

## Dynamic Residual Visual Prompt Tuning:

This code for running the Dynamic Residual VPT (Dynamic Rs-VPT) is derived from the [Visual Prompt Tuning Repo](https://github.com/KMnP/vpt), where necessary changes were added for a modular prompt embedding to the original Vision Transformer, a residual reparameterization of the prompt projections, and a prompt paritioning block which splits the prompts into a length variable prefix and suffix.

Link to VIT B-16: [256x256_GAN_checkpoint.pt](https://drive.google.com/file/d/1qgfyvTK-pO4g3QmtEYrJwkfNq7ql6hra/view?usp=share_link)
This model is trained on 4 of the 5 Fine Graine Visual Categorization Tasks described in the baseline paper. This model was trained on average for 60-90 epochs depending on the dataset. Links to the trained model checkpoints are provided below, to run the evaluation script effectively it is recommended to download them and place them into the appropriate dataset folder in vpt/output:

Link to trained CUB-200 model: [cub.pth](https://drive.google.com/file/d/1zgdde1ud6goFQEeih64YTbdYbXRXFwB-/view?usp=sharing)

Link to trained Oxford Flowers model: [flowers.pth](https://drive.google.com/file/d/1pzaqEhIM3KDzGL_JukgPzNBrbAYZkxj6/view?usp=sharing)

Link to trained Stanford Dogs model: [dogs.pth](https://drive.google.com/file/d/1jFCZMkmZtUR_TPHi3S8SgH8vctOUIYtS/view?usp=sharing)

Link to trained NA Birds model: [birds.pth](https://drive.google.com/file/d/1YLowRhSP44vnxCj0GatWHd5te3XPtPaa/view?usp=sharing)

You may also want to run the following command to download the required packages before getting started.
```
conda env create -f environment.yml
```


## Experiments

### Key configs:

- 🔥VPT related:
  - MODEL.PROMPT_SIZE: prompt length
- Vision backbones:
  - MODEL.MODEL_ROOT: folder with pre-trained model checkpoints
- Optimization related: 
  - SOLVER.BASE_LR: learning rate for the experiment
  - SOLVER.WEIGHT_DECAY: weight decay value for the experiment
  - DATA.BATCH_SIZE: defaulted to 32
- Datasets related:
  - DATA.NAME: name of dataset, benchmarked on
  - DATA.DATAPATH: where you put the datasets
  - DATA.NUMBER_CLASSES
- Others:
  - SEED: for reproducability
  - OUTPUT_DIR: output dir of the final model and logs
  - MODEL.SAVE_CKPT: if set to `True`, will save model ckpts and final output of both val and test set

### Datasets preperation:

See Table 8 in the Appendix for dataset details. 

- Fine-Grained Visual Classification tasks (FGVC): The datasets can be downloaded following the official links. The JSON files in the Datasets/DATA.NAME folder specifies the official train, val and test splits used in the VPT paper. Those splits were followed in this work to ensure similar experimentation conditions.

  - [CUB200 2011](https://data.caltech.edu/records/65de6-vp158)

  - [NABirds](http://info.allaboutbirds.org/nabirds/)

  - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)


### Training
To run the training script ensure to update the config yaml file in the vpt\configs\linear folder respective to the dataset to be trained on. The modifications to be made are straight forward. Just change the datapath to the root directory of the dataset and comment out the TOTAL_EPOCH==0 configuration. Also if needed the hyper parameters such as the base learning rate and weight decay can be changed. Then from the root directory simply run:

```

python vpt\train.py --config-file "path\to\root\vpt\configs\linear\dataset_name.yaml"

```
Here dataset_name.yaml corresponds to one of the four and should be one of {'cub', 'dogs', 'flowers', 'nabirds'}.
### Evaluation
The same file can be run to evaluate the benchmarked models. Ensure to update the prompt size in the config.py file in vpt\src\configs.  Simply un-comment out the TOTAL_EPOCH==0 configuration (if already commented from the previous step). and run the following command:

```

python vpt\train.py --config-file "path\to\root\vpt\configs\linear\dataset_name.yaml"

```

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
<th valign="bottom">Link</th>
<th valign="bottom">md5sum</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz">link</a></td>
<td align="center"><tt>d9715d</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MoCo v3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar">link</a></td>
<td align="center"><tt>8f39ce</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MAE</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">link</a></td>
<td align="center"><tt>8cad7c</tt></td>
</tr>
<tr><td align="left">Swin-B</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth">link</a></td>
<td align="center"><tt>bf9cc1</tt></td>
</tr>
<tr><td align="left">ConvNeXt-Base</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth">link</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://pytorch.org/vision/stable/models.html">link</a></td>
<td align="center"><tt>-</tt></td>
</tr>
</tbody></table>

### Examples for training and aggregating results

See [`demo.ipynb`](https://github.com/KMnP/vpt/blob/main/demo.ipynb) for how to use this repo.

### Hyperparameters for experiments in paper

The hyperparameter values used (prompt length for VPT / reduction rate for Adapters, base learning rate, weight decay values) in Table 1-2, Fig. 3-4, Table 4-5 can be found here: [Dropbox](https://cornell.box.com/s/lv10kptgyrm8uxb6v6ctugrhao24rs2z) / [Google Drive](https://drive.google.com/drive/folders/1ldhqkXelHDXq4bG7qpKn5YEfU6sRehJH?usp=sharing). 

## Citation

If you find our work helpful in your research, please cite it as:

```
@inproceedings{jia2022vpt,
  title={Visual Prompt Tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## License

The majority of VPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) are licensed under the Attribution-NonCommercial 4.0 International license.

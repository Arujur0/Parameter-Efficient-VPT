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

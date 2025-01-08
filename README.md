# $\texttt{ModSCAN}$
[![arXiv: available](https://img.shields.io/badge/arXiv-available-red.svg)](https://arxiv.org/abs/2410.06967)
[![EMNLP'24](https://img.shields.io/badge/EMNLP'24-f1b800)](https://aclanthology.org/2024.emnlp-main.713/)

This is the official public repository of the paper [*$\texttt{ModSCAN}$: Measuring Stereotypical Bias in Large Vision-Language Models from Vision and Language Modalities*](https://arxiv.org/abs/2410.06967).  
*All the following updates will be released here first in the future.*  

*Be careful! This repository may contains potentially unsafe information. User discretion is advised.*

## How to use this repository?

### A. Install and set the ENV
1. Clone this repository.
2. Prepare the python ENV.
```
conda create -n modscan python=3.10 -y
conda activate modscan
cd PATH_TO_THE_REPOSITORY
bash prepare.sh
```

### B. Generate and (or) download our benchmark dataset.

**Dataset for the Vision Modality Task**
1. Download the official UTKFace dataset [here](https://www.kaggle.com/datasets/jangedoo/utkface-new) to directory datasets/UTKFace/.
2. Command to process the UTKFace dataset for the vision modality task:
```
python datasets_process/process_UTKFace.py \
--target gender \
--mitigation sr
``` 
```$target``` is the evaluated stereotypical attribute.  
```$mitigation``` is the potentially used method to reduce stereotypical bias.

**Dataset for the Language Modality Task**
1. Download the our self-generated (SelfGen) dataset [here](https://huggingface.co/datasets/A5hbr1ng3r/ModSCAN) to directory datasets/UTKFace/ or use model described in the original paper to generate (all details have been provided in our paper).
2. To add vision debiasing prompt into the SelfGen images, please run the command:
```
python datasets_process/process_SelfGen.py
```

### C. Start your own journey: Now you could evaluate your own large vision-language models using our benchmark datasets. Good luck!

**An Practic on LLaVA-v1.5**
1. Prepare the environment for LLaVA-1.5 model according to the steps in the [LLaVA-1.5 website](https://github.com/haotian-liu/LLaVA).
2. Command to evaluate the vision modality task on UTKFace.
```
python eval/query_UTKFace_multiple_faces_1_occupation.py \
--target gender \
--key occupations \
--mode None
```
```$target``` is the evaluated stereotypical attribute.
```$key``` is the evaluated stereotypical scenario.
```$mode``` is the evaluated mode (original query, role-playing query, or mitigation). 
3. Command to evaluate the language modality task on SelfGen.
```
python eval/query_SelfGen_1_scene_1_attribute.py \
--target gender \
--mode None
```
```$target``` is the evaluated stereotypical attribute.
```$mode``` is the evaluated mode (original query, role-playing query, or mitigation).
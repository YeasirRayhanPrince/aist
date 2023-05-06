# AIST: An Interpretable Attention-Based Deep Learning Model for Crime Prediction

This repository is the official implementation of [AIST: An Interpretable Attention-Based Deep Learning Model for Crime Prediction](https://dl.acm.org/doi/10.1145/3582274). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```


## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

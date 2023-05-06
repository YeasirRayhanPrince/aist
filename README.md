# AIST: An Interpretable Attention-Based Deep Learning Model for Crime Prediction

This repository is the official implementation of [AIST: An Interpretable Attention-Based Deep Learning Model for Crime Prediction](https://dl.acm.org/doi/10.1145/3582274). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train AIST on 2019 'theft' data of Near North Side, Chicago, run this command:

```train
python train.py
```

To further train AIST on different crime categories and communities of Chicago, run this command:
```train
python train.py --tct=chicago --tr=ID1 --tc=ID2
```
For IDs of the communities or the crime-categories, check `data/chicago/chicago_cid_to_name.txt` and `data/chicago/chicago_crime-cat_to_id.txt`
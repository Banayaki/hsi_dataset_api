# Install

```bash
pip3 install git+https://gitlab+deploy-token-608170:j7p2ZQZKgxuuEQVBHxPL@gitlab.com/rustam-industries/hsi_dataset_api.git
```

# Dataset structure
Dataset should be stored in the following structure:

<pre>
{dataset_name}
├── train
│   ├── hsi
│   │   ├── 1.npy
│   │   └── 1.yml
│   └── masks
│   │   └── 1.png
├── test
│   ├── hsi
│   │   ├── 1.npy
│   │   └── 1.yml
│   └── masks
│   │   └── 1.png
└── meta.yaml
</pre>

# Meta.yml
In this file you **should** provide classes description (it's name and label). Also, you can store any helpful information that describes the dataset. 

For example:

```yaml
name: HSI Dataset example
description: Some additional info about dataset
source: URL
images: 10
classes:
- {name: background, label: 0}
- {name: crops, label: 1}
- {name: tree, label: 2}
```

# {number}.yml
In this file you can store HSI specific information such as date, name of humidity. 

For example:

```yaml
data: 10-10-2021-10:10
name: SalinasValley
humidity: 45
temperature: 30
```

# Python API
Via API presented in this repo you can access the dataset.

Example of using the API can be found in `test/test_dataset.py` file

## Importing

```python
from hsi_dataset_api import HsiDataset
```

## Create Data Access Object
```python
dataset = HsiDataset('../example/dataset_example')
```

## Getting the dataset meta information
```python
dataset.get_dataset_description()
```

## Getting the shuffled train data using python generator
```python
for data_point in dataset.data_iterator(opened=True, train=True, shuffled=True):
    hyperspecter = data_point.hsi
    mask = data_point.mask
    meta = data_point.meta
```


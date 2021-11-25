# Install

```bash
pip install git+https://gitlab+deploy-token-608170:j7p2ZQZKgxuuEQVBHxPL@gitlab.com/rustam-industries/hsi_dataset_api.git
```

# Dataset structure
Dataset should be stored in the following structure:

## Plain structure (#1)

<pre>
{dataset_name}
├── hsi
│   ├── 1.npy
│   └── 1.yml
├── masks
│   └── 1.png
└── meta.yaml
</pre>

Or in structure like this (such structure was created while using data cropping)

## Cropped data structure (#2)

<pre>
{dataset_name}
├── hsi
│   ├── specter_1
│   │   ├── 1.npy
│   │   ├── 1.yml
│   │   ├── 2.npy
│   │   └── 2.yml
│   └── specter_2
│       ├── 1.npy
│       └── 1.yml
├── masks
│   ├── specter_1
│   │   ├── 1.png
│   │   └── 2.png
│   └── specter_2
│       └── 1.png
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
from hsi_dataset_api import HsiDataset, HsiDataCropper
```

## Cropping the data

```python
base_path = '/mnt/data/corrected_hsi_data'
output_path = '/mnt/data/cropped_hsi_data'
classes = ['potato', 'tomato']
selected_folders = ['HSI_1', 'HSI_2']  # Completely optional

cropper = HsiDataCropper(side_size=512, step=8, objects_ratio=0.20, min_class_ratio=0.01)
cropper.crop(base_path, output_path, classes, selected_folders)
```

## Plot cropped data statistics

```python
cropper.draw_statistics()
```

## Using the data

### Create Data Access Object
```python
dataset = HsiDataset('../example/dataset_example', cropped_dataset=False)
```

Parameter `cropped_dataset` controls type of the dataset structure. If the dataset persist in the memory in
the structure like **second** **(#2)** - set this parameter to `True`

### Getting the dataset meta information
```python
dataset.get_dataset_description()
```

### Getting the shuffled train data using python generator
```python
for data_point in dataset.data_iterator(opened=True, shuffle=True):
    hyperspecter = data_point.hsi
    mask = data_point.mask
    meta = data_point.meta
```

# See examples in the folder `examples`

# Ordering Sentences and Paragraphs with Pre-trained Encoder-Decoder Transformers and Pointer Ensembles

Code, models and data for the paper *Ordering Sentences and Paragraphs with Pre-trained Encoder-Decoder Transformers and Pointer Ensembles* under review at NAACL2021.

## Getting started

Create the environnement, activate and install requirements.

```bash
conda create -n ordering python=3.7
conda activate ordering
pip install -r requirements.txt
```

## Datasets

### Download the datasets

ArXiv, VIST, ROCStory and Wikipedia are stored on Google Drive.
We can download the datasets using [``gdown``](https://pypi.org/project/gdown/).
For CNN-DailyMail, there is no need to download the data (see after).

```bash
pip install gdown==3.12.2 
```

#### ArXiv

```bash
gdown https://drive.google.com/uc?id=0B-mnK8kniGAieXZtRmRzX2NSVDg
mkdir dataset/arxiv
tar -xf dataSet.tgz -C dataset/arxiv
rm dataSet.tgz
```

#### VIST

```bash
gdown https://drive.google.com/uc?id=1Arc5vnthfeg6qEHpKU_--y6MKZd5DM78
unzip vist.zip -d dataset/vist
rm vist.zip
```

#### ROCStory

```bash
gdown https://drive.google.com/uc?id=1xXuy_7XWzgiwS4tYdclKizvmg_MZ-LLX
unzip ROCStory.zip -d dataset/rocstory
rm ROCStory.zip
```

#### Wikipedia

```bash
gdown https://drive.google.com/uc?id=1B05WiMNKYKjsi1TEweexu01GHElWwIDJ
unzip best_wikipedia.zip -d dataset/best_wikipedia
rm best_wikipedia.zip
```

### Use the datasets

We use the [``datasets``](https://github.com/huggingface/datasets) library from HuggingFace to load and access the datasets.
The dataset custom loading script are in the ``dataset/`` folder.
The loading script uses the downloaded datasets except for CNN-DailyMail where the loading script download the dataset itself.

To load a dataset, run:

```python
from datasets import load_dataset

dataset = load_dataset("path/to/dataset/python/file")
```

## Models

We present 3 models in the paper.

| Model | Number of parameters | 
| --- | --- |
| BART + Simple Pointer | 140601600 |
| BART + Deep Pointer | 144145152 |
| BART + Ensemble Pointer | 140601612 | 

> ``multi`` in the code or in a filename corresponds to the Ensemble Pointer in the paper.

### Train a model

We create our models on top of the [``transformers``](https://github.com/huggingface/transformers) library from Huggingface and use the [Trainer](https://github.com/huggingface/transformers/blob/v3.4.0/src/transformers/trainer.py) to train our models.
The configuration files for the models presented in the paper are in the ``training/args/`` folder.

To retrain a model, run:

```bash
python run.py --model model --args_file path/to/json/file
```

``model`` is the the model to train (``default`` for BART + simple PtrNet, ``deep`` for BART + deep PtrNet, ``multi`` for BART + Ensemble PtrNet, or ``baseline`` for our baseline LSTM+Attention) and ``path/to/json/file`` is the path to the configuration file to use (note that the configuration file should correspond to ``model``).

To change the training parameters you can directly change the configuration file or create a new one.

### Evaluate the models

To evaluate the models on a dataset, we create configuration files in the ``evaluation/args/``.

To run the evaluation, run:

```python
from evaluation.benchmark import Benchmark
ben = Benchmark.from_json("path/to/json/file")
df = ben.run()
print(df)
>> *dataframe containing the results*
```

> results may vary a bit as the passages are randomly mixed

### Use the models

Use the ``OrderingModel`` class from ``use.py``. 
For example, to use BART + multi PtrNet trained on the Wikipedia dataset:

```python
from use import OrderingModel
from training.scripts.models.bart_multi import  BartForSequenceOrderingWithMultiPointer

model = OrderingModel(BartForSequenceOrderingWithMultiPointer, "models/bart-base-multi-best-wikipedia", "facebook/bart-base")

PASSAGES_TO_ORDER = ["p3", "p2", "p1", "p4"]

model.order(PASSAGES_TO_ORDER)
>> ["p1", "p2", "p3", "p4"]
```

### Training time

We train our 3 models on a single GPU Quadro RTX 6000 for:

- arXiv : 68 hours
- VIST : 4 hours
- ROCStory : 6 hours
- Wikipedia : 25 hours
- CNN-DailyMail : 14 hours

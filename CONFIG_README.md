# T2R2 - Config Description

In T2R2 we use configuration files in `yaml` format to specify our needs. Below we describe sections that you may include in your config. We left some fields undescribed to omit redundancy. You may find an exemplary configuration at the bottom.

## Global

1. `random_state (int, default: None)` the seed for random operations 

## Model 

1. `model_name (str)` HF name of the model you need 
2. `output_dir (str, default: ./results/)` the path of an output dir of your model
3. `num_labels (int, default: 2)` the number of used labels
4. `max_length (int, default: 256)` the maximal sequence length 
5. `padding (str, default: max_length)` pad up to the given length parameter
6. `truncation (bool, default: True)` controls wether to use truncation
7. `return_tensors (str, default: pt)` returns tensors of a particular framework
8. `output_path (str, default: best_model)` name for a model file

Exemplary config of `model`:

```yaml
model:
  max_length: 128
  output_dir: ../../results/
  model_name: distilbert-base-uncased
  num_labels: 12
  padding: max_length
  return_tensors: pt
  truncation: False
  output_path: ../../output_model/
```

## Metrics

Pick metrics that you need as arguments for this section. Complete list of metrics we handle is available [here](/src/t2r2/metrics/metrics.py).

To ommit redundant descriptions - you may find metrics explained under this [link](https://scikit-learn.org/stable/modules/model_evaluation.html).

1. `name (str)` the name of the metric
2. `args (dict[str, Any], default: None)` the dictionary of pairs where keys are metrics' names and values are argument for those metrics

Exemplary config of `metrics`:

```yaml
metrics:
  - name: f1_score
    args:
      average: macro
  - name: slicing_scores
    args: 
      base_directory: 'slicing/'
      default_file_name: "slicing.pickle"
```

## Dataset

For each of the 3 following subsections (Training, Testing and Control) you may additionally provide the following arguments or have them in the additional section `data` to omit unnecessary repetitions:

1. `data_dir (str, default: ./data/)` the path of a directory with the data
2. `output_dir (str, default: ./results/)` the path of a directory for the output
3. `text_column_id (int, default: 0)` the index of the text column
4. `label_column_id (int, default: 1)` the index of the label column
5. `has_header (bool, default: True)` the variable that describes wether the dataset contain the header

Exemplary config of `data`:

```yaml
data:
  data_dir: "../../data/featuresets/"
  output_dir: "../../results/"
  has_header: False
  text_column_id: 1
  label_column_id: 2
```

Also you may enrich training and testing sections with `selectors` subsection.

### Selectors

The following paragraphs briefly describe selectors that we provide.

#### Data Cartography

Implementation of the concept of data cartography as outlined in the paper: https://aclanthology.org/2020.emnlp-main.746/.

To fully understand how it works - check this [notebook](https://github.com/radswn/t2r2/blob/master/notebooks/demo_data_cartography/distilbert/demo_data_cartography.ipynb)

Example of data cartography yaml snippet:

```yaml
perform_data_cartography: True
```

#### LLM selector

LLM selector was prepared with intention of running on high-performance machines with GPU / Google Colab. To provide you with LLM that is small enought to run it on your own we used `TheBloke/Mistral-7B-v0.1-AWQ`.

For this selector you need to provide a `prompt` argument in the config. Bear in mind that LLM is not perfect, and sometimes you might work on your prompt. On our side - we feed the model with your prompt and the data converted to string.

The exemplary snippet in yaml:

```yaml
  selectors:
    - name: llm
      args: 
        prompt: Generate more synthetic examples
```

A notebook that covers an installation of additional libraries needed to run LLM selector is [here](https://github.com/radswn/t2r2/blob/master/notebooks/demo_llm/demo.ipynb)

#### Slicing functions

Check our notebook that explains slicing functions [here](https://github.com/radswn/t2r2/blob/master/notebooks/demo_slicing/demo_slicing.ipynb).

Example of slicing functions yaml snippet:

```yaml
  selectors:
    - name: slicing
      args: 
        result_file: '../../data/slicing/train_slicing.pickle'
        list_of_slicing_functions: [short, textblob_polarity]
```

#### Undersampling

Undersampling performed randomly with [RandomUnderSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) from imblearn.

Example of undersampling yaml snippet:

```yaml
  selectors:
    - name: random_under_sampler
```

#### User Selector

We give you an opportunity to use your own selectors.

1. Prepare a class you want to use - it should inherit from `Selector` class from `t2r2.selector`. Implement its `select` method.
2. When declarating your own selector - provide `module_path` as one of the arguments.

Below we present a simple example how to do it.

Example of user selector yaml snippet:

```yaml
  selectors:
    - name: UserSelector
      args: 
        module_path: ./my_selector.py
```

`my_selector.py` code

```python
import pandas as pd
from t2r2.selector import Selector

class UserSelector(Selector):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[:5]
```

#### Curriculum Learning

To force specific order in which examples will be passed during training:
```yaml
training:
  curriculum_learning: True
```
Then you also need to provide the `order` column in your training data.

Basically, the examples will be _sorted_ according to order column and won't be shuffled.

You can also use the custom selector to dynamically provide the order of your training examples.
For example, to pass examples in the order of increasing lenght of text:
```python
class ClSelector(Selector):
    def select(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["order"] = [len(i) for i in dataset["text"]]
        return dataset
```

## Training

1. `dataset_path (str, default: train.csv)` the name of the file with data
2. `validation_dataset_path (str, default: None)` the validation dataset path
3. `results_file (str, default: train_results.pickle)` the name of the file with results
4. `epochs (int, default: 1)` the number of epochs
5. `batch_size (int, default: 32)` the batch size
6. `learning_rate (float, default: 0.00001)` the learning rate parameter
7. `validation_size (float, default: 0.2)` the validation size (between 0 and 1)
8. `metric_for_best_model (str, default: loss)` the metric that characterizes the best model
9. `perform_data_cartography (bool, default: False)` the switch for performing a data cartography
10. `data_cartography_results (str, default: ./data_cartography_metrics.pickle)` the name of a result file with data cartography metrics
11. `curriculum_learning (bool, default: False)` the switch for performing a curriculum learning

## Testing

1. `dataset_path (str, default: test.csv)` the name of the file with data
2. `results_file (str, default: test_results.pickle)` the name of the file with results

## Control 

1. `dataset_path (str, default: control.csv)` the name of the file with data
2. `results_file (str, default: control_results.pickle)` the name of the file with results

## MLFlow

Enables tracking an experiment (datasets, model and metrics) with the use of MLflow tools. 

1. `experiment_name (str)` the experiment name
2. `tags (dict[str, str])` additional tags such as `version`
3. `tracking_uri (str)` the endpoint of your server

Exemplary config of `mlflow`:

```yaml
mlflow:
  experiment_name: 'my_experiment_1'
  tags:
      version: 'v1'
  tracking_uri: "http://localhost:5000"
```

Check our notebook with MLflow enabled [here](https://github.com/radswn/t2r2/blob/master/notebooks/demo_slicing/demo_slicing.ipynb).

## DVC

Enables versioning of an experiment (datasets, model and metrics), showing differences in parameters as well as in metrics, easy checkouts. 

If you want to switch it on - add the following lines to your config

```yaml
dvc:
  enabled: true
```
(default: off)

For more tips check our notebook that presents a workflow with DVC enabled [here](https://github.com/radswn/t2r2/blob/master/notebooks/demo_data_cartography/distilbert/demo_data_cartography.ipynb).

## Exemplary configs

You may find exemplary `config.yaml` files with their notebooks in this [directory](https://github.com/radswn/t2r2/tree/master/notebooks).

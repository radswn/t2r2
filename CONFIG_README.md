# T2R2 - Config Description

In T2R2 we use configuration files in `yaml` format to specify our needs. Below we describe sections that you may include in your config. We left some fields undescribed to ommit redundancy. You may find an exemplary configuration at the bottom.

## Global

1. `random_state` - [int] the seed for random operations [default: None]
2. `data_dir` - [str - path] the path of a directory with the data [default: ./data/]
3. `output_dir` - [str - path] the path of a directory for the output [default: ./results/]

## Model 

1. `model_name` - [str] HF name of the model you need 
2. `output_dir` - [str] the path of an output dir of your model [default: ./results/]
3. `num_labels` - [int] the number of used labels [default: 2]
4. `max_length` - [int] the maximal sequence length [default: 256] 
5. `padding` - [str] pad up to the given length parameter [default: max_length]
6. `truncation` - [bool] controls wether to use truncation [default: True]
7. `return_tensors` - [str] returns tensors of a particular framework [default: pt]
8. `output_path` - [str] name for a model file [default: best_model]

Exemplary config of Model

```
model:
  model_name: bert-base-cased
  num_labels: 12
```

## Metrics

Pick metrics that you need as arguments for this section. Complete list of metrics we handle is available [here](src\t2r2\metrics\metrics.py)

To ommit redundant descriptions - you may find metrics explained under this [link] (https://scikit-learn.org/stable/modules/model_evaluation.html)

Exemplary config of Metrics

```
metrics:
  - name: accuracy_score
```

## Dataset Vaults

For each of the 3 following sections (Training, Testing and Control) you may additionally provide the following arguments:

1. `text_column_id`: [int] [default: 0]
2. `label_column_id`: [int] [default: 1]
3. `has_header`: [bool] [default: True]

Also you may enrich training and testing sections with `selectors` subsection.

### Selectors

#### Data Cartography

#### LLM selector

#### Slicing

#### Undersampling

Undersampling performed randomly with [RandomUnderSampler] (https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) from imblearn

Exemplary selectors config:
```
  selectors:
    - name: random_under_sampler
      args: { }
    - name: slicing
      args: 
        result_file: '../../data/slicing/train_slicing.pickle'
        list_of_slicing_functions: [short, textblob_polarity]
```

## Training

1. `dataset_path`: [str] the name of the file with data [default: train.csv]
2. `validation_dataset_path`: [str - path] if you want to additionally provide validation dataset - you may do it here [default: None]
3. `results_file`: [str] the name of the file with results [default: train_results.pickle]
4. `epochs`: [int] the number of epochs [default: 1]
5. `batch_size`: [int] the batch size [default: 32]
6. `learning_rate`: [float] learning rate parameter [default: 0.00001]
7. `validation_size`: [float] validation size (between 0 and 1) [default: 0.2]
8. `metric_for_best_model` [str] the metric that characterizes the best model [default: loss]
9. `perform_data_cartography` [bool] the switch for performing a data cartography [default: False]
10. `data_cartography_results` [str] the name of a result file with data cartography metrics  [default: ./data_cartography_metrics.pickle]
11. `curriculum_learning` [bool] the switch for performing a curriculum learning [default: False]

## Testing

1. `dataset_path`: [str] the name of the file with data [default: test.csv]
2. `results_file`: [str] the name of the file with results [default: test_results.pickle]

## Control 

1. `dataset_path`: [str] the name of the file with data [default: control.csv]
2. `results_file`: [str] the name of the file with results [default: control_results.pickle]

## MLFlow

1. `experiment_name` - [str] the experiment name
2. `tags` - [args] additional tags such as `version`
3. `tracking_uri`: [str] the endpoint of your server

###### TODO

Exemplary config of MLFlow:

```
mlflow:
  experiment_name: 'my_experiment_3'
  tags:
      version: 'v1'
  tracking_uri: "http://localhost:5000"
```

## DVC
###### TODO
If you want to switch it on - add the following lines to your config
```
dvc:
  enabled: true
```
[default: off]

## Exemplary config:

```
model:
  model_name: bert-base-cased
  num_labels: 12

metrics:
  - name: accuracy_score

training:
  epochs: 1
  batch_size: 2
  learning_rate: 1e-3
  validation_size: 0.2
  metric_for_best_model: accuracy_score

testing: { }

control: { }
```
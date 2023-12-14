# T2R2 - Config Description

In T2R2 we use configuration files in `yaml` format to specify our needs. Below we describe sections that you may include in your config. We left some fields undescribed to ommit redundancy. You may find an exemplary configuration at the bottom.

## Global

1. `random_state` - [int] sets seed for random operations [default: None]
2. `data_dir` - [str - path] sets data_dir to the given path [default: ./data/]
3. `output_dir` - [str - path] sets output_dir to the given path [default: ./results/]

## Model 

1. `model_name` - [str] HF name of the model you need. 
2. `output_dir` - [str] you may set an output_dir for your model [default: ./results/]
3. `num_labels` - [int] [default: 2] 
4. `max_length` - [int] [default: 256]
5. `padding` - [str] [default: max_length]
6. `truncation` - [bool] [default: True]
7. `return_tensors` - [str] [default: pt]
8. `output_path` - [str] [default: best_model]

Exemplary config of Model

```
model:
  model_name: bert-base-cased
  num_labels: 12
```

## Metrics

Pick metrics that you need as arguments for this section. Complete list is of metrics is available [here](src\t2r2\metrics\metrics.py)
###### TODO
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

Also you may enrich those sections with `selectors` subsection.

###### TODO

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
1. `dataset_path`: [str] name of the file with data [default: train.csv]
2. `validation_dataset_path`: [str - path] if you want to additionally provide validation dataset - you may do it here [default: None]
3. `results_file`: [str] name of the file with results [default: train_results.pickle]
4. `epochs`: [int] [default: 1]
5. `batch_size`: [int] [default: 32]
6. `learning_rate`: [float] [default: 0.00001]
7. `validation_size`: [float] [default: 0.2]
8. `metric_for_best_model` [str]  [default: loss]
9. `perform_data_cartography` [bool] [default: False]
10. `data_cartography_results` [str] [default: ./data_cartography_metrics.pickle]
11. `curriculum_learning` [bool] [default: False]

## Testing

1. `dataset_path`: [str] name of the file with data [default: test.csv]
2. `results_file`: [str] name of the file with results [default: test_results.pickle]

## Control 

1. `dataset_path`: [str] name of the file with data [default: control.csv]
2. `results_file`: [str] name of the file with results [default: control_results.pickle]

## MLFlow

1. `experiment_name` - [str] experiment name
2. `tags` - [args] additional tags such as `version`
3. `tracking_uri`: [str] endpoint of your server
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
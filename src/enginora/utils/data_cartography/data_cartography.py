import os
import logging
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import List
from collections import defaultdict

logger = logging.getLogger()


def compute_correctness(trend: List[float]) -> float:
    """
  Aggregate #times an example is predicted correctly during all training epochs.
  """
    return sum(trend)


def compute_forgetfulness(correctness_trend: List[float]) -> int:
    """
  Given a epoch-wise trend of train predictions, compute frequency with which
  an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
  Based on: https://arxiv.org/abs/1812.05159
  """
    if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
        return 1000
    learnt = False  # Predicted correctly in the current epoch.
    times_forgotten = 0
    for is_correct in correctness_trend:
        if (not learnt and not is_correct) or (learnt and is_correct):
            # nothing changed.
            continue
        elif learnt and not is_correct:
            # Forgot after learning at some point!
            learnt = False
            times_forgotten += 1
        elif not learnt and is_correct:
            # Learnt!
            learnt = True
    return times_forgotten


def compute_data_cartography_metrics(predictions, labels, epochs):
    """
        Given the training dynamics, compute metrics
        based on it, for data map coorodinates.
        Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
        the last two being baselines from prior work
        (Example Forgetting: https://arxiv.org/abs/1812.05159 and
         Active Bias: https://arxiv.org/abs/1704.07433 respectively).
        Returns:
        - DataFrame with these metrics.
        - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
        """
    confidence_ = {}
    variability_ = {}
    threshold_closeness_ = {}
    correctness_ = {}
    forgetfulness_ = {}
    num_tot_epochs = epochs
    # Functions to be applied to the data.
    variability_func = lambda conf: np.std(conf)
    threshold_closeness_func = lambda conf: conf * (1 - conf)
    training_accuracy = defaultdict(float)
    logger.info(f"Computing training dynamics across {num_tot_epochs} epochs")
    logger.info("Metrics computed: confidence, variability, correctness, forgetfulness, threshold_closeness")

    probas = {i: [] for i in range(len(predictions))}
    targets = {i: [] for i in range(len(predictions))}

    for guid in tqdm.tqdm(range(len(predictions))):
        correctness_trend = []
        true_probs_trend = []
        record = predictions.iloc[guid]
        for i, epoch_prob in enumerate(record):
            epoch_prob = epoch_prob[0]
            true_class_prob = float(epoch_prob[labels[guid]])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_prob)
            is_correct = (prediction == labels[guid]).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            probas[i].append(epoch_prob)
            targets[i].append(labels[guid])

        correctness_[guid] = compute_correctness(correctness_trend)
        confidence_[guid] = np.mean(true_probs_trend)
        variability_[guid] = variability_func(true_probs_trend)

        forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
        threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])

    column_names = ['guid',
                    'threshold_closeness',
                    'confidence',
                    'variability',
                    'correctness',
                    'forgetfulness', ]
    df = pd.DataFrame([[guid,
                        threshold_closeness_[guid],
                        confidence_[guid],
                        variability_[guid],
                        correctness_[guid],
                        forgetfulness_[guid],
                        ] for i, guid in enumerate(correctness_)], columns=column_names)

    return df


def create_plot(dataframe, output_dir :str, hue_metric='correct.', title='data_cartography', model_name='model', show_hist=True):
    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=25000 if dataframe.shape[0] > 25000 else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac=lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        ax0 = axs
    else:
        fig = plt.figure(figsize=(16, 10), )
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])

        ax0 = fig.add_subplot(gs[0, :])

    ### Make the scatterplot.

    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    an1 = ax0.annotate("ambiguous", xy=(0.9, 0.5), xycoords="axes fraction", fontsize=15, color='black',
                       va="center", ha="center", rotation=350, bbox=bb('black'))
    an2 = ax0.annotate("easy-to-learn", xy=(0.27, 0.85), xycoords="axes fraction", fontsize=15, color='black',
                       va="center", ha="center", bbox=bb('r'))
    an3 = ax0.annotate("hard-to-learn", xy=(0.35, 0.25), xycoords="axes fraction", fontsize=15, color='black',
                       va="center", ha="center", bbox=bb('b'))

    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=(1.01, 0.5), loc='center left', fancybox=True, shadow=True)
    else:
        plot.legend(fancybox=True, shadow=True, ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    if show_hist:
        plot.set_title(f"{model_name}- Data Map", fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')

        plot2 = sns.countplot(x="correct.", data=dataframe, color='#86bf91', ax=ax3)
        ax3.xaxis.grid(True)  # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('')

    fig.tight_layout()

    os.makedirs(f'{output_dir}/figures/', exist_ok=True)
    filename = f'{output_dir}/figures/{title}_{model_name}.pdf' if show_hist else f'{output_dir}/figures/compact_{title}_{model_name}.pdf'
    fig.savefig(filename, dpi=300)


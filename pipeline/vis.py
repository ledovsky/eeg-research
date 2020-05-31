import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_violins(df, features, labels=None, ax=None):
    """

    Args:
        df: pd.DataFrame with features, target and fn columns
        features: list of features to plot
        labels: mapping for target. for example {0: 'HC', 1: 'MDD'}
        ax: you can provide your ax

    Returns:
        ax

    """
    if ax is None:
        fig = plt.figure(figsize=[6, 2 * len(features)])
        ax = plt.gca()

    df_features = df[features + ['fn']].copy()
    df_features = (
        df_features
            .set_index('fn')
            .stack()
            .reset_index()
            .rename(columns={'level_1': 'feature', 0: 'value'})
            .merge(df[['fn', 'target']], on='fn')
            .rename(columns={'target': 'Group'})
    )

    if labels is not None:
        df_features['Group'] = df_features['Group'].apply(lambda val: labels[val])

    sns.violinplot(x="value", y="feature", hue="Group",
                   data=df_features,
                   order=features,
                   linewidth=0.5,
                   palette="deep",
                   split=True,
                   inner='stick',
                   orient='h',
                   scale='width',
                   ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('')

    return ax


def plot_lr_weights(model, features, ax=None):

    if ax is None:
        fig = plt.figure(figsize=[6, 0.5 * len(features)])
        ax = plt.gca()

    pal = sns.color_palette()

    idx = np.argsort(model.coef_[0])

    coefs = model.coef_[0][idx]
    colors = [pal[0] if c > 0 else pal[3] for c in coefs]
    ax.barh(np.array(features)[idx], coefs, color=colors)

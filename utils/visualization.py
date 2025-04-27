import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind 

def plot_trend(df, x, y, title):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    return fig

def plot_extreme_events(df, x, palette, title):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(data=df, x=x, palette=palette, ax=ax)
    ax.set_title(title)
    return fig

def plot_correlation_heatmap(df, cols):
    fig, ax = plt.subplots(figsize=(8,6))
    corr_matrix = df[cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig
def perform_t_test(df, date_col, value_col, period1_start, period1_end, period2_start, period2_end):
    period1_data = df[(df[date_col] >= str(period1_start)) & (df[date_col] <= str(period1_end))][value_col]
    period2_data = df[(df[date_col] >= str(period2_start)) & (df[date_col] <= str(period2_end))][value_col]

    t_stat, p_val = ttest_ind(period1_data, period2_data, nan_policy='omit')
    return t_stat, p_val
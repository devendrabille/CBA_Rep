from typing import Any, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> None:
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_boxplot(df: pd.DataFrame, column: str, title: str = "Boxplot") -> None:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column], color="#6BAED6")
    plt.title(f"{title}: {column}")
    plt.tight_layout()
    plt.show()

def plot_categorical_counts(df: pd.DataFrame, column: str, title: str = "Categorical Counts") -> None:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, palette="Set2")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_plot_as_image(fig: Any, filename: str) -> None:
    fig.savefig(filename, bbox_inches='tight')

def plot_outliers_boxplots(df: pd.DataFrame, numeric_columns: List[str]) -> None:
    for column in numeric_columns:
        plot_boxplot(df, column, title="Boxplot for Outlier Detection")
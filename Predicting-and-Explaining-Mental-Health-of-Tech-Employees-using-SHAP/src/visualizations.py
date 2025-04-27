import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
REPORT_DIR = '../reports'
FIGURE_DIR = os.path.join(REPORT_DIR, 'figures')
NUMERICAL_EDA_DIR = os.path.join(REPORT_DIR, 'numerical_eda') 
CATEGORICAL_EDA_DIR = os.path.join(REPORT_DIR, 'categorical_eda')

# Create directories if they don't exist
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(NUMERICAL_EDA_DIR, exist_ok=True) 
os.makedirs(CATEGORICAL_EDA_DIR, exist_ok=True) 


# 1. Target Variable Distribution
def plot_target_variable(df, target_variable_name="CurrentMHDisorder"):
    """
    Plots the distribution of the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_variable_name (str, optional): The name of the target variable column. Defaults to "CurrentMHDisorder".
    """
    target_counts = df[target_variable_name].value_counts(normalize=True) * 100
    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_counts.index, y=target_counts.values, hue=target_counts.index, palette='viridis')
    plt.title(f"Distribution of '{target_variable_name}'")
    plt.xlabel(target_variable_name)
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'target_variable.png'))
    plt.show()



# 2. Distribution of Numerical Features
def plot_numerical_features(df):
    """
    Plots the distribution of each numerical feature in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(NUMERICAL_EDA_DIR, f'{col}_distribution.png')) 
        plt.show()



# 3. Distribution of Categorical Features
def plot_categorical_features(df,categorical_cols):
    """
    Plots the distribution of each categorical feature in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index,hue=col, palette='Set2')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(CATEGORICAL_EDA_DIR, f'{col}_countplot.png')) 
        plt.show()



# 4. Relationship between Categorical Features and Target Variable
def categorical_vs_target(df, categorical_cols,target_variable_name): 
    
    """
    Plots the relationship between each categorical feature and the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_variable_name (str): The name of the target variable column.
        categorical_cols (list): A list of categorical column names.
    """
    
    for col in categorical_cols:
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x=col, hue=target_variable_name, order=df[col].value_counts().index, palette='inferno')
        plt.title(f"{col} vs {target_variable_name}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=target_variable_name)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f'{col}_vs_target.png'))
        plt.show()



# 5. Relationship between Numerical Features and Target Variable (Boxplots)
def numerical_vs_target(df, target_variable_name):
    """
    Plots the relationship between each numerical feature and the target variable using boxplots.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_variable_name (str): The name of the target variable column.
    """
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target_variable_name, y=col, data=df, palette='Set3')
        plt.title(f"{col} vs {target_variable_name}")
        plt.xlabel(target_variable_name)
        plt.ylabel(col)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f'{col}_boxplot_vs_target.png'))
        plt.show()



# 6. Correlation Heatmap of Numerical Features
def plot_correlation_heatmap(df):
    """
    Plots a heatmap of the correlation matrix for numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    correlation_matrix = df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'numerical_correlation_heatmap.png'))
    plt.show()


# 7. Pairplot of Numerical Features
def plot_pairplot(df):
    """
    Plots a pairplot of the numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    sns.pairplot(df[numerical_cols])
    plt.title("Pairplot of Numerical Features")
    plt.savefig(os.path.join(FIGURE_DIR, 'pairplot_numerical_features.png'))
    plt.show()
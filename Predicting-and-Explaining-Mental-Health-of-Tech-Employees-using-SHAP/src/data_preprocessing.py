import pandas as pd


def load_data(filepath):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
                           Returns None and prints an error message if the file is not found.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from: {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at: {filepath}")
        return None

# Example usage:
# df = load_data('./data/raw/mental-heath-in-tech-2016.csv')
# if df is not None:
#     print(df.head())

# src/data_preprocessing.py



def rename_columns(df, renamed_columns):
    """
    Renames the columns of a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to rename.
        renamed_columns (list): A list of new column names.
                                 The order of names in this list should correspond
                                 to the order of the original columns in the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    columns = df.columns.tolist()
    if len(columns) != len(renamed_columns):
        raise ValueError("The number of new column names must match the number of original columns.")
    df = df.rename(columns=dict(zip(columns, renamed_columns)))
    return df


# def rename_columns(df, *new_columns_name):
#     columns = df.columns.tolist()
#     df = df.rename(columns=dict(zip(columns, new_columns_name)))
#     return df


gender_mapping = {
    'Male': ['Male', 'male', 'Male ', 'M', 'm', 'man', 'Cis male', 'Male.',
             'male 9:1 female, roughly', 'Male (cis)', 'Man', 'Sex is male', 'cis male', 'Malr',
             'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
             'mail', 'M|', 'Male/genderqueer', 'male ', 'Cis Male', 'Male (trans, FtM)',
             'cisdude', 'cis man', 'MALE'],
    'Female': ['Female', 'female', 'I identify as female.', 'female ',
               'Female assigned at birth ', 'F', 'Woman', 'fm', 'f', 'Cis female ',
               'Transitioned, M2F', 'Genderfluid (born female)', 'Female or Multi-Gender Femme',
               'Female ', 'woman', 'female/woman', 'Cisgender Female', 'fem',
               'Female (props for making this a freeform field, though)', ' Female', 'Cis-woman',
               'female-bodied; no feelings about gender', 'AFAB'],
    'Others': ['Bigender', 'non-binary', 'Other/Transfeminine', 'Androgynous', 'Other',
               'nb masculine', 'none of your business', 'genderqueer', 'Human', 'Genderfluid',
               'Enby', 'genderqueer woman', 'mtf', 'Queer', 'Agender', 'Fluid', 'Nonbinary',
               'human', 'Unicorn', 'Genderqueer', 'Genderflux demi-girl', 'Transgender woman']
}


def standardize_gender(df):
    for category, replacements in gender_mapping.items():
        df['Gender'].replace(to_replace=replacements,
                             value=category, inplace=True)

    import logging
    logging.info("Gender feature standardized.")
    return df


def impute_missing_values(df):
    """Imputes missing values in a DataFrame using mode for categorical and mean for numerical columns."""
    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype == 'object' or df[column].dtype == 'category':
                df[column]=df[column].fillna(df[column].mode()[0])
            else:
                df[column]=df[column].fillna(df[column].mean())
    return df


# Analyze Categorical features
def analyze_categorical(df):
    """Analyzes categorical columns in the DataFrame."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols_with_multiple_values = [
        col for col in categorical_cols if df[col].nunique() > 1]

    import logging
    logging.info("Categorical columns and their unique values:")
    for col in categorical_cols_with_multiple_values:
        logging.info(
            f"{col} ({df[col].nunique()} unique values): {df[col].unique()}")
        print(
            f"Column '{col}' has {df[col].nunique()} unique values: {df[col].unique()}")

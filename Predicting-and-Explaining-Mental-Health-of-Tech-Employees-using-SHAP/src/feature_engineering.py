import pandas as pd
import numpy as np
from src.data_preprocessing import gender_mapping


def bin_age(df):
    bins = [18, 30, 40, 50, 60, float('inf')]
    labels = ['18-29', '30-39', '40-49', '50-59', '60-100']
    df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df = df.reindex(sorted(df.columns), axis=1)
    print("Age feature binned.")
    return df

def encode_categorical_columns(categorical_df):
    from sklearn.preprocessing import LabelEncoder
    encoded_df = pd.DataFrame()
    for col_name in categorical_df.columns:
        le = LabelEncoder()
        encoded_df[col_name] = le.fit_transform(categorical_df[col_name])
    return encoded_df




# More Feature Engineering
def features_engineering_extraction(df):

    df['CurrentMHDisorderConditions_count'] = df['CurrentMHDisorderConditions'].apply(
        lambda x: len(str(x).split('|')) if pd.notnull(x) else 0)
    df['MHSelfDiagnosisConditions_count'] = df['MHSelfDiagnosis'].apply(
        lambda x: len(str(x).split('|')) if pd.notnull(x) else 0)
    df['MHPHDiagnosisConditions_count'] = df['ProfessionalMHDiagnosisDetails'].apply(
        lambda x: len(str(x).split('|')) if pd.notnull(x) else 0)
    df['role_count'] = df['WorkPosition'].apply(
        lambda x: len(str(x).split('|')) if pd.notnull(x) else 0)
    df['EmployeeCount'] = df['EmployeeCount'].replace({'1-5': '2-25', '6-25': '2-25', '26-100': '26-100',
                                                       "100-500": "101-500", '500-1000': '500+',
                                                       'More than 1000': '500+'})
    df['EmployeeCount'] = np.where(
        (df['SelfEmployed'] == 'Yes'), '1', df['EmployeeCount'])

    # Combine Employment Features
    df['EmploymentCompanySize'] = 'Other'
    df.loc[(df['EmployeeCount'] == '1'), 'EmploymentCompanySize'] = 'Self-Employed'
    df.loc[(df['EmployeeCount'].isin(['2-25', '26-100', '101-500', '500+']))
        & (df['role_count'] >= 1), 'EmploymentCompanySize'] = 'Tech-Role'
    df.loc[(df['EmployeeCount'] == '2-25') & (df['TechCompany'] == 1.0) &
        (df['role_count'] >= 1), 'EmploymentCompanySize'] = 'Tech Employee Small Company'
    df.loc[(df['EmployeeCount'] == '26-100') & (df['TechCompany'] == 1.0) &
        (df['role_count'] >= 1), 'EmploymentCompanySize'] = 'Tech Employee Medium Company'
    df.loc[(df['EmployeeCount'] == '101-500') & (df['TechCompany'] == 1.0) &
        (df['role_count'] >= 1), 'EmploymentCompanySize'] = 'Tech Employee Large Company'
    df.loc[(df['EmployeeCount'] == '500+') & (df['TechCompany'] == 1.0) &
        (df['role_count'] >= 1), 'EmploymentCompanySize'] = 'Tech Employee Corporation Company'
    df.loc[(df['TechCompany'] == 0.0) & (df['TechRole'] == 0.0) | (
        df['role_count'] == 0), 'EmploymentCompanySize'] = 'Non-Tech Employee'

    return df
"""
Generate a clean, ready-to-use Titanic dataset with engineered features.
Useful for ML learners who want to skip feature engineering.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = '/run/media/bugs/Data/Febriyansyah-Other/Project/github-repo/NOTEBOOK-KAGGLE/CLASSIFICATION/titanic-survival'
OUT_DIR = '/run/media/bugs/Data/Febriyansyah-Other/Project/github-repo/NOTEBOOK-KAGGLE/CLASSIFICATION/titanic-survival/dataset_export'

os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))


def engineer_features(df):
    """Feature engineering applied identically to train & test."""
    df = df.copy()

    # Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # Family size & alone flag
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Cabin deck
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'] = df['Deck'].fillna('U')

    # Fare log
    df['Fare_log'] = np.log1p(df['Fare'])

    # Age imputation by Title median
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

    # Embarked imputation
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fare imputation
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Fare_log'] = np.log1p(df['Fare'])

    return df


train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Save clean datasets
train_fe.to_csv(os.path.join(OUT_DIR, 'train_engineered.csv'), index=False)
test_fe.to_csv(os.path.join(OUT_DIR, 'test_engineered.csv'), index=False)

print('Saved engineered datasets to', OUT_DIR)
print('train_engineered.csv:', train_fe.shape)
print('test_engineered.csv:', test_fe.shape)
print('\nColumns:', list(train_fe.columns))
print('\nMissing values (train):', int(train_fe.isna().sum().sum()))
print('Missing values (test):', int(test_fe.isna().sum().sum()))
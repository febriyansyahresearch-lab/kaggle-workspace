"""
Titanic model optimization - target: public score >= 0.77511 (bronze medal)
Strategy: reduce overfitting gap (CV 0.8372 -> public 0.76315) with better
feature engineering, regularization, and stacking.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, ExtraTreesClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
DATA_DIR = '/run/media/bugs/Data/Febriyansyah-Other/Project/github-repo/NOTEBOOK-KAGGLE/CLASSIFICATION/titanic-survival'

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))


def engineer_features(df):
    """Advanced feature engineering applied identically to train & test."""
    df = df.copy()

    # --- Title ---
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # --- Family ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Fare_per_person'] = df['Fare'] / df['FamilySize']

    # --- Cabin ---
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    df['Cabin_known'] = df['Cabin'].notna().astype(int)
    df['Cabin_count'] = df['Cabin'].fillna('').str.split(' ').apply(len)

    # --- Ticket frequency (group size by ticket) ---
    ticket_counts = df['Ticket'].map(df['Ticket'].value_counts())
    df['Ticket_freq'] = ticket_counts

    # --- Fare ---
    df['Fare_log'] = np.log1p(df['Fare'])
    df['Fare_per_person_log'] = np.log1p(df['Fare_per_person'])

    # --- Age imputation by Title median ---
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age_bin'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 30, 50, 80],
                           labels=['Child', 'Kid', 'Teen', 'Adult', 'Middle', 'Senior'])

    # --- Embarked ---
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # --- Fare imputation ---
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Fare_log'] = np.log1p(df['Fare'])
    df['Fare_per_person'] = df['Fare'] / df['FamilySize']
    df['Fare_per_person_log'] = np.log1p(df['Fare_per_person'])

    # --- Interactions ---
    df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
    df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
    df['Deck_Pclass'] = df['Deck'] + '_' + df['Pclass'].astype(str)

    # --- Name length ---
    df['Name_len'] = df['Name'].str.len()

    return df


train_fe = engineer_features(train)
test_fe = engineer_features(test)

print('Missing after engineering (train):', int(train_fe.isna().sum().sum()))
print('Missing after engineering (test): ', int(test_fe.isna().sum().sum()))

# Feature sets to compare
FEATURES_BASE = ['Pclass', 'Sex', 'Age', 'Fare_log', 'FamilySize', 'IsAlone',
                 'Title', 'Deck', 'Embarked']
FEATURES_V2 = ['Pclass', 'Sex', 'Age', 'Fare_log', 'Fare_per_person_log',
               'FamilySize', 'IsAlone', 'Title', 'Deck', 'Embarked',
               'Cabin_known', 'Cabin_count', 'Ticket_freq', 'Age_bin',
               'Sex_Pclass', 'Title_Pclass', 'Deck_Pclass', 'Name_len']

numeric_features = ['Age', 'Fare_log', 'Fare_per_person_log', 'FamilySize',
                    'Cabin_count', 'Ticket_freq', 'Name_len']
categorical_features = ['Pclass', 'Sex', 'IsAlone', 'Title', 'Deck', 'Embarked',
                        'Cabin_known', 'Age_bin', 'Sex_Pclass', 'Title_Pclass',
                        'Deck_Pclass']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ]), categorical_features),
])

X = train_fe[FEATURES_V2]
y = train_fe['Survived']
X_test = test_fe[FEATURES_V2]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

models = {
    'LogisticRegression': LogisticRegression(max_iter=3000, C=0.5, random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=1000, max_depth=6, min_samples_leaf=4,
                                           random_state=RANDOM_STATE),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=1000, max_depth=8, min_samples_leaf=3,
                                       random_state=RANDOM_STATE),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.03,
                                                   max_depth=3, subsample=0.8,
                                                   random_state=RANDOM_STATE),
    'SVC': CalibratedClassifierCV(SVC(C=1.0, gamma='scale', random_state=RANDOM_STATE),
                                  ensemble=False),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                             random_state=RANDOM_STATE, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                               random_state=RANDOM_STATE, verbose=-1),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    results[name] = scores.mean()
    print(f'{name:22s} CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})')

best = max(results, key=results.get)
print(f'\nBest single model: {best} ({results[best]:.4f})')

# --- Voting ensemble ---
voting = VotingClassifier(estimators=[
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                          random_state=RANDOM_STATE, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            random_state=RANDOM_STATE, verbose=-1)),
    ('rf', RandomForestClassifier(n_estimators=1000, max_depth=6, min_samples_leaf=4,
                                  random_state=RANDOM_STATE)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                                      subsample=0.8, random_state=RANDOM_STATE)),
    ('svc', CalibratedClassifierCV(SVC(C=1.0, gamma='scale', random_state=RANDOM_STATE),
                                   ensemble=False)),
], voting='soft', n_jobs=-1)

voting_pipe = Pipeline([('prep', preprocessor), ('clf', voting)])
voting_scores = cross_val_score(voting_pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'Voting ensemble CV accuracy: {voting_scores.mean():.4f} (+/- {voting_scores.std():.4f})')

# --- Stacking ensemble ---
stacking = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                              random_state=RANDOM_STATE, eval_metric='logloss')),
        ('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                random_state=RANDOM_STATE, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=1000, max_depth=6, min_samples_leaf=4,
                                      random_state=RANDOM_STATE)),
        ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                                          subsample=0.8, random_state=RANDOM_STATE)),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
    cv=5, n_jobs=-1, stack_method='predict_proba',
)

stacking_pipe = Pipeline([('prep', preprocessor), ('clf', stacking)])
stacking_scores = cross_val_score(stacking_pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'Stacking ensemble CV accuracy: {stacking_scores.mean():.4f} (+/- {stacking_scores.std():.4f})')

# --- Generate submission with best ensemble ---
best_pipe = stacking_pipe if stacking_scores.mean() >= voting_scores.mean() else voting_pipe
best_pipe.fit(X, y)
test_pred = best_pipe.predict(X_test)

out = pd.DataFrame({'PassengerId': test_fe['PassengerId'], 'Survived': test_pred})
out_path = os.path.join(DATA_DIR, 'submission_v2.csv')
out.to_csv(out_path, index=False)
print(f'\nSubmission saved to {out_path}')
print(out['Survived'].value_counts())
print(out.head(10))
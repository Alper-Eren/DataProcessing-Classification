import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import ttest_rel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier

# Load the CSV file
data = pd.read_csv('data.csv', delimiter=';')

# Convert categorical variables to numerical representations
data['Noise'] = data['Noise'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['Lighting'] = data['Lighting'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['HasEquipment'] = data['HasEquipment'].map({'No': 0, 'Yes': 1})
data['Communication'] = data['Communication'].map({'Weak': 0, 'Strong': 1})
data['AirConditioning'] = data['AirConditioning'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['Temparature'] = data['Temparature'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['Hygiene'] = data['Hygiene'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['Stress'] = data['Stress'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['BreakSatisfactoriness'] = data['BreakSatisfactoriness'].map({'Insufficient': 0, 'Sufficient': 1})
data['WorkPlace'] = data['WorkPlace'].map({'Remote': 0, 'Hybrid': 1, 'OnSite': 2})
data['Efficiency'] = data['Efficiency'].map({'Low': 0, 'Moderate': 1, 'High': 2})


# Split the data into features and target variables
X = data.drop('Efficiency', axis=1)
y = data['Efficiency']

# Create estimators
estimators = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('kNN', KNeighborsClassifier(n_neighbors=10))
]

# Create a pipeline with feature selection, PCA, normalization etc.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif)),
    ('pca', PCA()),
    ('estimator', None)
])

# Perform one-hot encoding for categorical variables
categorical_features = ['Noise', 'Lighting', 'HasEquipment', 'Communication', 'AirConditioning', 'Temparature', 'Hygiene', 'Stress', 'BreakSatisfactoriness', 'WorkPlace']
categorical_indices = [X.columns.get_loc(feature) for feature in categorical_features]

categorical_transformer = OneHotEncoder(sparse=False, drop='first')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_indices)
    ],
    remainder='passthrough'
)

X_final = preprocessor.fit_transform(X)

# Perform 10-fold cross-validation for each estimator
results = []
for name, estimator in estimators:
    pipeline.set_params(estimator=estimator)
    scores = cross_val_score(pipeline, X, y, cv=10)
    results.append((name, scores))

# Perform feature selection, PCA transformation, and normalization
feature_selection_scores = []
pca_scores = []
normalized_scores = []

# Calculate score for the pipeline with normalized data
for k in range(1, len(X.columns) + 1):
    pipeline.set_params(feature_selection__k=k)
    feature_selection_scores.append(cross_val_score(pipeline, X, y, cv=10).mean())

for n_components in range(1, len(X.columns) + 1):
    pipeline.set_params(pca__n_components=n_components)
    pca_scores.append(cross_val_score(pipeline, X, y, cv=10).mean())

normalized_scores.append(cross_val_score(pipeline, X, y, cv=10).mean())

# Perform t-test for pairwise comparisons
ttest_results = []
for i in range(len(results)):
    for j in range(i + 1, len(results)):
        estimator_i = results[i][1]
        estimator_j = results[j][1]
        t_statistic, p_value = ttest_rel(estimator_i, estimator_j)
        ttest_results.append((results[i][0], results[j][0], t_statistic, p_value))

# Plot accuracy scores for each estimator
plt.figure(figsize=(10, 6))
plt.boxplot([scores for _, scores in results])
plt.xticks(range(1, len(estimators) + 1), [name for name, _ in estimators])
plt.xlabel('Estimators')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores for Different Estimators')
plt.show()

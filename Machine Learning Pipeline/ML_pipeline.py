###############################################################################
# This Python script performs a comprehensive analysis for feature selection 
# and model interpretation in a classification task using radiomic data. It 
# begins by loading a CSV dataset, preprocessing the features by removing 
# special characters, and separating the input features (X) from the labels
# (y). It then evaluates each feature individually using a pipeline that 
# includes BorderlineSMOTE for class balancing, standardization, and logistic 
# regression, reporting the F1 score via stratified cross-validation. A 
# horizontal bar plot is generated to visualize the performance of each feature.
# The script also assesses the classification performance using all features 
# together and applies Recursive Feature Elimination with Cross-Validation 
# (RFECV) to identify the optimal subset of features that maximize the F1 score.
# The best features selected by RFECV are then used to train a logistic 
# regression model, and LIME (Local Interpretable Model-agnostic Explanations) 
# is employed to explain the model's decision for a single sample instance. 
# Throughout the process, the script saves various plots to provide visual
# insights into feature relevance and model interpretability.
#
# Author:      Dr. Pamela Franco
# Time-stamp:  2025-05-08
# E-mail:      pamela.franco@unab.cl /pafranco@uc.cl
# Python 3.10 compatible
###############################################################################
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE

from lime.lime_tabular import LimeTabularExplainer

# Plot settings
plt.rcParams['text.usetex'] = True  
plt.rcParams['font.family'] = 'serif'

###############################################################################
# Load dataset
df = pd.read_csv("FeatureExtraction_Data_AllTexture2.csv")
le = LabelEncoder()

X = df.drop(['Label'], axis=1)
y = df['Label'].values

# Clean feature names
df_clean = X.rename(columns=lambda x: re.sub('[^*A-Za-z0-9_ ]+', '', x))
feature_names = list(df_clean.columns)
X = df_clean

###############################################################################
def evaluate_features_univariate(X, y, cv=5, save_path='univariate_feature_evaluation.png',
                                  highlight_top_n=5):
    print("Evaluating each feature individually using Logistic Regression + BorderlineSMOTE...")
    results = []

    for feature in X.columns:
        X_feature = X[[feature]]

        pipeline = ImbPipeline([
            ('smote', BorderlineSMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
        ])

        scores = cross_val_score(pipeline, X_feature, y,
                                 cv=StratifiedKFold(cv, shuffle=True, random_state=42),
                                 scoring='f1')
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"Feature: {feature:<30} | Mean F1: {mean_score:.4f} ± {std_score:.4f}")
        results.append((feature, mean_score, std_score))

    df_results = pd.DataFrame(results, columns=['Feature', 'Mean_F1', 'Std_F1'])
    df_sorted = df_results.sort_values(by='Mean_F1', ascending=False).reset_index(drop=True)

    bar_colors = ['blue' if i < highlight_top_n else 'skyblue' for i in range(len(df_sorted))]

    Font_size = 24
    fig, ax = plt.subplots(figsize=(14, 0.4 * len(df_sorted) + 4))
    bars = ax.barh(df_sorted['Feature'], df_sorted['Mean_F1'],
                   xerr=df_sorted['Std_F1'],
                   color=bar_colors, edgecolor='black',
                   error_kw=dict(ecolor='black', capsize=4, elinewidth=1.5))

    ax.set_xlabel('Mean F1 Score', fontsize=Font_size)
    ax.set_ylabel('Features', fontsize=Font_size)
    ax.tick_params(axis='x', labelsize=Font_size)
    ax.tick_params(axis='y', labelsize=Font_size)

    ax.invert_yaxis()
    ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.7)
    ax.set_ylim(-1, len(df_sorted))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Plot saved as: {save_path}")
    return df_sorted

###############################################################################
def evaluate_all_features(X, y, cv=5):
    print("\nEvaluating all features together with Logistic Regression + BorderlineSMOTE...")
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', BorderlineSMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    scores = cross_val_score(pipeline, X, y,
                             cv=StratifiedKFold(cv, shuffle=True, random_state=42),
                             scoring='f1')
    print(f"All features combined - Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

###############################################################################
def recursive_feature_elimination(X, y, cv=5, save_path='rfecv_curve.png'):
    print("\nRunning RFECV (SMOTE cannot be used inside RFECV; using class_weight='balanced')...")

    model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=StratifiedKFold(cv, shuffle=True, random_state=42),
        scoring='f1',
        min_features_to_select=1
    )

    rfecv.fit(X, y)

    print(f"\nOptimal number of features: {rfecv.n_features_}")
    selected_features = X.columns[rfecv.support_]
    print(f"Selected features: {list(selected_features)}\n")

    mean_scores = rfecv.cv_results_['mean_test_score']
    std_scores = rfecv.cv_results_['std_test_score']
    num_features = range(1, len(mean_scores) + 1)

    print("F1 Score per number of selected features:")
    for i, (mean, std) in enumerate(zip(mean_scores, std_scores), start=1):
        print(f"{i} feature(s): F1 = {mean:.4f} ± {std:.4f}")

    # Find the index of first global max with number of features = 3 if possible
    # Otherwise fallback to global max (np.argmax)
    target_features = 3

    def is_global_max_at(i, scores):
        val = scores[i]
        return all(val >= s for s in scores[i:])

    best_idx = None
    for i in range(len(mean_scores)):
        if (i + 1) == target_features and is_global_max_at(i, mean_scores):
            best_idx = i
            break
    if best_idx is None:
        best_idx = np.argmax(mean_scores)

    max_idx = best_idx + 1
    max_score = mean_scores[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        num_features,
        mean_scores,
        yerr=std_scores,
        marker='o',
        capsize=4,
        elinewidth=1,
        ecolor='gray',
        label='F1 Score ± std'
    )
    ax.plot(max_idx, max_score, 'ro', markersize=10, label=f'Max at {max_idx} features')

    ax.set_xlabel("Number of Features Selected", fontsize=20)
    ax.set_ylabel("Cross-Validation F1 Score", fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, len(mean_scores))
    ax.legend(fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"\nRFECV curve saved as: {save_path}")

    return selected_features

###############################################################################
# Run evaluations
univariate_results = evaluate_features_univariate(X, y)
all_features_scores = evaluate_all_features(X, y)
selected_features = recursive_feature_elimination(X, y)

###############################################################################
# Select top 3 most important features based on model coefficients
model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
model.fit(X[selected_features], y)

coef_importance = np.abs(model.coef_[0])
feature_importance = pd.Series(coef_importance, index=selected_features)
top_features = feature_importance.nlargest(3).index

print(f"\nTop 3 features (by model importance): {list(top_features)}")

X_top = X[top_features]

explainer = LimeTabularExplainer(
    training_data = X_top.values,
    feature_names = list(top_features),
    class_names = list(np.unique(y).astype(str)),
    mode = 'classification',
    discretize_continuous=True,
    random_state=42
)

idx = 0
sample = X_top.iloc[idx].values.reshape(1, -1)

model.fit(X_top, y)

exp = explainer.explain_instance(
    data_row = sample[0],
    predict_fn = model.predict_proba,
    num_features = 3
)

print("\nLIME explanation for sample index", idx)
for feature, weight in exp.as_list():
    print(f"Feature: {feature:20} | Weight (local importance): {weight:.4f}")

fig = exp.as_pyplot_figure()
plt.xlabel('Weight', fontsize=20)
plt.ylabel('Features', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('lime_explanation_top3.png', dpi=300)
plt.show()

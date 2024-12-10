import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from joblib import Parallel, delayed
from scipy.stats import spearmanr
data = pd.read_csv('heart.csv')  
print("\nDataset Shape:", data.shape)
print("\nDataset Info:")
data.info()
print("\nFirst Few Rows:")
print(data.head())
target = 'HadHeartAttack'  


if data[target].dtype == 'object':
    label_encoder = LabelEncoder()
    data[target] = label_encoder.fit_transform(data[target])


X = data.drop(columns=[target])
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()


for col in categorical_features:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])


sns.countplot(x=target, data=data)
plt.title("Target Variable Distribution")
plt.show()


for col in numeric_features:
    plt.figure()
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()


correlation_results = {}


for col in numeric_features:
    pearson_corr = data[col].corr(data[target])
    spearman_corr, _ = spearmanr(data[col], data[target])
    correlation_results[col] = max(abs(pearson_corr), abs(spearman_corr))


def cramers_v(x, y):
    from scipy.stats import chi2_contingency
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

for col in categorical_features:
    correlation = cramers_v(data[col], data[target])
    correlation_results[col] = correlation


correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index', columns=['Correlation with Target'])
correlation_df = correlation_df.sort_values(by='Correlation with Target', ascending=False)
print("\nCorrelation Table:")
print(correlation_df)


plt.figure(figsize=(10, 8))
sns.barplot(y=correlation_df.index, x=correlation_df['Correlation with Target'], palette='coolwarm', hue=correlation_df.index, dodge=False, legend=False)
plt.title("Feature Correlation with Target")
plt.xlabel("Correlation")
plt.ylabel("Features")
plt.show()


for col in categorical_features:
    label_encoder = LabelEncoder()
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train.loc[:, numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])
X_test.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20, min_samples_leaf=10),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=50, subsample=0.8),
    "Neural Network": MLPClassifier(random_state=42, max_iter=300),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(probability=True, random_state=42, kernel='rbf', C=1.0, gamma='scale', max_iter=50)
}


def evaluate_model(name, model):
    
    if name == "Support Vector Machine":
        svm_sample_size = min(25000, len(X_train))  
        X_train_svm = X_train.sample(n=svm_sample_size, random_state=42)
        y_train_svm = y_train[X_train_svm.index]
        model.fit(X_train_svm, y_train_svm)
    else:
        model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Results:")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    if roc_auc is not None:
        print("ROC-AUC:", roc_auc)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot(cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    return {
        'Model': name,
        'Accuracy': accuracy,
        'ROC-AUC': roc_auc
    }

results = []
for name, model in models.items():
    results.append(evaluate_model(name, model))


results_df = pd.DataFrame(results)
results_df.set_index('Model', inplace=True)

plt.figure(figsize=(12, 6))
results_df.plot(kind='bar', rot=45, colormap='viridis')
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

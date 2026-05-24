import pandas as pd
import missingno as msno

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# ML Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('credit_risk_analysis_dataset.csv')

# Exploring the dataset
def explore_data(data):
    print('-' * 50)
    print("First 5 rows of the dataset:")
    print('-' * 50)
    print(data.head())
    print('-' * 50)
    print("Dataset information:")
    print('-' * 50)
    print(data.info())
    print('-' * 50)
    print("Summary statistics:")
    print('-' * 50)
    print(data.describe())
    print('-' * 50)
    print("Missing values in each column:")
    print('-' * 50)
    print(data.isnull().sum())
    print('-' * 50)

# Visualize missing values
def visualize_missing_values(data):    
    msno.matrix(data)
    plt.savefig('missing_values_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualise the distribution of risk levels among the customers
def visualize_risk_levels(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Default_Risk', data=data)
    plt.title('Distribution of Risk Levels')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.savefig('risk_level_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualise distribution of variables
def visualize_distribution(data):
    data.hist(figsize=(12, 10))
    plt.savefig('variable_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Correlation Heatmap
def visualize_correlation(data):
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# Risk vs features
def visualize_risk_vs_features(data):
    features = ['Age', 'Annual_Income', 'Loan_Amount', 'Credit_Score']
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Default_Risk', y=feature, data=data)
        plt.title(f'{feature} vs Default Risk')
        plt.xlabel('Default Risk')
        plt.ylabel(feature)
        plt.savefig(f'{feature}_vs_default_risk.png', dpi=300, bbox_inches='tight')
        plt.show()

# Aspects of features for each risk category
def risk_category_analysis(data):
    risk_categories = data['Default_Risk'].unique()
    aspects = ['Age', 'Annual_Income', 'Loan_Amount', 'Credit_Score']
    for category in risk_categories:
        subset = data[data['Default_Risk'] == category]
        print('-' * 50)
        print(f"Analysis for Risk Category: {category}")
        for aspect in aspects:
            aspect_mean = round(subset[aspect].mean(),0)
            print(f"Average {aspect}:", aspect_mean) 
        print('-' * 50)

# Build ML model to classify customers in risk levels
def classify(data):
    # Preprocessing
    X = data.drop('Default_Risk', axis=1)
    y = data['Default_Risk']
    y = LabelEncoder().fit_transform(y)  # Encode target variable

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline that combines the preprocessor with a classifier
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print('-' * 50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print('-' * 50)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print('-' * 50)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Feature importance
    feature_importances = clf.named_steps['classifier'].feature_importances_


explore_data(data)
visualize_missing_values(data)
visualize_risk_levels(data)
visualize_distribution(data)
visualize_correlation(data)
visualize_risk_vs_features(data)
risk_category_analysis(data)
classify(data)

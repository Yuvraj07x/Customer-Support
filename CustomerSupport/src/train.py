import os
import joblib
import pandas as pd


from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import (
    accuracy_score,
    classification_report
)



# =====================================================
# Folders
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "outputs",
    exist_ok=True
)



# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(
    "data/customer_support_cleaned.csv"
)



print("="*60)
print("DATASET LOADED")
print("="*60)

print("Shape:", df.shape)



# =====================================================
# Features and Target
# =====================================================

X = df["clean_text"]

y = df["intent"]



# =====================================================
# Encode Intent Labels
# =====================================================

encoder = LabelEncoder()


y_encoded = encoder.fit_transform(
    y
)



# =====================================================
# TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(

    max_features=10000,

    ngram_range=(1,2),

    min_df=2
)



X_vectorized = vectorizer.fit_transform(
    X
)



# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_vectorized,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)



print()

print(
    "Training Samples:",
    X_train.shape[0]
)

print(
    "Testing Samples:",
    X_test.shape[0]
)



# =====================================================
# Models
# =====================================================

models = {


"Naive Bayes":
MultinomialNB(),


"Logistic Regression":
LogisticRegression(
    max_iter=2000
),


"Linear SVM":
LinearSVC(),


"Random Forest":
RandomForestClassifier(
    n_estimators=200
)

}



best_model = None

best_name = ""

best_accuracy = 0



results=[]



print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)



for name, model in models.items():


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    results.append({

        "Model":name,

        "Accuracy":round(
            accuracy*100,
            2
        )

    })


    print(
        f"{name:<25}",
        f"{accuracy:.2%}"
    )



    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_name=name




# =====================================================
# Evaluation
# =====================================================


print("\n")
print("="*60)
print("BEST MODEL")
print("="*60)


print(
    "Model:",
    best_name
)


print(
    "Accuracy:",
    f"{best_accuracy:.2%}"
)



predictions = best_model.predict(
    X_test
)



report = classification_report(

    y_test,

    predictions,

    target_names=encoder.classes_

)



print(report)



# =====================================================
# Save Files
# =====================================================

joblib.dump(

    best_model,

    "models/classifier.pkl"

)


joblib.dump(

    vectorizer,

    "models/vectorizer.pkl"

)


joblib.dump(

    encoder,

    "models/label_encoder.pkl"

)



pd.DataFrame(results).to_csv(

    "outputs/model_comparison.csv",

    index=False

)



with open(

    "outputs/classification_report.txt",

    "w"

) as f:

    f.write(report)



print("\nModel saved successfully!")

print("Training completed!")
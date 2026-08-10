import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from wordcloud import WordCloud

# ==========================================================
# Create output folder
# ==========================================================

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/customer_support_cleaned.csv")

# ==========================================================
# Load Saved Objects
# ==========================================================

model = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load the same test data used during training
X_test, y_test = joblib.load("models/test_data.pkl")

# ==========================================================
# Prediction
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(y_test, predictions)

print("=" * 70)
print(f"Accuracy : {accuracy:.2%}")
print("=" * 70)

# ==========================================================
# Classification Report
# ==========================================================

report = classification_report(
    y_test,
    predictions,
    target_names=label_encoder.classes_
)

print(report)

with open(
    "outputs/classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(10, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png")

plt.close()

print("Confusion matrix saved.")

# ==========================================================
# Category Distribution
# ==========================================================

plt.figure(figsize=(10, 6))

df["category"].value_counts().plot(kind="bar")

plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("outputs/category_distribution.png")

plt.close()

print("Category distribution saved.")

# ==========================================================
# Word Cloud
# ==========================================================

text = " ".join(df["clean_text"])

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(text)

plt.figure(figsize=(12, 6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Word Cloud")

plt.tight_layout()

plt.savefig("outputs/wordcloud.png")

plt.close()

print("Word cloud saved.")

# ==========================================================
# Feature Importance
# ==========================================================

if hasattr(model, "coef_"):

    feature_names = vectorizer.get_feature_names_out()

    rows = []

    for class_index, class_name in enumerate(label_encoder.classes_):

        top_indices = model.coef_[class_index].argsort()[-15:]

        for idx in reversed(top_indices):

            rows.append({
                "Category": class_name,
                "Word": feature_names[idx],
                "Weight": model.coef_[class_index][idx]
            })

    feature_df = pd.DataFrame(rows)

    feature_df.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )

    print("Feature importance saved.")

print("\nEvaluation completed successfully!")
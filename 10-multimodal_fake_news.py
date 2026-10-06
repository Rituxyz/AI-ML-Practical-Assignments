import os
import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.sparse import hstack, csr_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATASET_FILE = "dataset.csv"

print("Loading multimodal dataset...")

df = pd.read_csv(DATASET_FILE)

print("\nDataset loaded successfully!")
print("Total samples:", len(df))

print("\nDataset:")
print(df)


# ==========================================
# 2. CHECK DATA
# ==========================================

print("\nChecking labels...")

print(df["label"].value_counts())

print("\nChecking missing values...")
print(df.isnull().sum())


# Remove rows with missing values
df = df.dropna(subset=["text", "image", "label"])


# ==========================================
# 3. CONVERT LABELS
# ==========================================

df["label"] = df["label"].map({
    "Fake": 0,
    "Real": 1
})

print("\nLabels converted:")
print(df["label"].value_counts())


# ==========================================
# 4. IMAGE FEATURE EXTRACTION
# ==========================================

def extract_image_features(image_path):

    try:

        image = Image.open(image_path).convert("RGB")

        # Resize image
        image = image.resize((32, 32))

        # Convert image to numpy array
        image_array = np.array(image) / 255.0

        # Calculate mean RGB values
        mean_rgb = image_array.mean(axis=(0, 1))

        # Calculate standard deviation
        std_rgb = image_array.std(axis=(0, 1))

        # Calculate RGB histograms
        hist_r, _ = np.histogram(
            image_array[:, :, 0],
            bins=16,
            range=(0, 1)
        )

        hist_g, _ = np.histogram(
            image_array[:, :, 1],
            bins=16,
            range=(0, 1)
        )

        hist_b, _ = np.histogram(
            image_array[:, :, 2],
            bins=16,
            range=(0, 1)
        )

        # Normalize histograms
        hist_r = hist_r / hist_r.sum()
        hist_g = hist_g / hist_g.sum()
        hist_b = hist_b / hist_b.sum()

        # Combine all image features
        features = np.concatenate([
            mean_rgb,
            std_rgb,
            hist_r,
            hist_g,
            hist_b
        ])

        return features

    except Exception as error:

        print("Image error:", image_path)
        print(error)

        return None


print("\nExtracting image features...")

image_features = []
valid_indices = []

for index, row in df.iterrows():

    image_path = row["image"]

    if os.path.exists(image_path):

        features = extract_image_features(image_path)

        if features is not None:
            image_features.append(features)
            valid_indices.append(index)

    else:

        print("Image not found:", image_path)


# Keep only valid images
df = df.loc[valid_indices].reset_index(drop=True)

image_features = np.array(image_features)

print("\nImage feature extraction completed.")

print("Number of images:", len(image_features))
print("Image feature size:", image_features.shape)


# ==========================================
# 5. TEXT DATA
# ==========================================

texts = df["text"].values
labels = df["label"].values


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_text_train, X_text_test, X_image_train, X_image_test, y_train, y_test = train_test_split(
    texts,
    image_features,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

print("\nTraining samples:", len(y_train))
print("Testing samples:", len(y_test))


# ==========================================
# 7. TEXT FEATURE EXTRACTION
# ==========================================

print("\nExtracting text features using TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words="english",
    ngram_range=(1, 2)
)

X_text_train = vectorizer.fit_transform(X_text_train)

X_text_test = vectorizer.transform(X_text_test)

print("Text feature extraction completed.")

print("Text feature size:", X_text_train.shape)


# ==========================================
# 8. COMBINE TEXT + IMAGE FEATURES
# ==========================================

print("\nCombining text and image features...")

X_image_train = csr_matrix(X_image_train)
X_image_test = csr_matrix(X_image_test)

X_train = hstack([
    X_text_train,
    X_image_train
])

X_test = hstack([
    X_text_test,
    X_image_test
])

print("Multimodal feature combination completed.")

print("Final training feature shape:", X_train.shape)
print("Final testing feature shape:", X_test.shape)


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000
)

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# ==========================================
# 11. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========================================")
print("MULTIMODAL FAKE NEWS DETECTION RESULTS")
print("========================================")

print("Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake", "Real"],
        zero_division=0
    )
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 14. DISPLAY CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Multimodal Fake News Detection")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()


# ==========================================
# 15. FINAL MESSAGE
# ==========================================

print("\n========================================")
print("MODEL EXECUTION COMPLETED")
print("========================================")

print("Input modalities:")
print("1. Text")
print("2. Image")

print("\nText technique:")
print("TF-IDF")

print("\nImage technique:")
print("RGB statistics + RGB histograms")

print("\nMachine Learning algorithm:")
print("Logistic Regression")

print("\nOutput:")
print("Fake News / Real News")

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =====================================
# SET EXPERIMENT NAME
# =====================================
mlflow.set_experiment("Flipkart_Sentiment_Analysis")

# =====================================
# LOAD DATASET (YOUR CLEANED FILE)
# =====================================
df = pd.read_csv("cleaned_data.csv")

X = df["clean_review"]
y = df["sentiment"]

# =====================================
# HYPERPARAMETER LOOP (🔥 IMPROVED)
# =====================================
C_values = [0.01, 0.1, 1, 10, 50]   # more runs = better plots
max_features_list = [3000, 5000]

for C in C_values:
    for max_features in max_features_list:

        run_name = f"LogReg_C_{C}_TFIDF_{max_features}"

        with mlflow.start_run(run_name=run_name):

            # =============================
            # TF-IDF Vectorizer
            # =============================
            vectorizer = TfidfVectorizer(max_features=max_features)
            X_vec = vectorizer.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_vec, y, test_size=0.2, random_state=42
            )

            # =============================
            # MODEL
            # =============================
            model = LogisticRegression(C=C, max_iter=200)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            # =============================
            # LOG PARAMETERS
            # =============================
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("C", C)
            mlflow.log_param("max_features", max_features)

            # =============================
            # LOG METRICS
            # =============================
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # =============================
            # CONFUSION MATRIX ARTIFACT
            # =============================
            ConfusionMatrixDisplay.from_predictions(y_test, preds)
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.clf()

            # =============================
            # SAVE MODEL LOCALLY (for Streamlit)
            # =============================
            joblib.dump(model, "sentiment_model.pkl")
            joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

            # =============================
            # REGISTER MODEL
            # =============================
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="FlipkartSentimentModel"
            )

            # =============================
            # TAGGING (Assignment Requirement)
            # =============================
            mlflow.set_tag("project", "Flipkart Sentiment Analysis")
            mlflow.set_tag("algorithm", "LogisticRegression")
            mlflow.set_tag("dataset", "cleaned_data.csv")

print("✅ All MLflow runs completed successfully!")

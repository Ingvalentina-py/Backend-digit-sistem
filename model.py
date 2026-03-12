import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MLP_PATH = os.path.join(MODELS_DIR, "mlp.joblib")
SVM_PATH = os.path.join(MODELS_DIR, "svm.joblib")

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "optdigits.tra"), header=None)
    test  = pd.read_csv(os.path.join(DATA_DIR, "optdigits.tes"), header=None)

    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test  = test.iloc[:, :-1].values
    y_test  = test.iloc[:, -1].values
    return X_train, y_train, X_test, y_test

def train_and_save():
    X_train, y_train, X_test, y_test = load_data()

    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    svm_model = svm.SVC(kernel="linear", probability=True)
    svm_model.fit(X_train, y_train)

    dump(mlp, MLP_PATH)
    dump(svm_model, SVM_PATH)

    return evaluate_models(mlp, svm_model, X_test, y_test)

def load_models():
    mlp = load(MLP_PATH)
    svm_model = load(SVM_PATH)
    return mlp, svm_model

def evaluate_models(mlp, svm_model, X_test, y_test):
    pred_mlp = mlp.predict(X_test)
    pred_svm = svm_model.predict(X_test)

    result = {
        "mlp": {
            "accuracy": float(accuracy_score(y_test, pred_mlp)),
            "confusion_matrix": confusion_matrix(y_test, pred_mlp).tolist(),
            "report": classification_report(y_test, pred_mlp, output_dict=True)
        },
        "svm": {
            "accuracy": float(accuracy_score(y_test, pred_svm)),
            "confusion_matrix": confusion_matrix(y_test, pred_svm).tolist(),
            "report": classification_report(y_test, pred_svm, output_dict=True)
        }
    }
    return result

def ensure_models_exist():
    if not (os.path.exists(MLP_PATH) and os.path.exists(SVM_PATH)):
        return train_and_save()
    return None

def predict_from_pixels(model, pixels_64):
    x = np.array(pixels_64, dtype=float).reshape(1, 64)
    pred = int(model.predict(x)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0].tolist()

    return pred, proba
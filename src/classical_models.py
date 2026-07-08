from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_models():
    models = {
        "SVM": SVC(
            kernel="rbf",
            C=1,
            gamma="scale"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        )
    }

    return models
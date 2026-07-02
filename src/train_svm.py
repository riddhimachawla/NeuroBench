from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from mne.decoding import CSP


def train_svm(X, y):

    # Convert labels
    y = y - 7

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pipeline = Pipeline([
        ("csp", CSP(
            n_components=6,
            reg=None,
            log=True,
            norm_trace=False
        )),
        ("svm", SVC(
            kernel="rbf",
            C=1,
            gamma="scale"
        ))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    return pipeline
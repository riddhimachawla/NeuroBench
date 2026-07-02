from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from mne.decoding import CSP


def train_svm(X, y):

    # Convert labels from 7,8,9,10 to 0,1,2,3
    y = y - 7

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # CSP + SVM Pipeline
    model = Pipeline([
        ("csp", CSP(n_components=4)),
        ("svm", SVC(kernel="rbf"))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}\n")

    print(classification_report(y_test, predictions))

    return model
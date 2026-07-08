from sklearn.model_selection import GridSearchCV


class HyperparameterTuner:

    def __init__(self):

        self.svm_params = {

            "C": [0.1, 1, 10, 100],

            "kernel": [

                "linear",

                "rbf"

            ],

            "gamma": [

                "scale",

                "auto"

            ]

        }

    def tune_svm(self, model, X_train, y_train):

        print("\n" + "=" * 60)
        print("Running Grid Search for SVM...")
        print("=" * 60)

        grid = GridSearchCV(

            estimator=model,

            param_grid=self.svm_params,

            cv=5,

            scoring="accuracy",

            n_jobs=-1,

            verbose=2

        )

        grid.fit(

            X_train,

            y_train

        )

        print("\nBest Parameters")

        print(grid.best_params_)

        print("\nBest Cross Validation Accuracy")

        print(f"{grid.best_score_:.4f}")

        return grid.best_estimator_
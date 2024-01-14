'''
Autolog with Mlflow
'''

# Description ---------
# The practice of logging in Machine Learning is essential for the experimentation phase since each hypothesis is accompanied by its evidence of change,
# and therefore, the records of each experiment are crucial for detailing each result.
# Now let's see an example provided by the MLflow blog

## Libraries ---------

import mlflow
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

## Code in Action !!!

#===============================================#
# Generate a 4 class classification problem     #
#===============================================#
X, y = datasets.make_classification(
    n_samples=3000,
    class_sep=0.6,
    random_state=42,
    n_classes=4,
    n_informative=3,
)

#=================================================#
#                 Split data                      #
#=================================================#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#=================================================#
#                 Enable autolog                  #
#=================================================#
mlflow.sklearn.autolog()

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [1,5,10 ,20],
    'max_depth': [1, 5, 7,10,13,15,20],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4]
}

# Start an MLflow run
with mlflow.start_run():
    # Initialize the GridSearchCV object
    clf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5
    )

    # Train the model with cross-validation
    clf.fit(X_train, y_train)

    # The best model found by GridSearchCV
    best_model = clf.best_estimator_

    # Evaluate the best model on the test set
    test_score = best_model.score(X_test, y_test)

    # Get the current run to log additional information
    run = mlflow.active_run()
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # Fetch the run details
    run_details = mlflow.get_run(run_id)

    # Autologger

    print('='*32)
    print('Autologger')

    # Print metrics, parameters, and tags
    print("\nMetrics:")
    print(run_details.data.metrics)

    print("\nParams:")
    print(run_details.data.params)

    print("\nTags:")
    print(run_details.data.tags)

    print('=' * 32)

import config
import model_dispatcher
import joblib
import os
import cv2
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from numpy import mean

from data_preparation import load_input, code_image

def read_data():
    X,y = load_input()
    return X,y

def evaluate(model):
    # read the data
    X, y = read_data()

    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=config.NUM_FOLDS, n_repeats=1, random_state=1)

    # evaluate the model and collect the scores
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    return mean(scores)

def try_all_models():
    names = model_dispatcher.models.keys()
    scores = {}
    for name in names:
        print(f"Evaluating {name}")
        model = model_dispatcher.models[name]
        score = evaluate(model)
        scores[name] = score
    print(scores)

def tuning():
    # read the data
    X, y = read_data()

    # define model
    model = GradientBoostingClassifier(random_state=0)

    # find best parameters
    param_test = {'n_estimators':range(110,151,20)}
    gsearch = GridSearchCV(estimator = model,
                           param_grid = param_test,
                           scoring='accuracy',n_jobs=-1,
                           cv = RepeatedKFold(n_splits=config.NUM_FOLDS, n_repeats=1, random_state=1))
    gsearch.fit(X, y)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

def train(model):
    # read the data
    X, y = read_data()

    # fit the model on the whole dataset
    model.fit(X, y)

    return model

def train_rf():
    model = RandomForestClassifier(random_state=0)
    model = train(model)
    joblib.dump(model, os.path.join(config.models_directory, "rf.bin"))
    return model

def load_rf():
    model = joblib.load(os.path.join(config.models_directory, "rf.bin"))
    return model

def train_gbc():
    model = GradientBoostingClassifier(random_state=0)
    model = train(model)
    joblib.dump(model, os.path.join(config.models_directory, "gbc.bin"))
    return model

def load_gbc():
    model = joblib.load(os.path.join(config.models_directory, "gbc.bin"))
    return model

def create_predictions_list(model):
    predictions = []
    for filename in os.listdir(config.test_images_path):
        img = cv2.imread(config.test_images_path + filename)
        coded_image = code_image(img)
        if len(coded_image) == 0:
            predictions.append(4)
            continue
        prediction = model.predict([coded_image])[0]
        predictions.append(prediction)
    return predictions

def create_csv_submission(model):
    predicted_values = create_predictions_list(model)
    test_images = []
    for _, _, filenames in os.walk(config.test_images_path):
        test_images = filenames
    submission = pd.DataFrame()
    submission['image_id'] = test_images
    submission['class_6'] = predicted_values
    submission.to_csv(config.directory + 'sub.csv', index=False)


if __name__ == "__main__":
    model = load_gbc()
    create_csv_submission(model)

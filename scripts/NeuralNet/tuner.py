
import keras_tuner as kt
from keras_models import  cnn_model, rnn_model, ffnn_model

tuner = kt.Hyperband(
    lambda hp: ffnn_model(hp, inputShape=(301), outputShape=2),
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    directory='hyperparameter_tuning',
    project_name='ffnn'
)

# Get best trial IDs
best_trials = tuner.oracle.get_best_trials(num_trials=1)
best_trial_id = best_trials[0].trial_id
print(f"Best Trial ID: {best_trial_id}")

import json
import os

# Locate the trial folder
trial_folder = os.path.join("./hyperparameter_tuning", "ffnn", f"trial_{best_trial_id}")

# Load the hyperparameters
with open(os.path.join(trial_folder, "trial.json"), "r") as hp_file:
    hp_data = json.load(hp_file)

    # Print the best hyperparameters
    print("Best Hyperparameters:")
    for hp, value in hp_data["hyperparameters"]["values"].items():
        print(f"{hp}: {value}")

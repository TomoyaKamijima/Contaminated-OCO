import os
import numpy as np

# Save the results of the experiment
def save_experiment_results(experiment_name, file_name, **data):
    folder_name = experiment_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, file_name)
    np.savez(file_path, **data)
import logging
import os

logging.getLogger().setLevel(logging.INFO)
model_path = 'test/models'
model_0_name = 'model_0_test'

# Create initial model
logging.info('Step 0: Creating initial model')
os.system(f'python3 0_create_new_model.py --model_path {model_path} --model_name   {model_0_name}')
path_to_model_0 = f'{model_path}/{model_0_name}'
assert os.path.isdir(path_to_model_0)

# Self play
logging.info('Step 1: Self play')
path_to_experience_results = 'test/models/experiences'
number_games = 5
seed = 26
rounds_per_action = 2
max_movements = 1
max_actions_per_game = 2
os.system(
    f'python3 1_self_play.py --path_to_model {path_to_model_0} --path_to_results {path_to_experience_results} --number_games {number_games} '
    f'--rounds_per_action {rounds_per_action} --max_movements {max_movements} --seed {seed} --max_actions_per_game {max_actions_per_game}')

# Train model
model_1_name = 'model_1_test'
lr = 0.01
batch_size = 2
logging.info('Step 2: Training the model')
os.system(
    f'python3 2_train_model.py --path_to_model {path_to_model_0} --path_to_experiences {path_to_experience_results} '
    f' --new_model_directory {model_path} --new_model_name {model_1_name} --lr {lr} --batch_size {batch_size}')

path_to_model_1 = f'{model_path}/{model_1_name}'
assert os.path.isdir(path_to_model_1)

# Select best model
logging.info('Step 3: Select best model')
results_path = 'test/models/results'
os.system(
    f'python3 3_select_best_model.py --path_to_models {path_to_model_0} {path_to_model_1} '
    f'--model_names {model_0_name} {model_1_name} '
    f'--path_to_results {results_path} --seed {seed} --number_games {number_games} --rounds_per_action {rounds_per_action} '
    f'--max_movements {max_movements} --max_actions_per_game {max_actions_per_game}')

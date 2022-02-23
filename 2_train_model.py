import argparse
import glob
import logging
import os

import h5py
import keras.models

from src.agent.alphazero import AlphaZeroAgent
from src.experience.alphazero_experience import load_experience, combine_experience

logging.getLogger().setLevel(logging.INFO)


def get_experience_buffer(path_to_experiences):
    experiences = []
    for hdf5_filename in glob.glob(f'{path_to_experiences}/*.hdf5'):
        with h5py.File(hdf5_filename, 'r') as experience_hdf5:
            experience = load_experience(experience_hdf5)
            experiences.append(experience)
    return combine_experience(experiences)


def self_play(path_to_model, path_to_experiences, new_model_directory, new_model_name, lr, batch_size):
    assert os.path.isdir(new_model_directory)
    assert os.path.isdir(path_to_experiences)

    logging.info('Loading experiences')
    experience_buffer = get_experience_buffer(path_to_experiences)

    logging.info('Loading model ' + path_to_model)
    model = keras.models.load_model(path_to_model)

    logging.info('Starting training')
    alphazero_agent = AlphaZeroAgent(model)
    alphazero_agent.train(experience_buffer, lr, batch_size)

    model.save(f'{new_model_directory}/{new_model_name}')


if __name__ == '__main__':
    # self_play('models/model_0', 'experiences/test', 'models', 'model_1', 0.01, 1)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_model',
        type=str,
        required=True,
        help='Path to the model'
    )
    parser.add_argument(
        '--path_to_experiences',
        type=str,
        required=True,
        help='Path to experiences'
    )
    parser.add_argument(
        '--new_model_directory',
        type=str,
        required=True,
        help='Path to store the new model'
    )
    parser.add_argument(
        '--new_model_name',
        type=str,
        required=True,
        help='Path to store the new model'
    )
    parser.add_argument(
        '--lr',
        type=float,
        required=False,
        default=0.01,
        help='Learning rate for training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=True,
        help='Batch size for training'
    )
    args = parser.parse_args()
    self_play(args.path_to_model, args.path_to_experiences, args.new_model_directory, args.new_model_name, args.lr,
              args.batch_size)

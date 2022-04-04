import os
import pickle
import random
import agent_code.bb_agent.rl as rl
import numpy as np


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if os.path.isfile(rl.MODEL_FILE_NAME):
        with open(rl.MODEL_FILE_NAME, "rb") as file:
            self.beta = pickle.load(file)
    else:
        self.beta = np.zeros((len(rl.ACTIONS), rl.STATE_F_LENGTH))
    self.bomb_memory = []

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    return rl.select_action(self, game_state)

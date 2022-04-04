import os
import pickle
import random
import torch

import numpy as np
import settings as s

#from train import neuralnet, TrainAlg

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

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
    """if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:"""
    self.logger.info("Loading model from saved state.")
    with open("my-saved-model.pt", "rb") as file:
        self.model = pickle.load(file)

    """fake_dict = {"round": 0,
                 "step": 0,
                 "field": np.zeros((s.COLS, s.ROWS)),
                 "self": ("fakename", 0, False, (0, 0)),
                 "others": [("fakename", 0, False, (0, 0)),
                            ("fakename", 0, False, (0, 0)),
                            ("fakename", 0, False, (0, 0))],
                 "bombs": [(0, 0, 0)],
                 "coins": [(0, 0)],
                 "user_input": "WAIT"
                 }
    
    n_input = state_to_features(fake_dict)
    self.model = neuralnet(n_input, 256, 128, 6)"""


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # todo Exploration vs exploitation
    random_prob = .1

    field = game_state["field"]
    [xpos, ypos] = list(game_state["self"][3])

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        p = [.2, .2, .2, .2, .15, .05]
        if field[xpos, ypos - 1] != 0:
            p[0] = 0
        if field[xpos + 1, ypos] != 0:
            p[1] = 0
        if field[xpos, ypos + 1] != 0:
            p[2] = 0
        if field[xpos - 1, ypos] != 0:
            p[3] = 0
        if not game_state["self"][2] != 0:
            p[4] = 0
        p = p / np.sum(p)
        return np.random.choice(ACTIONS, p = p)


    self.logger.debug("Querying model for action.")

    state_tensor = torch.tensor(state_to_features(game_state), dtype=torch.float)
    q_values = self.model(state_tensor)
    min_q = torch.min(q_values).item()

    if field[xpos, ypos - 1] != 0:
        q_values[0] = min_q
    if field[xpos + 1, ypos] != 0:
        q_values[1] = min_q
    if field[xpos, ypos + 1] != 0:
        q_values[2] = min_q
    if field[xpos - 1, ypos] != 0:
        q_values[3] = min_q
    if not game_state["self"][2] != 0:
        q_values[4] = min_q

    action_idx = torch.argmax(q_values).item()
    action = ACTIONS[action_idx]

    self.logger.debug(f"Chose action {action}.")

    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array

    fake_dict = {"round": 0,  #int
                 "step": 0,   #int
                 "field": np.zeros((s.COLS,s.ROWS)), #int s.COLS x s.ROWS
                 "self": ("fakename", 0, False, (0, 0)),  #str, int(score), bool, tuple(int,int)
                 "others": [("fakename", 0, False, (0, 0)),
                     ("fakename", 0, False, (0, 0)),
                     ("fakename", 0, False, (0, 0))],
                 "bombs": [(0,0,0)],
                 "coins": [(0,0)],
                 "user_input": "WAIT"
                 }
    """
    gs = game_state
    own_pos = np.array(gs["self"][3])

    simple_vals = np.array([gs["round"],                           # 0
                                 gs["step"],                            # 1
                                 gs["self"][1],                         # 2
                                 own_pos[0], own_pos[1]])               # 3 - 4

    field_flat = np.reshape(gs["field"], s.ROWS*s.COLS)                 # 5 - 293 (assuming 17x17 field)

    others_vec = np.zeros(12)                                           # 294 - 305
    for i in range(len(gs["others"])):
        others_vec[i * 4] = gs["others"][i][1]
        others_vec[i * 4 + 1] = int(gs["others"][i][2])
        others_vec[i * 4 + 2] = gs["others"][i][3][0] - own_pos[0]
        others_vec[i * 4 + 3] = gs["others"][i][3][1] - own_pos[1]

    bombs_rel = np.zeros(12)                                            # 305 - 316

    if len(gs["bombs"]) >= 1 and gs["step"] > 2 :
        for i in range(len(gs["bombs"])):
            bombs_rel[i * 3] = gs["bombs"][i][0][0] - own_pos[0]
            bombs_rel[i * 3 + 1] = gs["bombs"][i][0][1] - own_pos[1]
            bombs_rel[i * 3 + 2] = gs["bombs"][i][1]

    near_coins_rel = np.zeros(10)  # 317 - 326
    if len(gs["coins"]) >= 1:
        coins = np.array(gs["coins"])
        coin_dist = coins - own_pos[None,:]
        coin_dist = np.sum(np.square(coin_dist), axis = -1)
        min_dist_idx = np.argsort(coin_dist)

        nearest_coins = coins[min_dist_idx]

        for coin in nearest_coins:
            near_coins_rel[i * 2] = coin[0] - own_pos[0]
            near_coins_rel[i * 2 + 1] = coin[1] - own_pos[1]

    feature_vector = np.append(simple_vals, field_flat)
    feature_vector = np.append(feature_vector, others_vec)
    feature_vector = np.append(feature_vector, bombs_rel)
    feature_vector = np.append(feature_vector, near_coins_rel)

    return feature_vector.astype(float)

    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
    """

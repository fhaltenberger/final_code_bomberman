from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np

import events as e
import settings as s
from .callbacks import state_to_features

import torch

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
#BOMB_DANGER = ["","BOMB1","BOMB2","BOMB3"]  #strings for bomb in distance x explodes in y timesteps

BOMB_DANGER = [["", ""      , ""      , ""      ],  #strings for bomb in distance x explodes in y timesteps
               ["", "BOMB11", "BOMB12", "BOMB13"],
               ["", "BOMB21", "BOMB22", "BOMB23"],
               ["", "BOMB31", "BOMB32", "BOMB33"]]

# Available actions
possible_actions = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
action_idx = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

class neuralnet(torch.nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super().__init__()
        self.conn1 = torch.nn.Linear(n_input, n_hidden1)
        self.conn2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.conn3 = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conn1(x))
        x = torch.nn.functional.relu(self.conn2(x))
        x =self.conn3(x)

        return x

class TrainAlg:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimization = torch.optim.Adam(model.parameters(), lr = self.lr)
        self.lossfct = torch.nn.MSELoss()

    def train(self, old_state, action, new_state, reward, game_over):
        old_state = torch.unsqueeze(torch.tensor(old_state, dtype=torch.float), 0)
        new_state = torch.unsqueeze(torch.tensor(new_state, dtype=torch.float), 0)
        #reward = torch.unsqueeze(torch.tensor(reward, dtype=torch.float), 0)

        predicted_q = self.model(old_state)

        td_q_values = predicted_q
        td_q_values[0][action_idx[action]] = reward
        if not game_over:
            td_q_values[0][action_idx[action]] += torch.max(self.model(new_state))

        self.optimization.zero_grad()
        loss = self.lossfct(td_q_values, predicted_q)
        loss.backward()

        self.optimization.step()


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    with open("training_data.pt","rb") as td:
        training_data = pickle.load(td)
    
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions_archive = training_data
    self.transitions = []

    # setup model, need fake state dict to determine size of feature vector for NN input size
    fake_dict = {"round": 0, 
                 "step": 0, 
                 "field": np.zeros((s.COLS,s.ROWS)),
                 "self": ("fakename", 0, False, (0, 0)), 
                 "others": [("fakename", 0, False, (0, 0)),
                     ("fakename", 0, False, (0, 0)),
                     ("fakename", 0, False, (0, 0))], 
                 "bombs": [(0,0,0)],
                 "coins": [(0,0)],
                 "user_input": "WAIT"
                 }
    self.n_features = np.size(state_to_features(fake_dict))
    self.lr = 0.5
    self.gamma = 0.5
    #self.model = neuralnet(self.n_features, 256, 128, len(possible_actions))
    with open("my-saved-model.pt", "rb") as trained_network:
        self.model = pickle.load(trained_network)
    self.trainer = TrainAlg(self.model, self.lr, self.gamma)




def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards


    # simplify state parameters
    round_new = new_game_state["round"]
    step_new = new_game_state["step"]
    field_new = new_game_state["field"]
    bombs_new = new_game_state["bombs"]
    coins_new = new_game_state["coins"]
    others_new = new_game_state["others"]    
    self_new = new_game_state["self"]
    expmap_new = new_game_state["explosion_map"]

    score_new = self_new[1]
    bombs_left_new = self_new[2]
    pos_new = self_new[3]
    
    # only set old_state if there is one
    if step_new > 1:
        round_old = old_game_state["round"]
        step_old = old_game_state["step"]
        field_old = old_game_state["field"]
        bombs_old = old_game_state["bombs"]
        coins_old = old_game_state["coins"]
        others_old = old_game_state["others"]    
        self_old = old_game_state["self"]
        expmap_old = old_game_state["explosion_map"]

        score_old = self_old[1]
        bombs_left_old = self_old[2]
        pos_old = self_old[3]    

    # check if agent is in danger zone of a bomb:
    """if expmap_new[pos_new[0], pos_new[1]] != 0:
        events.append(BOMB_DANGER[int(expmap_new[pos_new[0], pos_new[1]])])"""

    # rewards for being in danger zone and reducing/increasing distance to bomb
    for bomb in bombs_new:
        [bombx, bomby] = list(bomb[0])
        bomb_time = bomb[1]
        if bombx == pos_new[0] and abs(bomby - pos_new[1]) <= 3:
            bomb_dist = bomby - pos_new[1]
            events.append(BOMB_DANGER[abs(bomb_dist)][bomb_time])
            if step_new > 1:
                old_bomb_dist = bomby - pos_old[1]
                if abs(bomb_dist) < abs(old_bomb_dist):
                    events.append("CLOSER_TO_BOMB")
                elif abs(bomb_dist) > abs(old_bomb_dist):
                    events.append("FURTHER_FROM_BOMB")
        elif bomby == pos_new[1] and abs(bombx - pos_new[0]) <= 3:
            bomb_dist = bombx - pos_new[0]
            events.append(BOMB_DANGER[abs(bomb_dist)][bomb_time])
            if step_new > 1:
                old_bomb_dist = bombx - pos_old[0]
                if abs(bomb_dist) < abs(old_bomb_dist):
                    events.append("CLOSER_TO_BOMB")
                elif abs(bomb_dist) > abs(old_bomb_dist):
                    events.append("FURTHER_FROM_BOMB")

    # reward for escaping danger zone
    if step_new > 1:
        if expmap_old[pos_old[0], pos_old[1]] != 0 and expmap_new[pos_new[0], pos_new[1]] == 0:
            events.append("FLED_BOMB")

    # reward for getting closer / further to nearest coin
    if step_new > 1 and len(coins_old) > 0 and len(coins_new) > 0:
        coin_distances_old = np.zeros(len(coins_old))
        for i, coin in enumerate(coins_old):
            coin_distances_old[i] = (pos_old[0] - coin[0])**2 + (pos_old[1] - coin[1])**2
        nearest_coin_old = coins_old[np.argmin(coin_distances_old)]
        near_coin_dist_old = np.linalg.norm(np.array(nearest_coin_old) - np.array(pos_old))

        coin_distances_new = np.zeros(len(coins_new))
        for i, coin in enumerate(coins_new):
            coin_distances_new[i] = (pos_new[0] - coin[0]) ** 2 + (pos_new[1] - coin[1]) ** 2
        nearest_coin_new = coins_new[np.argmin(coin_distances_new)]
        near_coin_dist_new = np.linalg.norm(np.array(nearest_coin_new) - np.array(pos_new))

        if near_coin_dist_new < near_coin_dist_old:
            events.append("CLOSER_TO_COIN")
        if near_coin_dist_new > near_coin_dist_old and not e.COIN_COLLECTED in events:
            events.append("FURTHER_FROM_COIN")




    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    self.trainer.train(state_to_features(old_game_state), self_action,
                       state_to_features(new_game_state), reward_from_events(self, events),
                       ("GOT_KILLED" in events))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    print(f"\n agent died in step {last_game_state['step']}")
    print(f"score: {last_game_state['self'][1]}")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        BOMB_DANGER[1][1]: -1.8,
        BOMB_DANGER[1][2]: -1,
        BOMB_DANGER[1][3]: -0.4,
        BOMB_DANGER[2][1]: -1.4,
        BOMB_DANGER[2][2]: -0.8,
        BOMB_DANGER[2][3]: -0.4,
        BOMB_DANGER[3][1]: -1.0,
        BOMB_DANGER[3][2]: -0.4,
        BOMB_DANGER[3][3]: -0.2,
        "FLED_BOMB": 1,
        "CLOSER_TO_BOMB": -1.3,
        "FURTHER_FROM_BOMB": 1,
        e.CRATE_DESTROYED: 0.4,
        "CLOSER_TO_COIN": .3,
        "FURTHER_FROM_COIN": -.3
        }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

import pickle
from typing import List
from collections import deque

import events as e
import settings as s
import numpy as np
import shutil
import os
import agent_code.bb_agent.rl as rl


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if os.path.isfile(rl.XP_BUFFER_FILE_NAME):
        with open(rl.XP_BUFFER_FILE_NAME, "rb") as file:
            self.experience_buffer = pickle.load(file)
    else:
        self.experience_buffer = deque(maxlen=rl.EXPERIENCE_BUFFER_EPISODES)
    print(f'xp buffer: {len(self.experience_buffer)}')
    print(f'saving every {rl.SAVE_INTERVAL} rounds')
    self.current_episode_buffer = []
    self.tau = 1
    self.old_beta = self.beta
    self.stepsdone = 0
    self.maxsteps = 0
    self.maxscore = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    events = events + rl.detect_custom_events(self, old_game_state, new_game_state, events)
    reward = rl.reward_from_events(events)
    self.current_episode_buffer.append((
        rl.extract_features(self, old_game_state),
        self_action,
        rl.extract_features(self, new_game_state),
        reward,
        ))


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
    #compute y
    y = rl.y_for_episode(self.current_episode_buffer, self.beta)
    score = last_game_state['self'][1]
    self.experience_buffer.append((self.current_episode_buffer, y, score))
    
    #select batch
    current_buf_len = len(self.experience_buffer)
    mask = np.arange(current_buf_len)
    mask = np.random.choice(mask, size=min(rl.BATCH_SIZE, current_buf_len))
    mask = np.sort(mask)
    batch = []
    for i in mask:
        batch.append(self.experience_buffer[i])
    
    #compute new beta matrix
    self.beta = rl.update_beta(batch, self.beta)
    
    #save xp buffer and model and back up old one
    if self.tau % rl.SAVE_INTERVAL == 0:
        if os.path.isfile(rl.MODEL_FILE_NAME):
            shutil.copyfile(rl.MODEL_FILE_NAME, rl.MODEL_FILE_NAME+'.BAC')
        with open(rl.MODEL_FILE_NAME, "wb") as file:
            pickle.dump(self.beta, file)
        
        if os.path.isfile(rl.XP_BUFFER_FILE_NAME):
            shutil.copyfile(rl.XP_BUFFER_FILE_NAME, rl.XP_BUFFER_FILE_NAME+'.BAC')
        with open(rl.XP_BUFFER_FILE_NAME, "wb") as file:
            pickle.dump(self.experience_buffer, file)

    #measure convergence
    self.stepsdone += last_game_state['step']
    self.maxsteps = max(self.maxsteps, last_game_state['step'])
    self.maxscore = max(self.maxscore, last_game_state['self'][1])
    if self.tau % rl.INFO_INTERVAL == 0:
        change = np.average(np.abs(self.old_beta - self.beta))
        avg = np.average(np.abs(self.beta))
        self.old_beta = self.beta
        print(f'beta change: {np.round(100*change/avg, 3)} %')
        print(f'beta avg.: {np.round(avg, 3)}')
        print(f'avg. steps: {self.stepsdone / rl.INFO_INTERVAL}')
        print(f'max. steps: {self.maxsteps}')
        print(f'high score: {self.maxscore}')
        self.maxsteps = 0
        self.stepsdone = 0
        self.maxscore = 0
    
    self.current_episode_buffer = []
    self.tau += 1
import numpy as np
import events as e
import settings as s


STATE_F_LENGTH = 16
GAMMA = 0.8
ALPHA = 0.07
EXPERIENCE_BUFFER_EPISODES = 1000
BATCH_SIZE = 500
SOFTMAX_TEMP_TRAIN = 2
SOFTMAX_TEMP_PLAY = 1.5
EPSILON_TRAIN = 0.15
EPSILON_PLAY = 0.0
EPSILON_INVALID = 0.03
CLIP_MIN = 0.0000001
SCORE_WEIGHT = 0

#custom events
E_IN_BOMB_RANGE = "IN_BOMB_RANGE"   # ended a turn in range of a bomb
E_TOOK_COVER = "TOOK_COVER"         # took cover from a bomb before it exploded
E_LONG_GAME = "LONG_GAME"
E_CRATE_DESTROYED_SAFELY = "CRATE_DESTROYED_SAFELY"
E_COIN_FOUND_SAFELY = "COIN_FOUND_SAFELY"
E_BOMB_DROPPED_AT_START = "BOMB_DROPPED_AT_START"
E_CLOSER_TO_COIN = "CLOSER_TO_COIN"
E_CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
E_FARTHER_FROM_COIN = "FARTHER_FROM_COIN"
E_FARTHER_FROM_CRATE = "FARTHER_FROM_CRATE"
E_FARTHER_FROM_BOMB = "FARTHER_FROM_BOMB"   # distance to a bomb has increased
E_CLOSER_TO_BOMB = "CLOSER_TO_BOMB"
E_BOMB_NEXT_TO_CRATE = "BOMB_NEXT_TO_CRATE"
E_BOMB_NEXT_TO_ENEMY = "BOMB_NEXT_TO_ENEMY"
E_ENEMY_IN_BOMB_RANGE = "ENEMY_IN_BOMB_RANGE"

REWARDS = {
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -8,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 7,
        e.COIN_FOUND: 7,
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 200,
        e.KILLED_SELF: -150,
        e.GOT_KILLED: -150,
        e.OPPONENT_ELIMINATED: 20,
        e.SURVIVED_ROUND: 0,
        E_IN_BOMB_RANGE: -50,
        E_TOOK_COVER: 30,
        E_LONG_GAME: -50,
        E_CRATE_DESTROYED_SAFELY: 100,
        E_COIN_FOUND_SAFELY: 30,
        E_BOMB_DROPPED_AT_START: -440,
        E_CLOSER_TO_COIN: 10,
        E_CLOSER_TO_CRATE: 1,
        E_FARTHER_FROM_COIN: -13,
        E_FARTHER_FROM_CRATE: -1,
        E_FARTHER_FROM_BOMB: 30,
        E_CLOSER_TO_BOMB: -30,
        E_BOMB_NEXT_TO_CRATE: 100,
        E_BOMB_NEXT_TO_ENEMY: 250,
        E_ENEMY_IN_BOMB_RANGE: 0,
}

MODEL_FILE_NAME = "model.pt"
XP_BUFFER_FILE_NAME = "experience-buffer.pt"
SAVE_INTERVAL = 200
INFO_INTERVAL = 100
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def extract_features(self, state):
    # features: 10^14 states
        # wall above?
        # wall below?
        # wall left?
        # wall right?
        # map border?
        # vec to closest agent
        # vec to 2 closest bombs
        # number of bombs present (0, 1, >1)
        # vec to nearest crate
        # vec to nearest coin
        
    res = np.zeros(STATE_F_LENGTH)
    own_pos = state['self'][3]

    #wall adjacency
    ret = get_adjacent_walls(state)
    res[0] = ret[0]
    res[1] = ret[1]
    res[2] = ret[2]
    res[3] = ret[3]

    #map border
    if (1 in own_pos) or (s.COLS-1 == own_pos[0]) or (s.ROWS-1 == own_pos[1]):
        res[4] = 1

    #nearest agent
    enemy_count = len(state['others'])
    if enemy_count > 0:
        dist_vecs = np.zeros((enemy_count, 2))
        for i in range(enemy_count):
            dist_vecs[i,0] = state['others'][i][3][0] - own_pos[0]
            dist_vecs[i,1] = state['others'][i][3][1] - own_pos[1]
        closest = np.argmin(np.linalg.norm(dist_vecs, axis=1))
        res[5] = dist_vecs[closest, 0]
        res[6] = dist_vecs[closest, 1]

    #bombs
    ret = two_closest_bombs_dist(self, state)
    res[7] = ret[0]
    res[8] = ret[1]
    res[9] = ret[2]
    res[10] = ret[3]
    res[11] = ret[4]

    #nearest crate
    ret = closest_crate_dist(state)
    res[12] = ret[0]
    res[13] = ret[1]

    #nearest coin
    ret = closest_coin_dist(state)
    res[14] = ret[0]
    res[15] = ret[1]
    
    return res


def get_adjacent_walls(state):
    """
    return: [above, below, left, right]
    """
    own_pos = state['self'][3]
    res = np.zeros(4)

    if state['field'][own_pos[0], own_pos[1]-1] == -1:
        res[0] = 1
    if state['field'][own_pos[0], own_pos[1]+1] == -1:
        res[1] = 1
    if state['field'][own_pos[0]-1, own_pos[1]] == -1:
        res[2] = 1
    if state['field'][own_pos[0]+1, own_pos[1]] == -1:
        res[3] = 1

    return res


def extended_bomb_list(self, state):
    exploded_bombs = []
    i = 0
    while i < len(self.bomb_memory):
        if state['step'] >= self.bomb_memory[i][1]:
            del self.bomb_memory[i]
            continue
        if self.bomb_memory[i][1] != s.EXPLOSION_TIMER + state['step']:
            exploded_bombs.append(self.bomb_memory[i])
        i += 1
    for b in state['bombs']:
        if b[1] == 0:
            add_bomb = True
            for mb in self.bomb_memory:
                if mb[0] == b[0]:
                    add_bomb = False
                    break
            if add_bomb:
                self.bomb_memory.append( [(b[0][0], b[0][1]), s.EXPLOSION_TIMER + state['step'] + 1] )
    return state['bombs'] + exploded_bombs


def is_in_bomb_range(self, state, pos=None):
    if pos == None:
        pos = state['self'][3]
    bombs = extended_bomb_list(self, state)
    bomb_count = len(bombs)
    for i in range(bomb_count):
        x_dist = abs(bombs[i][0][0] - pos[0])
        y_dist = abs(bombs[i][0][1] - pos[1])
        if (y_dist == 0 or x_dist == 0) and (max(x_dist, y_dist) <= s.BOMB_POWER):
            return True
    return False


def closest_coin_dist(state):
    own_pos = state['self'][3]
    coin_count = len(state['coins'])
    res = np.zeros(2)
    if coin_count > 0:
        dist_vecs = np.zeros((coin_count, 2))
        for i in range(coin_count):
            dist_vecs[i,0] = state['coins'][i][0] - own_pos[0]
            dist_vecs[i,1] = state['coins'][i][1] - own_pos[1]
        closest = np.argmin(np.linalg.norm(dist_vecs, axis=1))
        res[0] = dist_vecs[closest, 0]
        res[1] = dist_vecs[closest, 1]
    return res


def closest_crate_dist(state):
    # TODO: vectorize!
    own_pos = state['self'][3]
    dist_vecs = []
    res = np.zeros(2)
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if state['field'][x,y] == 1:
                dist_vecs.append([x-own_pos[0], y-own_pos[1]])
    if len(dist_vecs) > 0:
        dist_vecs = np.array(dist_vecs)
        closest = np.argmin(np.linalg.norm(dist_vecs, axis=1))
        res[0] = dist_vecs[closest, 0]
        res[1] = dist_vecs[closest, 1]
    return res


def two_closest_bombs_dist(self, state):
    bombs = extended_bomb_list(self, state)
    res = np.zeros(5)
    own_pos = state['self'][3]
    bomb_count = len(bombs)
    if bomb_count > 0:
        dist_vecs = np.zeros((bomb_count, 2))
        for i in range(bomb_count):
            dist_vecs[i,0] = bombs[i][0][0] - own_pos[0]
            dist_vecs[i,1] = bombs[i][0][1] - own_pos[1]
        indices = np.argsort(np.linalg.norm(dist_vecs, axis=1))
        if bomb_count == 1:
            res[0] = dist_vecs[indices[0], 0]
            res[1] = dist_vecs[indices[0], 1]
            res[4] = 1
        if bomb_count > 1:
            res[2] = dist_vecs[indices[1], 0]
            res[3] = dist_vecs[indices[1], 1]
            res[4] = 2
    return res


def detect_custom_events(self, old_state, new_state, events):
    res = []
    own_pos = new_state['self'][3]

    #in_bomb_range
    in_range_now = is_in_bomb_range(self, new_state)
    if in_range_now:
        res.append(E_IN_BOMB_RANGE)
    
    #took_cover
    if is_in_bomb_range(self, old_state) and (not in_range_now):
        res.append(E_TOOK_COVER)

    #long game
    if new_state['step'] > 100:
        res.append(E_LONG_GAME)

    #crate destroyed safely
    if e.CRATE_DESTROYED in events and (not e.KILLED_SELF in events):
        res.append(E_CRATE_DESTROYED_SAFELY)

    #coin found safely
    if e.COIN_FOUND in events and (not e.KILLED_SELF in events):
        res.append(E_COIN_FOUND_SAFELY)

    #BOMB_DROPPED_AT_START
    in_starting_pos = own_pos == (1,1) or own_pos == (1,s.ROWS-2) or own_pos == (s.COLS-2,1) or own_pos == (s.COLS-2, s.ROWS-2)
    if e.BOMB_DROPPED in events and in_starting_pos:
        res.append(E_BOMB_DROPPED_AT_START)

    coin_dist_old = np.linalg.norm(closest_coin_dist(old_state))
    coin_dist_new = np.linalg.norm(closest_coin_dist(new_state))
    if coin_dist_new != 0 and coin_dist_old != 0:

        #CLOSER_TO_COIN
        if coin_dist_new < coin_dist_old:
            res.append(E_CLOSER_TO_COIN)

        #FARTHER_FROM_COIN
        else:
            res.append(E_FARTHER_FROM_COIN)

    crate_dist_old = np.linalg.norm(closest_crate_dist(old_state))
    crate_dist_new = np.linalg.norm(closest_crate_dist(new_state))
    if crate_dist_new != 0 and crate_dist_old != 0:

        #CLOSER_TO_CRATE
        if crate_dist_new < crate_dist_old:
            res.append(E_CLOSER_TO_CRATE)

        #CLOSER_TO_CRATE
        else:
            res.append(E_FARTHER_FROM_CRATE)

    #closer/farther from bomb
    bomb_dist_old = two_closest_bombs_dist(self, old_state)
    bomb_dist_new = two_closest_bombs_dist(self, new_state)
    if bomb_dist_old[4] > 0 and bomb_dist_new[4] > 0:
        old_dist = np.linalg.norm(bomb_dist_old[:2])
        new_dist = np.linalg.norm(bomb_dist_new[:2])
        if old_dist < new_dist:
            res.append(E_FARTHER_FROM_BOMB)
        elif old_dist > new_dist:
            res.append(E_CLOSER_TO_BOMB)
        if bomb_dist_old[4] == 2 and bomb_dist_new[4] == 2:
            old_dist = np.linalg.norm(bomb_dist_old[2:4])
            new_dist = np.linalg.norm(bomb_dist_new[2:4])
            if old_dist < new_dist:
                res.append(E_FARTHER_FROM_BOMB)
            elif old_dist > new_dist:
                res.append(E_CLOSER_TO_BOMB)

    if e.BOMB_DROPPED in events:
        for b in new_state['bombs']:
            if b[0] == new_state['self'][3]:
                x = b[0][0]
                y = b[0][1]
        enemy_pos = []
        for o in new_state['others']:
            enemy_pos.append(o[3])
        for r in range(s.BOMB_POWER+1):
            for (xx,yy) in ((x-r,y),(x+r,y),(x,y-r),(x,y+r)):
                xx = min(max(xx,0), s.COLS-1)
                yy = min(max(yy,0), s.ROWS-1)
                
                #BOMB_NEXT_TO_CRATE
                if new_state['field'][xx,yy] == 1:
                    res.append(E_BOMB_NEXT_TO_CRATE)

                #BOMB_NEXT_TO_ENEMY
                if (xx,yy) in enemy_pos:
                    res.append(E_BOMB_NEXT_TO_ENEMY)

    #ENEMY_IN_BOMB_RANGE
    for o in new_state['others']:
        if is_in_bomb_range(self, new_state, o[3]):
            res.append(E_ENEMY_IN_BOMB_RANGE)

    return res


def reward_from_events(events):
    res = 0
    for ev in events:
        if ev in REWARDS:
            res += REWARDS[ev]
    return res


def q_function(state_f, action, beta):
    return np.dot(state_f, beta[ ACTIONS.index(action) ].T)


def y_for_episode(episode_buffer, beta):
    T = len(episode_buffer)
    y = np.zeros(T)
    # for t in range(T):
    #     t_prime = t + 1
    #     while t_prime < T:
    #         y[t] += GAMMA**(t_prime - t - 1) * episode_buffer[t_prime][3]
    #         t_prime += 1
    for t in range(T-1):
        maxq = 0
        for a in ACTIONS:
            maxq = max(maxq, q_function(episode_buffer[t+1][0], a, beta) )
        y[t] = episode_buffer[t][3] + GAMMA * maxq
    return y


def update_beta(batch, old_beta):
    new_beta = np.zeros(np.shape(old_beta))

    #sort batch by action
    action_batches = {}
    for a in ACTIONS:
        action_batches[a] = []
    for episode in batch:
        for t in range(len(episode[0])):
            action = episode[0][t][1]
            state_f = episode[0][t][0]
            y = episode[1][t]
            # episodes with a higher official score have more influence
            weight = 1 + episode[2] * SCORE_WEIGHT
            # save x, y and weight for each timestep sorted by action
            action_batches[action].append(np.concatenate((state_f, (y, weight))))

    #update each action
    for a, action in enumerate(ACTIONS):
        sub_batch = action_batches[action]
        sb_len = len(sub_batch)
        if sb_len > 0:
            sbnp = np.array(sub_batch)
            summ = np.sum(sbnp[:, :-2].T * (sbnp[:, -2] - np.dot(sbnp[:, :-2], old_beta[a,:])), axis=1)
            new_beta[a,:] = old_beta[a,:] + (ALPHA * weight / sb_len) * summ
    
    return new_beta


def select_action(self, state):
    ret = np.zeros(len(ACTIONS))
    for a, action in enumerate(ACTIONS):
        ret[a] = q_function(extract_features(self, state), action, self.beta)
    
    if self.train:
        return epsilon_greedy_select(state, ret, True)
    else:
        return epsilon_greedy_select(state, ret, False)


def softmax_select(state, rewards, train):
    if train:
        mult = 1 / SOFTMAX_TEMP_TRAIN
    else:
        mult = 1 / SOFTMAX_TEMP_PLAY
    distrib = np.exp(rewards*mult) / np.clip(np.sum(np.exp(rewards*mult)), a_min=CLIP_MIN, a_max=None)
    return np.random.choice(ACTIONS, p=distrib)


def epsilon_greedy_select(state, rewards, train):
    if train:
        eps = EPSILON_TRAIN
    else:
        eps = EPSILON_PLAY
    roll = np.random.rand()
    if roll < eps:
        if roll < EPSILON_INVALID:
            # select random action
            return np.random.choice(ACTIONS)
        else:
            # select random valid action with higher probability
            valid = get_valid_actions(state)
            return np.random.choice(valid)
    else:
        return ACTIONS[np.argmax(rewards)]


def get_valid_actions(state):
    valid_actions = ['WAIT']

    walls = get_adjacent_walls(state)
    if walls[0] == 0:
        valid_actions.append('UP')
    if walls[1] == 0:
        valid_actions.append('DOWN')
    if walls[2] == 0:
        valid_actions.append('LEFT')
    if walls[3] == 0:
        valid_actions.append('RIGHT')
    if state['self'][2]:
        valid_actions.append('BOMB')

    return valid_actions

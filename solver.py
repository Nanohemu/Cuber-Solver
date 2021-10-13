import collections
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from env import *

doc = '''
    We start training with the simplest situation (1 step before solved),
    then train more complex situation (n <- n + 1 until n == 20 steps before solved),
    with model of last step as initial Q_net and target_Q_net.
    Each training step has different hyper-parameters (each line of TRAIN_PARA).
'''

TRAIN_NUM = '_006'
GAMMA = 0.9
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000
MODEL_ROOT = './models/'
INIT_MODEL = MODEL_ROOT + 'best_3_005.dat'
EPOCH_LIMIT = 1e5
EPSILON_DECAY_LAST_FRAME = 1e5
DEV = 'cuda'
TRAIN_START = 4
TEST = False

TRAIN_PARA = [
    # EPOCH_LIMIT = 2e5
    # EPSILON_DECAY_LAST_FRAME = 1e5
    # [rand, max_step, exp_reward, mean_space, epsilon_start, epsilon_final]
    [1, 10, -1.5, 10, 1.0, 0.1],
    [2, 15, -2, 15, 0.9, 0.1],
    [3, 20, -2.5, 20, 0.9, 0.09],

    [4, 25, -3.5, 25, 0.9, 0.09],
    [5, 30, -4.5, 20, 0.9, 0.08],
    [6, 30, -5.5, 25, 0.8, 0.08],
    [7, 35, -7, 25, 0.8, 0.07],
    [8, 35, -8, 30, 0.8, 0.07],
    [9, 40, -9, 30, 0.8, 0.06],
    [10, 40, -10.5, 35, 0.8, 0.06],
]

TRAIN_PARA_005 = [
    # EPOCH_LIMIT = 2e5
    # EPSILON_DECAY_LAST_FRAME = 1e5
    # [rand, max_step, exp_reward, mean_space, new_map_epoch, epsilon_start, epsilon_final]
    [1, 10, -2, 10, 1, 1.0, 0.1],
    [2, 15, -3, 15, 2, 0.9, 0.1],

    [3, 20, -4, 15, 3, 0.9, 0.09],
    [4, 20, -5, 20, 3, 0.9, 0.09],
    [5, 25, -7, 20, 3, 0.9, 0.08],
    [6, 25, -8, 25, 3, 0.8, 0.08],
    [7, 30, -9, 25, 3, 0.8, 0.07],
    [8, 30, -10, 30, 3, 0.8, 0.07],
    [9, 35, -11, 30, 3, 0.8, 0.06],
    [10, 35, -12, 35, 3, 0.8, 0.06],
]

TRAIN_PARA_004 = [
    # EPOCH_LIMIT = 2e5
    # EPSILON_DECAY_LAST_FRAME = 1e5
    # [rand, max_step, exp_reward, mean_space, new_map_epoch, epsilon_start, epsilon_final]
    [1, 10, -2, 10, 1, 1.0, 0.15],
    [2, 15, -3, 15, 2, 0.9, 0.15],

    [3, 20, -4, 15, 3, 0.9, 0.15],
    [4, 20, -6, 20, 3, 0.9, 0.15],
    [5, 25, -8, 20, 3, 0.9, 0.15],
    [6, 35, -10, 25, 3, 0.8, 0.15],
    [7, 30, -12, 25, 3, 0.8, 0.15],
    [8, 30, -15, 30, 3, 0.8, 0.15],
    [9, 35, -18, 30, 3, 0.8, 0.15],
    [10, 35, -21, 35, 3, 0.8, 0.15],
]

TRAIN_PARA_003 = [
    # EPOCH_LIMIT = 2e5
    # EPSILON_DECAY_LAST_FRAME = 2e4
    # [rand, max_step, exp_reward, mean_space, new_map_epoch, epsilon_start, epsilon_final]
    [1, 10, -2, 10, 1, 1.0, 0.02],
    [2, 15, -3, 15, 2, 0.9, 0.02],
    [3, 20, -4, 15, 3, 0.9, 0.02],
    [4, 20, -6, 20, 3, 0.9, 0.02],
    [5, 25, -8, 20, 3, 0.9, 0.02],
    [6, 35, -10, 25, 3, 0.8, 0.02],
    [7, 30, -12, 25, 3, 0.8, 0.02],
    [8, 30, -15, 30, 3, 0.8, 0.02],
    [9, 35, -18, 30, 3, 0.8, 0.02],
    [10, 35, -21, 35, 3, 0.8, 0.02],
]

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.bool), np.array(next_states)


class Solver(CuberEnv):
    def __init__(self, exp_buffer, map=None, rand=20, max_step=200):
        if map:
            super().__init__(map=map, max_step=max_step)
        else:
            super().__init__(rand=rand, max_step=max_step)
        self.exp_buffer = exp_buffer
        self.total_reward = 0.0
        self.state = make_img(self.map)

    def play_step(self, net, epsilon=0.0, device="cpu", test=False):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.sample()
        else:
            state_v = torch.tensor([self.state]).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_map, reward, done, solved = self.step(action)
        self.total_reward += reward
        new_state = make_img(new_map)
        exp = Experience(self.state, action, reward, solved, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if done:
            done_reward = self.total_reward
            if test:
                pass
                # print(self.action_his)
            else:
                self.reset()
                self.total_reward = 0.0
                self.state = make_img(self.map)
        return done_reward, solved


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    # done_mask = torch.ByteTensor(dones).to(device)
    done_mask = torch.tensor(dones).bool().to(device)
    actions_v = actions_v.long()

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def train(rand, max_step, exp_reward, mean_space, epsilon_start, epsilon_final, net=None):
    device = torch.device(DEV)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Solver(buffer, rand=rand, max_step=max_step)
    if not net:
        net = DQN(agent.state.shape, agent.nA).to(device)
    elif isinstance(net, str):
        net_dict = torch.load(net)
        net = DQN(agent.state.shape, agent.nA).to(device)
        net.load_state_dict(net_dict)
    net.train().to(device)
    tgt_net = DQN(agent.state.shape, agent.nA).to(device)
    tgt_net.train()
    tgt_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    steps = 0
    best_mean_reward = None
    done_count = 0
    s_file = None
    epoch_steps = 0
    while True:
        steps += 1
        # epsilon = max(epsilon_final, epsilon_start - steps / EPSILON_DECAY_LAST_FRAME)
        epsilon_start *= max(0., 1. - done_count / EPOCH_LIMIT)
        epsilon = max(epsilon_final, epsilon_start * (1. - epoch_steps / max_step))
        epoch_steps += 1

        reward, done = agent.play_step(net, epsilon, device=DEV)
        if reward is not None:
            epoch_steps = 0
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-mean_space:]) if steps >= rand * mean_space else -200
            done_count += done
            print("%d_%d: done %d games, solved %d games, mean reward %.3f, eps %.2f" % (
                rand, steps, len(total_rewards), done_count, mean_reward, epsilon))
            if best_mean_reward is None or best_mean_reward < mean_reward:
                s_file = MODEL_ROOT + "best_" + str(rand) + TRAIN_NUM + ".dat"
                torch.save(net.state_dict(), s_file)
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > exp_reward and done_count >= rand ** 2:
                print("Solved in %d epoch(s)!" % steps)
                break
            # if done_count % new_map_epoch == 0
            if done:
                agent = Solver(buffer, rand=rand, max_step=max_step)
                tgt_net.load_state_dict(net.state_dict())

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if steps % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=DEV)
        loss_t.backward()
        optimizer.step()

        if steps >= EPOCH_LIMIT * (1 + rand / 10):
            s_file = MODEL_ROOT + "final_" + str(rand) + TRAIN_NUM + ".dat"
            torch.save(net.state_dict(), s_file)
            break

    return s_file if s_file is not None else net


def net_test(rand, max_step, net, plot=False, action=False, map=None):
    buffer = ExperienceBuffer(REPLAY_SIZE)
    if map:
        agent = Solver(buffer, map=map, max_step=max_step)
    else:
        agent = Solver(buffer, rand=rand, max_step=max_step)
    if plot:
        agent.render(plot=True)
    if isinstance(net, str):
        net_dict = torch.load(net)
        net = DQN(agent.state.shape, agent.nA)
        net.load_state_dict(net_dict)
    net = net.cpu()
    net.eval()

    step = 0
    reward = None
    done = False
    while step < max_step:
        step += 1
        reward, done = agent.play_step(net, test=True)
        if plot:
            agent.render(plot=True)
        if reward is not None:
            break
    if not done:
        print(agent.state_his[0])
        if action:
            print(agent.action_his)
    # print('steps:%d reward:%d' % (step, reward))
    return done


if __name__ == "__main__":
    if not TEST:
        buffer = ExperienceBuffer(REPLAY_SIZE)
        agent = Solver(buffer)
        net_dict = torch.load(INIT_MODEL)
        device = torch.device(DEV)
        net = DQN(agent.state.shape, agent.nA).to(device)
        tgt_net = DQN(agent.state.shape, agent.nA).to(device)
        net.load_state_dict(net_dict)
        tgt_net.load_state_dict(net_dict)
        p1, p2, p3, p4, p5, p6 = TRAIN_PARA[TRAIN_START - 1]
        net = train(p1, p2, p3, p4, p5, p6, INIT_MODEL)

        for para in TRAIN_PARA[TRAIN_START:]:
            p1, p2, p3, p4, p5, p6 = para
            net = train(p1, p2, p3, p4, p5, p6, net)

    for para in TRAIN_PARA[TRAIN_START - 1:]:
        p1, p2, p3, p4, p5, p6 = para
        model = MODEL_ROOT + 'best_' + str(p1) + TRAIN_NUM + '.dat'
        if os.path.exists(model):
            done_count = 0
            for _ in range(100):
                done_count += net_test(rand=p1, max_step=p2, net=model)
                # done_count += net_test(rand=p1, max_step=p2, net=INIT_MODEL, action=True)
            print(model, done_count / 100.0)

        model = MODEL_ROOT + 'final_' + str(p1) + TRAIN_NUM + '.dat'
        if os.path.exists(model):
            done_count = 0
            for _ in range(100):
                done_count += net_test(rand=p1, max_step=p2, net=model)
                # done_count += net_test(rand=p1, max_step=p2, net=INIT_MODEL, action=True)
            print(model, done_count / 100.0)


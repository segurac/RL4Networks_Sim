#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from src.common.custom_env import CustomEnv


class QBrain(torch.nn.Module):

    def __init__(self, state_space_size: int, action_space_size: int, hidden_neurons: int, hidden_layers: int):
        super(QBrain, self).__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.layers = torch.nn.ModuleList()
        input_size = self.state_space_size
        for i in range(hidden_layers):
            output_size = hidden_neurons if i < (hidden_layers - 1) else self.action_space_size
            self.layers.append(torch.nn.Linear(input_size, output_size))
            input_size = output_size

    def forward(self, state: torch.tensor) -> torch.tensor:
        """Q values estimation

        :param state: torch.tensor of shape = [batch_size, state_space_size]
        :return torch.tensor, the Q values of shape = [batch_size, action_space_size]
        """
        for i, l in enumerate(self.layers):
            state = l(state)
            if i < (len(self.layers) - 1):
                state = torch.nn.functional.relu(state)
        return state


class DDQNAgent:
    def __init__(self, state_size: int, action_size: int, device: str, memory_size: int, gamma: float,
                 epsilon_start: float, epsilon_min: float, epsilon_decay: float, learning_rate: float,
                 hidden_layers: int, hidden_neurons: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon_start  # At the begining
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.Prev_Mean = 0  # initial mean of the targets (for target normalization)
        self.Prev_std = 1  # initial std of the targets (for target normalization)
        self.model = QBrain(self.state_size, self.action_size, hidden_layers, hidden_neurons).to(device)
        self.target_model = self.model = QBrain(self.state_size, self.action_size, hidden_layers, hidden_neurons).to(device)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _huber_loss(self, y_true: torch.tensor, y_pred: torch.tensor, clip_delta=1.0):
        error = y_true - y_pred
        cond = torch.abs(error) <= clip_delta

        squared_loss = 0.5 * error**2
        quadratic_loss = 0.5 * clip_delta**2 + clip_delta * (torch.abs(error) - clip_delta)

        return torch.mean(torch.where(cond, squared_loss, quadratic_loss))

    def update_target_model(self):
        # copy weights from the CIO selection network to target network
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.array):
        with torch.no_grad():
            act_values = self.model.forward(torch.tensor(state, dtype=torch.float32).to(self.device)).to('cpu').numpy()
            #print("Predicted action for this state is: {}".format(np.argmax(act_values[0])))
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size: int):
        minibatch = random.sample(self.memory, batch_size)
        target_a = []  # for batch level target normalization
        with torch.no_grad():
            for _, _, reward, next_state, _ in minibatch:  # calculate the target array
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                a = self.model.forward(next_state)[0].to('cpu').numpy()
                t = self.target_model.forward(next_state)[0].to('cpu').numpy()
                b = t[np.argmax(a)]  # needs de_normalization
                b *= self.Prev_std
                b = +self.Prev_Mean
                target_a.append(reward + self.gamma * b)

        mean_mini_batch = np.mean(np.asarray(target_a), axis=0)  # mean of the targets in this mini-batch
        std_mini_batch = np.std(np.asarray(target_a), axis=0)  # std of the targets in this mini-batch

        losses = []
        for state, action, reward, next_state, done in minibatch:
            # Casting arrays/tensors
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            self.optimizer.zero_grad()
            predictions = self.model.forward(state)
            if done:
                tg = reward
            else:
                with torch.no_grad():
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                    a = self.model.forward(next_state)[0].to('cpu').numpy()
                    t = self.target_model.forward(next_state)[0].to('cpu').numpy()  # DDQN feature
                    b = t[np.argmax(a)]  # needs de_normalization
                    b *= self.Prev_std  # denormalize the future reward by the mean and std of the previous mini-batch
                    b = +self.Prev_Mean  # denormalized future reward
                    tg = reward + self.gamma * b  #
                    tg -= mean_mini_batch
                    tg /= std_mini_batch  # normalized target
                    self.Prev_std = std_mini_batch
                    self.Prev_Mean = mean_mini_batch
            loss = self._huber_loss(torch.tensor(tg, dtype=torch.float32).detach().to(self.device), predictions[0][action])
            loss.backward()
            self.optimizer.step()
            losses.append(loss.to('cpu').item())
        minibatch_loss = np.mean(np.array(losses))
        if self.epsilon > self.epsilon_min:  # To balance the exploration and exploitation
            self.epsilon *= self.epsilon_decay
        return minibatch_loss

    def load(self, name):
        pass
        # self.model.load_weights(name)

    def save(self, name):
        pass
        # self.model.save_weights(name)


def simulation(sim_agent: DDQNAgent, sim_env: CustomEnv, episode_number: int, batch_size: int,
               save_rewards: bool = False) -> None:
    """DQN Simulation workflow

    :param sim_agent: An instance of DDQNAgent
    :param sim_env: An instance of POC_Env
    :param episode_number: int, number of episodes in the simulation
    :param batch_size: int, batch size for the Agent learning step
    :param save_rewards: bool, True if the cumulative rewards of each episode has to be saved in a file.
    :return None
    """
    writer = SummaryWriter()

    # To trace cumulative reward
    if save_rewards:
        cumulative_rewards = {}
        for k in sim_env.cumulative_rewards.keys():
            cumulative_rewards[k] = []

    for e in range(episode_number):
        # Reset the environment
        state, _ = sim_env.reset()

        # For over episode timesteps
        for time in range(sim_env.get_episode_length()):
            print("*******************************")
            print("episode: {}/{}, step: {}".format(e + 1, episode_number, time))

            # Agent takes an action
            action_index = sim_agent.act(state)

            # Environment reacts to the agent action
            next_state, reward, done, _, _ = sim_env.step(action_index)

            # Sanity check: To avoid crashing the simulation if the handover failiure occured in NS-3 simulation
            OK = 1  # No handover failiure occured
            if next_state is None:
                OK = 0  # Handover failiure occured
                episode_number = episode_number + 1
                break

            # Save SARS in the experience buffer
            sim_agent.remember(state, action_index, reward, next_state, done)
            state = next_state
            print("Step reward:{}".format(reward))

            if len(sim_agent.memory) > batch_size:
                # Agent leaning step
                loss = sim_agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("loss: {:.4f}".format(loss))
        if OK == 1:
            # The episode completed without any NS-3 error

            # Update the target model at the end of each episode (Should it be configurable?)
            sim_agent.update_target_model()

            # Update tensorboard
            for k in sim_env.cumulative_rewards.keys():
                tmp_rew = sim_env.cumulative_rewards[k] / sim_env.get_episode_length()
                writer.add_scalar(k, tmp_rew, e)
                if save_rewards:
                    cumulative_rewards[k].append(tmp_rew)

            writer.flush()
            if (e + 1) % 10 == 0:
                # Model checkpoint
                sim_agent.save("./LTE-DDQN.h5")  # Save the model

                if save_rewards:
                    Result_row = []
                    with open('Rewards_' + 'DDQN' + '.csv', 'w', newline='') as rewardcsv:
                        results_writer = csv.writer(rewardcsv, delimiter=';', quotechar=';', quoting=csv.QUOTE_MINIMAL)
                        for k, v in sim_env.cumulative_rewards.items():
                            Result_row.clear()
                            Result_row = Result_row + v
                            results_writer.writerow(Result_row)
                    rewardcsv.close()
    writer.close()





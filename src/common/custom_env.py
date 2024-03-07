from typing import Any, Dict, List, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ns3gym import ns3env


class CustomEnv(gym.Env):
    """Custom Environment implementing the gym interface.

    This is the class base for POC and RealSce environment where the methods related with the action space are
    overwritten
    """
    # TODO: It seems not used, Do we eliminate it?
    metadata = {"render_modes": ["human"], "render_fps": 30}
    # NOTE: For each cell, 4 global features are traced:
    #  - UserCount: Number of connected users,
    #  - dlThroughput: for each cell, the total download Throughput (higher is better).
    #                  It needs to be normalized, which is the criterium?
    #  - rbUtil: resource block utilization, it a measure of cell saturation 0 <= x <= 1
    #  - MCSPen: modulation and coding scheme. For each cell, only the first 10 MSC index are
    #            taken into account. They are related to low SINR
    def __init__(self,
                 port: int,
                 sim_step_time: float,
                 sim_start_time: float,
                 seed: int,
                 sim_args: Dict,
                 max_env_steps: int,
                 num_of_cells: int,
                 num_of_users: int,
                 max_throu: int,
                 max_msc_idx: int,
                 sum_up_mcs: bool,
                 step_CIO: int,
                 debug: bool):
        """

        :param port: int, port used in the NS3 simulator
        :param sim_step_time: simulation step measured in seconds
        :param sim_start_time: float, simulation start time
        :param seed: int
        :param sim_args: Dict
        :param max_env_steps: int, temporal horizon (length of episode)
        :param num_of_cells: int, number of cells, is should be sincronized with the value in the simulator
        :param num_of_users: int, number of UEs
        :param max_throu: max throughput, it is used only to normalize the throughput
        :param step_CIO: int, step used to discretize CIO values. If step_CIO < 0, CIO values are treated as continous
        :param debug: Bool
        """
        super(CustomEnv, self).__init__()
        self.env = ns3env.Ns3Env(port=port,
                                 stepTime=sim_step_time,
                                 startSim=sim_start_time,
                                 simSeed=seed,
                                 simArgs=sim_args,
                                 debug=debug)
        self.env._max_episode_steps = max_env_steps
        self.max_msc_idx = max_msc_idx
        self.sum_up_mcs = sum_up_mcs
        if self.sum_up_mcs:
            self.feature_per_cell = 3 + 1
        else:
            self.feature_per_cell = 3 + self.max_msc_idx

        self.num_of_cells = num_of_cells
        self.num_of_users = num_of_users
        self.max_throu = max_throu
        self.state_dim = self.num_of_cells * self.feature_per_cell
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high

        self.step_CIO = step_CIO
        self.a_level = None
        self.a_num = None
        if step_CIO > 0:
            # Step CIO > 0 implies action space discretized
            ac_space = self.env.action_space  # Getting the action space
            self.a_level = int(ac_space.high[0])  # CIO levels
            self.a_num = int(ac_space.shape[0])
            self.action_space = spaces.Discrete(int(self.a_level) ** int(self.a_num))
        else:
            self.action_space = spaces.Box(low=-1, high=1,
                                           shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.num_of_users,
                                            shape=(self.state_dim,), dtype=np.float32)
        self.cumulative_rewards = self._reset_cumulative_rewards()

    def step(self, action):
        """Step function.
        1) Translate action from agent to environment format
        2) Call the env.step method
        3) Update the cumulative rewards
        4) Translate the "next_state" from environment to agent format

        :param action: The action selected by the agent (expressed in the agent format)
        :return: ext_observation, reward, done, truncated, info
        """
        action = self._agent_action_2_env_action(action)
        next_state, reward, done, info = self.env.step(action)
        info = {'reward_per_agent': info}
        if next_state is None:
            print('WARNING: next state is NONE!')
            return None, None, None, None, None
        self._update_cumulative_rewards(next_state)
        next_observation = self._state_2_observation(next_state)
        truncated = False
        return next_observation, reward, done, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[np.array, Dict]:
        """Reset function

        1) Reset the cumulative rewards
        2) Reset the environment state
        3) Translate the initial state from environment to agent format

        :param seed
        :param options
        :return the initial state (np.array) and the info dict
        """
        info = {}
        self.cumulative_rewards = self._reset_cumulative_rewards()
        state = self.env.reset()
        observation = self._state_2_observation(state)
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed: int):
        pass

    @staticmethod
    def _reset_cumulative_rewards() -> Dict:
        """Reset/Initialize the cumulative rewards
        :return a Dict containing the initial cumulative rewards
        """
        return {
            'Average_Overall_Throughput': 0.,
            'Average_Deviation': 0.,
            'Percentage_of_Non_blocked_num_of_users': 0.
        }

    def _update_cumulative_rewards(self, state: Dict):
        """Update the cumulative rewards

        :param state: current state, in the environment format
        :return None
        """
        rewards = [j for sub in np.reshape(state['rewards'], [3, 1]) for j in sub]
        self.cumulative_rewards['Average_Overall_Throughput'] += rewards[0]
        self.cumulative_rewards['Average_Deviation'] += rewards[1]
        self.cumulative_rewards['Percentage_of_Non_blocked_num_of_users'] += rewards[2]

    def _state_2_observation(self, state: Dict) -> np.array:
        """Translate the state from environment to agent format

        :param state: Dict, the state in the environment format
        :return state in the agent format, i.e. a np.array
        """
        state1 = np.reshape(state['rbUtil'], [self.num_of_cells, 1])  # Reshape the matrix
        state2 = np.reshape(state['dlThroughput'], [self.num_of_cells, 1])
        state2 = state2 / self.max_throu
        state3 = np.reshape(state['UserCount'], [self.num_of_cells, 1])  # Reshape the matrix
        state3 = state3 / self.num_of_users
        state4 = np.array(state['MCSPen'])[:, :self.max_msc_idx] # For each cell,
                                                                 # only the first n=10 MSC index are taken into account
                                                                 # They are related to low SINR
        if self.sum_up_mcs:
            state4 = np.reshape(np.sum(state4, axis=1), [self.num_of_cells, 1])

        return np.reshape(np.concatenate((state1, state2, state3, state4), axis=None), [1, self.state_dim])

    def _agent_action_2_env_action(self, action_index) -> List:
        """Translate the action from agent to enviroment format

        :param action_index
        :return action vector as a List
        """
        if self.step_CIO > 0:
            print(action_index)

            action = np.base_repr(action_index + int(self.a_level) ** int(self.a_num),
                                  base=int(self.a_level))[-self.a_num:]
            action = [int(a) for a in action]
            action = np.concatenate((np.zeros(self.a_num - len(action)), action), axis=None)
            action = [self.step_CIO * (x - np.floor(self.a_level / 2)) for x in action]  # action vector
            return action
        else:
            print(action_index * self.action_bound)
            return action_index * self.action_bound

    def get_episode_length(self) -> int:
        return self.env._max_episode_steps

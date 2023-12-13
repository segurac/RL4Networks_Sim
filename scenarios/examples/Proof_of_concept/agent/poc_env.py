from typing import Any, Dict, List, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ns3gym import ns3env

from common.custom_env import CustomEnv

class POC_Env(CustomEnv):

    def __init__(self,
                 port: int,
                 sim_step_time: float,
                 sim_start_sim: int,
                 seed: int,
                 sim_args: Dict,
                 max_env_steps: int,
                 num_of_cells: int,
                 num_of_users: int,
                 max_throu: int,
                 debug: bool):
        """POC scenario environment

        :param port: int, port used in the NS3 simulator
        :param sim_step_time: simulation step measured in seconds
        :param sim_start_time: float, simulation start time
        :param seed: int
        :param sim_args: Dict
        :param max_env_steps: int, temporal horizon (length of episode)
        :param num_of_cells: int, number of cells, it should be sincronized with the value in the simulator
        :param num_of_users: int, number of UEs
        :param max_throu: max throughput, it is used only to normalize the throughput
        :param debug: Bool
        """
        super(POC_Env, self).__init__(port, sim_step_time, sim_start_sim, seed, sim_args, max_env_steps, num_of_cells,
                                      num_of_users, max_throu, debug)
        ac_space = self.env.action_space  # Getting the action space
        self.a_level = int(ac_space.high[0])  # CIO levels
        self.a_num = int(ac_space.shape[0])
        self.step_CIO = 3

    def _agent_action_2_env_action(self, action_index) -> List:
        """Translate the action from agent to enviroment format

        :param action_index
        :return action vector as a List
        """
        action = np.base_repr(action_index + int(self.a_level) ** int(self.a_num), base=int(self.a_level))[-self.a_num:]
        action = [int(a) for a in action]
        action = np.concatenate((np.zeros(self.a_num - len(action)), action), axis=None)
        action = [self.step_CIO * (x - np.floor(self.a_level / 2)) for x in action]  # action vector
        return action

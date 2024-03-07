from stable_baselines3 import DQN, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.dqn import MlpPolicy as DQNMlpPolicy
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.td3 import MlpPolicy as TD3MlpPolicy
from subprocess import Popen
import numpy as np
import os
import time

from src.common.custom_env import CustomEnv
from src.processes.process_base import Process
from src.common.utils import Conf
from src.common.vec_monitor import VecMonitor
from src.rl.td3.custom_td3_policy import CustomTD3Policy
from src.rl.td3.marl_td3 import MarlTD3

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
                          It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        print("Steps: {}".format(self.num_timesteps))

        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


class StableBaselineProcess(Process):
    name = "stable_baseline"

    def __init__(self, conf: Conf):
        super(StableBaselineProcess, self).__init__()
        # Number of episodes to run
        self.episodes = conf.n_episodes * conf.max_steps_per_episode
        self.evaluate_agent_each_n_steps = conf.evaluate_agent_each_n_steps
        self.n_eval_episodes = conf.n_eval_episodes

        # Temporal horizon: Maximum number of steps in every episode
        # Note: the duration in seconds of an episode is given by self.max_env_steps * self.sim_time_step
        # Paper values: 10 seconds
        self.max_env_steps = conf.max_steps_per_episode

        # Time step (in seconds)
        self.sim_time_step = conf.envStepTime

        # Initial simulation time.
        self.sim_time_start = 0

        # Number of UEs (User Equipments)
        self.num_of_users = conf.nUEs

        # Number of Base station
        self.num_of_cells = conf.nMacroEnbSites

        # Max throughtput (only used to normalize the signal)
        self.max_throu = conf.max_troughput_normalization

        # MCS params
        self.max_msc_idx = conf.max_msc_idx
        self.sum_up_mcs = conf.sum_up_mcs

        # Step CIO: if step_CIO <= 0 --> CIO are modeled as continous variable
        #           if step_CIO > 0  --> CIO are discretized by "step_CIO"
        self.step_CIO = conf.step_CIO

        # Agent params
        self.agent_type = conf.agent_type
        self.agent_params = conf.agent_params
        self.adjacency_matrix = conf.adjacency_matrix
        self.tensorboard_log_dir = conf.tensorboard_log_dir.format(conf.agent_type)
        self.model_path = conf.model_path
        self.ns3_path = conf.ns3_path
        self.launch_sim_file = conf.launch_sim_file

        # Port: Should be consistent with NS-3 simulation port
        self.port = conf.openGymPort

        # Seed
        self.seed = 3
        self.simArgs = {}

    def run(self):
        # Create log dir
        log_dir = "tmp_{}/".format(int(time.time()))
        os.makedirs(log_dir, exist_ok=True)

        # If required, launch cpp simulator directly from python
        p = None
        if len(self.launch_sim_file) > 0:
            p = Popen([self.launch_sim_file],
                      cwd=self.ns3_path,
                      shell=True,
                      executable='/bin/bash')

        # Build RealSce_env
        custom_env = CustomEnv(port=self.port,
                               sim_step_time=self.sim_time_step,
                               sim_start_time=self.sim_time_start,
                               seed=self.seed,
                               sim_args=self.simArgs,
                               max_env_steps=self.max_env_steps,
                               num_of_cells=self.num_of_cells,
                               num_of_users=self.num_of_users,
                               max_throu=self.max_throu,
                               max_msc_idx=self.max_msc_idx,
                               step_CIO=self.step_CIO,
                               sum_up_mcs=self.sum_up_mcs,
                               debug=True)

        # wrap it
        env = make_vec_env(lambda: custom_env, n_envs=1)
        env = VecMonitor(env, log_dir)

        model = self.get_rl_model(self.agent_type, env)

        # Create the callback: check every episode, check_freq = max_env_steps
        save_callback = SaveOnBestTrainingRewardCallback(check_freq=self.max_env_steps, log_dir=log_dir)
        eval_callback = EvalCallback(env,
                                     n_eval_episodes=self.n_eval_episodes,
                                     best_model_save_path="./logs/best_model",  # TODO: a parametro
                                     log_path="./logs/results",  # TODO: a parametro
                                     eval_freq=self.evaluate_agent_each_n_steps)
        # Create the callback list
        callback = CallbackList([save_callback, eval_callback])

        # Train the agent
        print('Start learning...')
        time_steps = self.episodes
        model.learn(total_timesteps=int(time_steps), callback=callback)
        model.save(self.model_path.format(self.agent_type, int(time.time())))

        if len(self.launch_sim_file) > 0:
            p.terminate()

    def get_rl_model(self, rl_alg: str, env):
        model = None
        if rl_alg == 'sb_marltd3':
            print('MARLTD3 ALGORITHM')
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=(self.agent_params['common_params']['action_noise_sigma'] *
                                                    np.ones(n_actions)))
            policy = CustomTD3Policy
            tmp = self.agent_params['distributed_params'].copy()
            tmp.update({'adjacency_matrix': self.adjacency_matrix})
            policy_kwargs = dict(custom_params=tmp)
            model = MarlTD3(policy,
                            env,
                            reward_per_agent=self.agent_params['common_params']['reward_per_agent'],
                            buffer_size=self.agent_params['common_params']['buffer_size'],
                            learning_starts=self.agent_params['common_params']['learning_starts'],
                            learning_rate=self.agent_params['common_params']['learning_rate'],
                            batch_size=self.agent_params['common_params']['batch_size'],
                            tau=self.agent_params['common_params']['tau'],
                            gamma=self.agent_params['common_params']['gamma'],
                            train_freq=self.agent_params['common_params']['train_freq'],
                            gradient_steps=self.agent_params['common_params']['gradient_steps'],
                            policy_delay=self.agent_params['common_params']['policy_delay'],
                            target_policy_noise=self.agent_params['common_params']['target_policy_noise'],
                            target_noise_clip=self.agent_params['common_params']['target_noise_clip'],
                            stats_window_size=self.agent_params['common_params']['stats_window_size'],
                            action_noise=action_noise,
                            policy_kwargs=policy_kwargs,
                            verbose=1,
                            tensorboard_log=self.tensorboard_log_dir)
        elif rl_alg == 'sb_td3':
            print('TD3 ALGORITHM')
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=(self.agent_params['common_params']['action_noise_sigma'] *
                                                    np.ones(n_actions)))
            policy = TD3MlpPolicy
            policy_kwargs = dict(net_arch=self.agent_params['policy_params']['net_arch'])
            model = TD3(policy,
                        env,
                        buffer_size=self.agent_params['common_params']['buffer_size'],
                        learning_starts=self.agent_params['common_params']['learning_starts'],
                        learning_rate=self.agent_params['common_params']['learning_rate'],
                        batch_size=self.agent_params['common_params']['batch_size'],
                        tau=self.agent_params['common_params']['tau'],
                        gamma=self.agent_params['common_params']['gamma'],
                        train_freq=self.agent_params['common_params']['train_freq'],
                        gradient_steps=self.agent_params['common_params']['gradient_steps'],
                        policy_delay=self.agent_params['common_params']['policy_delay'],
                        target_policy_noise=self.agent_params['common_params']['target_policy_noise'],
                        target_noise_clip=self.agent_params['common_params']['target_noise_clip'],
                        stats_window_size=self.agent_params['common_params']['stats_window_size'],
                        action_noise=action_noise,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log=self.tensorboard_log_dir)
        elif rl_alg == 'sb_ppo':
            print('PPO ALGORITHM')
            policy = PPOMlpPolicy
            policy_kwargs = dict(net_arch=self.agent_params['policy_params']['net_arch'])
            model = PPO(policy,
                        env,
                        n_steps=self.agent_params['common_params']['n_steps'],
                        batch_size=self.agent_params['common_params']['batch_size'],
                        n_epochs=self.agent_params['common_params']['n_epochs'],
                        learning_rate=self.agent_params['common_params']['learning_rate'],
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log=self.tensorboard_log_dir)
        elif rl_alg == 'sb_dqn':
            print('DQN ALGORITHM')
            policy = DQNMlpPolicy
            policy_kwargs = dict(net_arch=self.agent_params['policy_params']['net_arch'])
            model = DQN(policy,
                        env,
                        learning_rate=self.agent_params['learning_rate'],   # tuned: default value --> 1e-4,
                        buffer_size=self.agent_params['buffer_size'],  # tuned: 20K --> default=1e6
                        learning_starts=self.agent_params['learning_starts'],  # tuned: default --> 50000
                        batch_size=self.agent_params['batch_size'],       # tuned: OK, is the default value
                        tau=self.agent_params['tau'],
                        gamma=self.agent_params['gamma'],   # tuned: default value 0.99
                        train_freq=1,   #tuned: every step --> : Union[int, Tuple[int, str]] = 4,
                        gradient_steps=1,
                        replay_buffer_class=None,
                        replay_buffer_kwargs=None,
                        optimize_memory_usage=False,
                        target_update_interval=self.agent_params['target_update_episodes'] * self.max_env_steps,   # tuned: every episode, default: 10000,
                        exploration_fraction=self.agent_params['exploration_fraction'],
                        exploration_initial_eps=self.agent_params['exploration_initial_eps'],
                        exploration_final_eps=self.agent_params['exploration_final_eps'],
                        max_grad_norm=10,
                        stats_window_size=100,
                        tensorboard_log=self.tensorboard_log_dir,
                        policy_kwargs=policy_kwargs,  # tuned: default None
                        seed=None,
                        device="auto",
                        verbose=1)
        if model is None:
            raise Exception(f'RL algorithm {rl_alg} is not currently supported')
        return model

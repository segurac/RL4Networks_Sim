from src.processes.process_base import Process
from src.common.utils import Conf
from src.common.custom_env import CustomEnv
from src.rl.ddqn.ddqn_agent_pytorch import DDQNAgent, simulation


class POC_Process(Process):
    name = 'poc'

    def __init__(self, conf: Conf):
        super(POC_Process, self).__init__()
        # Number of episodes to run
        self.episodes = conf.n_episodes

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

        # Port: Should be consistent with NS-3 simulation port
        self.port = conf.openGymPort

        # Seed
        self.seed = 3
        self.simArgs = {}

        # Discrete action parametrization
        self.step_CIO = conf.step_CIO  # CIO value step in the discrete set {-6, -3, 0, 3, 6}

        # Agent Params
        self.batch_size = conf.agent_params['batch_size']
        self.learning_rate = conf.agent_params['learning_rate']
        self.device = conf.device
        self.memory_size = conf.agent_params['memory_size']
        self.gamma = conf.agent_params['gamma']  # Discount rate
        self.epsilon = conf.agent_params['epsilon_start']  # At the beginning
        self.epsilon_min = conf.agent_params['epsilon_min']
        self.epsilon_decay = conf.agent_params['epsilon_decay']
        # The hidden layer number has to include also the input and output layers
        self.hidden_layers = conf.agent_params['hidden_layers']
        self.hidden_neurons = conf.agent_params['hidden_neurons']

    def run(self):
        # Build POC_env
        env = CustomEnv(port=self.port,
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

        # Action and state space info
        ac_space = env.env.action_space  # Getting the action space
        state_size = self.num_of_cells * env.feature_per_cell
        a_level = int(ac_space.high[0])  # CIO levels  --> 5 in this case, corresponding to [-6, -3, 0, 3, 6]
        a_num = int(ac_space.shape[0])   # number of required relative CIOs --> 3 cells - 1 = 2
        action_size = a_level ** a_num

        # Build Agent
        agent = DDQNAgent(state_size,
                          action_size,
                          device=self.device,
                          memory_size=self.memory_size,
                          gamma=self.gamma,
                          epsilon_start=self.epsilon,
                          epsilon_min=self.epsilon_min,
                          epsilon_decay=self.epsilon_decay,
                          learning_rate=self.learning_rate,
                          hidden_layers=self.hidden_layers,
                          hidden_neurons=self.hidden_neurons)

        # Simulate
        simulation(agent, env, self.episodes, self.batch_size)

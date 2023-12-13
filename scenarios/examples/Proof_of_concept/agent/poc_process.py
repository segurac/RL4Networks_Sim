from common.process_base import Process
from scenarios.examples.Proof_of_concept.agent.poc_env import POC_Env
from scenarios.examples.Proof_of_concept.agent.ddqn_agent_pytorch import DDQNAgent, simulation


class POC_Process(Process):
    def __init__(self):
        super(POC_Process, self).__init__()
        """
        TODO: All the parameters should coming from a config file. All the Agent parameters are currently hard-coded the 
              Agent file!
        """
        # Number of episodes to run
        self.episodes = 200

        # Port: # Should be consistent with NS-3 simulation port
        self.port = 1122

        # Time step (in seconds)
        self.sim_time_step = 0.2

        # Initial simulation time. TODO: To investigate: it seems related with the mobility file
        self.sim_time_start = 0

        # Number of UEs (User Equipments) TODO:  Should be consistent with NS-3 params?
        self.num_of_users = 41

        # Number of Base station  TODO: Should be consistent with NS-3
        self.num_of_cells = 3

        # Max throughtput (only used to normalize the signal)
        self.max_throu = 18

        # Seed
        self.seed = 3
        self.simArgs = {}

        # Discrete action parametrization
        self.step_CIO = 3  # CIO value step in the discrete set {-6, -3, 0, 3, 6}

        # Temporal horizon: Maximum number of steps in every episode
        # Note: the duration in seconds of an episode is given by self.max_env_steps * self.sim_time_step
        # Paper values: 10 seconds
        self.max_env_steps = 50

        # Agent train batch_size
        self.batch_size = 32
        # NOTE: All the DDQN Agent hiperparameters are hard-coded in the DDQN_Agent __init__

    def run(self):
        # Build POC_env
        env = POC_Env(port=self.port,
                      sim_step_time=self.sim_time_step,
                      sim_start_sim=self.sim_time_start,
                      seed=self.seed,
                      sim_args=self.simArgs,
                      max_env_steps=self.max_env_steps,
                      num_of_cells=self.num_of_cells,
                      num_of_users=self.num_of_users,
                      max_throu=self.max_throu,
                      debug=True)

        # Action and state space info
        ac_space = env.env.action_space  # Getting the action space
        state_size = 12
        a_level = int(ac_space.high[0])  # CIO levels  --> 5 in this case, corresponding to [-6, -3, 0, 3, 6]
        a_num = int(ac_space.shape[0])   # number of required relative CIOs --> 3 cells - 1 = 2
        action_size = a_level ** a_num

        # Build Agent
        agent = DDQNAgent(state_size, action_size, device='cpu')

        # Simulate
        simulation(agent, env, self.episodes, self.batch_size)

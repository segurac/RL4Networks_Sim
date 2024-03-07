from gymnasium import spaces
from typing import Any, Dict, List, Optional, Type, Tuple, Union
from torch import nn
from stable_baselines3.common.policies import BasePolicy, BaseModel, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.td3.policies import Actor, TD3Policy
import torch as th

from src.rl.td3.brain.gnn_brain import GNNBrainActor, GNNBrainCLSActor, GNNBrainCritic
from src.rl.td3.brain.mlp_brain import MLPBrain


class CustomActor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    input_keys = ['observation_space', 'action_space', 'custom_params',
                  'features_extractor', 'features_dim']

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        custom_params: Dict,
        features_extractor: nn.Module,
        features_dim: int,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=True,
            squash_output=True,
        )
        self.custom_params = custom_params
        self.adjacency_matrix = custom_params['adjacency_matrix']
        self.number_of_cells = self.adjacency_matrix.shape[0]
        self.features_dim = features_dim
        self.action_dim = get_action_dim(self.action_space)

        features_per_cells = self.features_dim // self.number_of_cells
        if custom_params['actor_gnn']:
            if custom_params['actor_gnn_cls']:
                self.mu = GNNBrainCLSActor(self.number_of_cells,
                                           features_per_cells,
                                           latent_space_size=custom_params['latent_space_size'],
                                           gnn_stack_num=custom_params['actor_gnn_stack_size'],
                                           gnn_mlp_deep=custom_params['gnn_mlp_deep'],
                                           gnn_dropout=custom_params['gnn_dropout'],
                                           adjacency_matrix=self.adjacency_matrix)
            else:
                self.mu = GNNBrainActor(self.number_of_cells,
                                        features_per_cells,
                                        latent_space_size=custom_params['latent_space_size'],
                                        gnn_stack_num=custom_params['actor_gnn_stack_size'],
                                        gnn_mlp_deep=custom_params['gnn_mlp_deep'],
                                        gnn_dropout=custom_params['gnn_dropout'],
                                        adjacency_matrix=self.adjacency_matrix)
        else:
            self.mu = MLPBrain(self.number_of_cells,
                               features_per_cells,
                               output_size=self.action_dim,
                               latent_space_size=custom_params['latent_space_size'],
                               hidden_layers=custom_params['mlp_hidden_layers'],
                               is_critic=False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                custom_params=self.custom_params,
                features_dim=self.features_dim,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        return self.mu.forward_actor(features)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)


class CustomContinuousCritic(BaseModel):
    """
        Critic network(s) for DDPG/SAC/TD3.
        It represents the action-state value function (Q-value function).
        Compared to A2C/PPO critics, this one represents the Q-value
        and takes the continuous action as input. It is concatenated with the state
        and then fed to the network which outputs a single value: Q(s, a).
        For more recent algorithms like SAC/TD3, multiple networks
        are created to give different estimates.

        By default, it creates two critic networks used to reduce overestimation
        thanks to clipped Q-learning (cf TD3 paper).

        :param observation_space: Obervation space
        :param action_space: Action space
        :param features_extractor: Network to extract features
            (a CNN when using images, a nn.Flatten() layer otherwise)
        :param features_dim: Number of features
        :param n_critics: Number of critic networks to create.
        :param share_features_extractor: Whether the features extractor is shared or not
            between the actor and the critic (this saves computation time)
        """

    features_extractor: BaseFeaturesExtractor
    input_keys = ['observation_space', 'action_space', 'custom_params',
                  'features_extractor', 'features_dim']

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        custom_params: Dict,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=True,
        )
        self.adjacency_matrix = custom_params['adjacency_matrix']
        self.number_of_cells = self.adjacency_matrix.shape[0]
        self.features_dim = features_dim
        self.action_dim = get_action_dim(self.action_space)

        features_per_cells = self.features_dim // self.number_of_cells
        assert (self.features_dim % self.number_of_cells) == 0

        def create_nn():
            if custom_params['critic_gnn']:
                return GNNBrainCritic(self.number_of_cells,
                                      features_per_cells,
                                      latent_space_size=custom_params['latent_space_size'],
                                      gnn_stack_num=custom_params['critic_gnn_stack_size'],
                                      gnn_mlp_deep=custom_params['gnn_mlp_deep'],
                                      gnn_dropout=custom_params['gnn_dropout'],
                                      adjacency_matrix=self.adjacency_matrix)
            else:
                return MLPBrain(self.number_of_cells,
                                features_per_cells,
                                output_size=self.action_dim,
                                latent_space_size=custom_params['latent_space_size'],
                                hidden_layers=custom_params['mlp_hidden_layers'],
                                is_critic=True)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net = create_nn()
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net.forward_critic(features, actions) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0].forward_critic(features, actions)


class CustomTD3Policy(TD3Policy):
    """
        Policy class (with both actor and critic) for TD3.

        :param observation_space: Observation space
        :param action_space: Action space
        :param lr_schedule: Learning rate schedule (could be constant)
        :param net_arch: The specification of the policy and value networks.
        :param activation_fn: Activation function
        :param features_extractor_class: Features extractor to use.
        :param features_extractor_kwargs: Keyword arguments
            to pass to the features extractor.
        :param normalize_images: Whether to normalize images or not,
             dividing by 255.0 (True by default)
        :param optimizer_class: The optimizer to use,
            ``th.optim.Adam`` by default
        :param optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
        :param n_critics: Number of critic networks to create.
        :param share_features_extractor: Whether to share or not the features extractor
            between the actor and the critic (this saves computation time)
        """

    actor: CustomActor
    actor_target: CustomActor
    critic: CustomContinuousCritic
    critic_target: CustomContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        custom_params: Dict,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        self.custom_params = custom_params
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Union[Actor, CustomActor]:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update({"custom_params": self.custom_params })
        final_actor_kwargs = {k: v for k, v in actor_kwargs.items() if k in CustomActor.input_keys}
        return CustomActor(**final_actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Union[ContinuousCritic,
                                                                                               CustomContinuousCritic]:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update({"custom_params": self.custom_params })
        final_ic_kwargs = {k: v for k, v in critic_kwargs.items() if k in CustomContinuousCritic.input_keys}
        return CustomContinuousCritic(**final_ic_kwargs).to(self.device)

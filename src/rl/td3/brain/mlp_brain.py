import torch
import torch.nn as nn


class MLPBrain(nn.Module):
    def __init__(self, number_of_cells: int, features_per_cells: int, output_size: int,
                 latent_space_size: int, hidden_layers: int, is_critic: bool):
        super(MLPBrain, self).__init__()
        total_input_features = number_of_cells * features_per_cells
        if is_critic:
            total_input_features += output_size

        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_layers):
            input_size = latent_space_size if i > 0 else total_input_features
            self.hidden_layers.append(nn.Linear(input_size, latent_space_size))
        self.output = nn.Linear(latent_space_size, output_size)

    def forward_actor(self, observations: torch.tensor) -> torch.tensor:
        for c_layer in self.hidden_layers:
            observations = nn.functional.relu(c_layer(observations))
        return nn.functional.tanh(self.output(observations))

    def forward_critic(self, observations: torch.tensor, actions: torch.tensor) -> torch.tensor:
        x = torch.cat([observations, actions], dim=1)
        for c_layer in self.hidden_layers:
            x = nn.functional.relu(c_layer(x))
        return self.output(x).unsqueeze(dim=2)


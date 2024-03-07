from typing import Optional
import torch
import torch.nn as nn

from src.rl.td3.brain.interaction_network import InteractionNetwork


class GNNBrainCritic(nn.Module):
    def __init__(self, number_of_cells: int, features_per_cells: int,
                 latent_space_size: int, gnn_stack_num: int, gnn_mlp_deep: int, gnn_dropout: float,
                 adjacency_matrix: torch.tensor):
        super(GNNBrainCritic, self).__init__()
        self.features_per_cells = features_per_cells
        self.number_of_cells = number_of_cells
        self.adjacency_matrix = adjacency_matrix

        # Adapt adjacency matrix to PytorchGeometric format
        count = 0
        n1n2_edges = [0 for _ in range(number_of_cells)] + [(i + 1) for i in range(number_of_cells)]
        n2n1_edges = [(i + 1) for i in range(number_of_cells)] + [0 for _ in range(number_of_cells)]
        for i in range(number_of_cells):
            for j in range(i):
                if self.adjacency_matrix[i, j] > 0:
                    n1n2_edges.extend([i + 1, j + 1])
                    n2n1_edges.extend([j + 1, i + 1])
                    count += 1
        self.gnn_adj_matrix = torch.tensor([n1n2_edges, n2n1_edges], dtype=torch.long)
        self.cios = count

        cls = torch.Tensor(1, latent_space_size)
        self.cls = nn.Parameter(nn.init.xavier_uniform_(cls))

        # Observations embedding
        self.feature_embedding = nn.Linear(features_per_cells, latent_space_size)
        self.norm_embedding = nn.LayerNorm([number_of_cells, latent_space_size])

        # Action embedding
        self.action_embedding = nn.Linear(self.cios, latent_space_size)
        self.norm_acts_embedding = nn.LayerNorm(latent_space_size)

        # GNN
        self.node_dropouts = nn.ModuleList()
        self.edge_dropouts = nn.ModuleList()
        self.gnns = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        for i in range(gnn_stack_num):
            self.node_dropouts.append(nn.Dropout(p=gnn_dropout))
            if i > 0:
                self.edge_dropouts.append(nn.Dropout(p=gnn_dropout))
            self.gnns.append(InteractionNetwork(1,
                                                latent_space_size,
                                                latent_space_size,
                                                gnn_mlp_deep,
                                                with_input_channels=(i > 0),
                                                with_edge_features=(i > 0)))
            self.node_norms.append(nn.LayerNorm(latent_space_size))
            if i < (gnn_stack_num - 1):
                self.edge_norms.append(nn.LayerNorm(latent_space_size))

        # Final MLP
        self.output = nn.Linear(2 * latent_space_size, self.cios)

    def forward_embedding(self, observations: torch.tensor) -> torch.tensor:
        """First common embedding

        :param observations: torch.tensor of shape = [batch_size, num_cells * features_per_cell]
        :return embedding: torch.tensor of shape = [batch_size, num_cells + 1, latent_space_size]
        """
        batch_size = observations.shape[0]
        observations = torch.nn.functional.relu(self.feature_embedding(observations
                                                                       .reshape(observations.shape[0],
                                                                                self.number_of_cells,
                                                                                self.features_per_cells)))
        return torch.cat([self.cls.repeat(batch_size, 1, 1), self.norm_embedding(observations)], dim=1)

    def forward_acts_embedding(self, actions: torch.tensor) -> torch.tensor:
        """

        :param actions: torch.tensor of shape = [batch_size, number of cio]
        :param torch.tensor of shape = [batch_size, latent_space_size]
        """
        result = self.action_embedding(actions)
        return self.norm_acts_embedding(result)

    def forward_gnn(self, embedding: torch.tensor, action_embedding: Optional[torch.tensor]) -> torch.tensor:
        """

        :param embedding: torch.tensor, shape = [batch_size, num_cells + 1, latent_space_size]
        :param action_embedding: torch.tensor, shape = [batch_size, latent_space_size]
        :return Dict
        """
        node_features = embedding
        edge_features = None

        # Encoder GNN
        for i, gnn in enumerate(self.gnns):
            node_features = self.node_dropouts[i](node_features)
            if i > 0:
                self.edge_dropouts[i - 1](edge_features)
            node_features, edge_features = gnn(x=node_features,
                                               edge_index=self.gnn_adj_matrix,
                                               edge_attr=edge_features)
            node_features = self.node_norms[i](node_features.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            if i < (len(self.gnns) - 1):
                edge_features = self.edge_norms[i](edge_features.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        final_features = torch.cat([node_features[0, :, 0, :], action_embedding], dim=1)
        return self.output(final_features)

    def forward_critic(self, observations: torch.Tensor, actions: torch.tensor) -> torch.tensor:
        obs_embs = self.forward_embedding(observations)
        acts_embs = self.forward_acts_embedding(actions)
        return self.forward_gnn(obs_embs, acts_embs).unsqueeze(dim=2)


class GNNBrainCLSActor(nn.Module):
    def __init__(self, number_of_cells: int, features_per_cells: int,
                 latent_space_size: int, gnn_stack_num: int, gnn_mlp_deep: int, gnn_dropout: float,
                 adjacency_matrix: torch.tensor):
        super(GNNBrainCLSActor, self).__init__()
        self.features_per_cells = features_per_cells
        self.number_of_cells = number_of_cells
        self.adjacency_matrix = adjacency_matrix

        # Adapt adjacency matrix to PytorchGeometric format
        count = 0
        n1n2_edges = [0 for _ in range(number_of_cells)] + [(i + 1) for i in range(number_of_cells)]
        n2n1_edges = [(i + 1) for i in range(number_of_cells)] + [0 for _ in range(number_of_cells)]
        for i in range(number_of_cells):
            for j in range(i):
                if self.adjacency_matrix[i, j] > 0:
                    n1n2_edges.extend([i + 1, j + 1])
                    n2n1_edges.extend([j + 1, i + 1])
                    count += 1
        self.gnn_adj_matrix = torch.tensor([n1n2_edges, n2n1_edges], dtype=torch.long)
        self.cios = count

        cls = torch.Tensor(1, latent_space_size)
        self.cls = nn.Parameter(nn.init.xavier_uniform_(cls))

        # Observations embedding
        self.feature_embedding = nn.Linear(features_per_cells, latent_space_size)

        # GNN
        self.node_dropouts = nn.ModuleList()
        self.edge_dropouts = nn.ModuleList()
        self.gnns = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        for i in range(gnn_stack_num):
            self.node_dropouts.append(nn.Dropout(p=gnn_dropout))
            if i > 0:
                self.edge_dropouts.append(nn.Dropout(p=gnn_dropout))
            self.gnns.append(InteractionNetwork(1,
                                                latent_space_size,
                                                latent_space_size,
                                                gnn_mlp_deep,
                                                with_input_channels=(i > 0),
                                                with_edge_features=(i > 0)))
            self.node_norms.append(nn.LayerNorm(latent_space_size))
            if i < (gnn_stack_num - 1):
                self.edge_norms.append(nn.LayerNorm(latent_space_size))

        # Final MLP
        self.hidden = nn.Linear(latent_space_size, latent_space_size)
        self.output = nn.Linear(latent_space_size, self.cios)

    def forward_embedding(self, observations: torch.tensor) -> torch.tensor:
        """First common embedding

        :param observations: torch.tensor of shape = [batch_size, num_cells * features_per_cell]
        :return embedding: torch.tensor of shape = [batch_size, num_cells + 1, latent_space_size]
        """
        batch_size = observations.shape[0]
        observations = torch.nn.functional.relu(self.feature_embedding(observations
                                                                       .reshape(observations.shape[0],
                                                                                self.number_of_cells,
                                                                                self.features_per_cells)))
        return torch.cat([self.cls.repeat(batch_size, 1, 1), observations], dim=1)

    def forward_gnn(self, embedding: torch.tensor) -> torch.tensor:
        """

        :param embedding: torch.tensor, shape = [batch_size, num_cells + 1, latent_space_size]
        :return torch.tensor
        """
        node_features = embedding
        edge_features = None

        # Encoder GNN
        for i, gnn in enumerate(self.gnns):
            node_features = self.node_dropouts[i](node_features)
            if i > 0:
                self.edge_dropouts[i - 1](edge_features)
            node_features, edge_features = gnn(x=node_features,
                                               edge_index=self.gnn_adj_matrix,
                                               edge_attr=edge_features)
            node_features = self.node_norms[i](node_features.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            if i < (len(self.gnns) - 1):
                edge_features = self.edge_norms[i](edge_features.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        final_features = node_features[0, :, 0, :]
        return torch.nn.functional.tanh(
                self.output(torch.nn.functional.relu(self.hidden(final_features))))

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        obs_embs = self.forward_embedding(observations)
        return self.forward_gnn(obs_embs)


class GNNBrainActor(nn.Module):
    def __init__(self, number_of_cells: int, features_per_cells: int,
                 latent_space_size: int, gnn_stack_num: int, gnn_mlp_deep: int, gnn_dropout: float,
                 adjacency_matrix: torch.tensor):
        super(GNNBrainActor, self).__init__()
        self.features_per_cells = features_per_cells
        self.number_of_cells = number_of_cells
        self.adjacency_matrix = adjacency_matrix

        # Adapt adjacency matrix to PytorchGeometric format
        count = 0
        n1n2_edges = []
        n2n1_edges = []
        for i in range(number_of_cells):
            for j in range(i):
                if self.adjacency_matrix[i, j] > 0:
                    n1n2_edges.extend([i, j])
                    n2n1_edges.extend([j, i])
                    count += 1
        self.gnn_adj_matrix = torch.tensor([n1n2_edges, n2n1_edges], dtype=torch.long)
        self.cios = count

        self.adjacency_matrix_computation = torch.tensor([
            [0, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0],
        ])
        n1n2_edges = []
        n2n1_edges = []
        for i in range(number_of_cells):
            for j in range(i):
                if self.adjacency_matrix[i, j] > 0:
                    n1n2_edges.extend([i, j])
                    n2n1_edges.extend([j, i])
                    count += 1
        self.gnn_adj_matrix_computation = torch.tensor([n1n2_edges, n2n1_edges], dtype=torch.long)

        # Observations embedding
        self.feature_embedding = nn.Linear(features_per_cells, latent_space_size)

        # GNN
        self.node_dropouts = nn.ModuleList()
        self.edge_dropouts = nn.ModuleList()
        self.gnns = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        for i in range(gnn_stack_num):
            self.node_dropouts.append(nn.Dropout(p=gnn_dropout))
            if i > 0:
                self.edge_dropouts.append(nn.Dropout(p=gnn_dropout))
            self.gnns.append(InteractionNetwork(1,
                                                latent_space_size,
                                                latent_space_size,
                                                gnn_mlp_deep,
                                                with_input_channels=(i > 0),
                                                with_edge_features=(i > 0)))
            self.node_norms.append(nn.LayerNorm(latent_space_size))
            if i < (gnn_stack_num - 1):
                self.edge_norms.append(nn.LayerNorm(latent_space_size))

        # Final MLP
        self.hidden = nn.Linear(2 * latent_space_size, latent_space_size)
        self.output = nn.Linear(latent_space_size, 1)

    def forward_embedding(self, observations: torch.tensor) -> torch.tensor:
        """First common embedding

        :param observations: torch.tensor of shape = [batch_size, num_cells * features_per_cell]
        :return embedding: torch.tensor of shape = [batch_size, num_cells, latent_space_size]
        """
        return torch.nn.functional.relu(self.feature_embedding(observations
                                                               .reshape(observations.shape[0],
                                                                        self.number_of_cells,
                                                                        self.features_per_cells)))

    def forward_gnn(self, embedding: torch.tensor) -> torch.tensor:
        """

        :param embedding: torch.tensor, shape = [batch_size, num_cells, latent_space_size]
        :return torch.tensor
        """
        cio_predict = []

        node_features = embedding
        edge_features = None

        # Encoder GNN
        for i, gnn in enumerate(self.gnns):
            node_features = self.node_dropouts[i](node_features)
            if i > 0:
                self.edge_dropouts[i - 1](edge_features)
            node_features, edge_features = gnn(x=node_features,
                                               edge_index=self.gnn_adj_matrix,
                                               edge_attr=edge_features)
            node_features = self.node_norms[i](node_features.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            if i < (len(self.gnns) - 1):
                edge_features = self.edge_norms[i](edge_features.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        for c in range(self.cios):
            # Recover the corresponding edge features
            # final_features = torch.cat([edge_features[0, :, 2 * c, :], edge_features[0, :, 2 * c + 1, :]], dim=1)
            n1 = self.gnn_adj_matrix[0, 2*c]
            n2 = self.gnn_adj_matrix[1, 2*c]
            final_features = torch.cat([node_features[0, :, n1, :], node_features[0, :, n2, :]], dim=1)

            # Final prediction
            cio_predict.append(torch.nn.functional.tanh(
                self.output(torch.nn.functional.relu(self.hidden(final_features)))))
        return torch.cat(cio_predict, dim=1)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        obs_embs = self.forward_embedding(observations)
        return self.forward_gnn(obs_embs)

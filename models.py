"""
models.py
Shared GNN encoder, pretraining heads, and losses for the crystal SSL benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

NUM_ATOM_TYPES = 101
MASK_TOKEN_ID = 100


class GNNEncoder(nn.Module):
    """
    Edge-aware GNN encoder with Jumping Knowledge.

    Uses:
    - Atom embeddings
    - Edge-conditioned GINEConv
    - LayerNorm
    - Jumping Knowledge concatenation
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()

        self.embed = nn.Embedding(
            NUM_ATOM_TYPES,
            hidden_channels,
        )

        self.edge_encoders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        for _ in range(num_layers):

            self.edge_encoders.append(
                nn.Sequential(
                    nn.Linear(1, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_channels,
                        hidden_channels,
                    ),
                )
            )

            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(
                            hidden_channels,
                            hidden_channels,
                        ),
                        nn.ReLU(),
                        nn.Linear(
                            hidden_channels,
                            hidden_channels,
                        ),
                    )
                )
            )

            self.bns.append(
                nn.LayerNorm(hidden_channels)
            )

        # Jumping Knowledge projection
        self.jk_proj = nn.Sequential(
            nn.Linear(
                hidden_channels * (num_layers + 1),
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                hidden_channels,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):

        x = torch.clamp(
            x.squeeze(-1).long(),
            0,
            NUM_ATOM_TYPES - 1,
        )

        x = self.embed(x)

        layer_outputs = [x]

        for conv, norm, edge_mlp in zip(
            self.convs,
            self.bns,
            self.edge_encoders,
        ):

            edge_emb = edge_mlp(edge_attr)

            x = conv(
                x,
                edge_index,
                edge_attr=edge_emb,
            )

            x = norm(x)
            x = F.relu(x)

            layer_outputs.append(x)

        x = torch.cat(
            layer_outputs,
            dim=-1,
        )

        x = self.jk_proj(x)

        return x


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

    def forward(self, x):

        return F.normalize(
            self.net(x),
            dim=-1,
        )


class MaskedAtomDecoder(nn.Module):
    """
    Predict masked atom identity.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_classes: int = NUM_ATOM_TYPES,
    ):
        super().__init__()

        self.net = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(self, x):

        return self.net(x)


class RegHead(nn.Module):
    def __init__(self, input_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
):

    z1 = F.normalize(
        z1,
        dim=1,
    )

    z2 = F.normalize(
        z2,
        dim=1,
    )

    batch_size = z1.size(0)

    out = torch.cat(
        [z1, z2],
        dim=0,
    )

    sim_matrix = torch.exp(
        torch.mm(
            out,
            out.t().contiguous(),
        )
        / temperature
    )

    mask = (
        torch.ones_like(sim_matrix)
        - torch.eye(
            2 * batch_size,
            device=sim_matrix.device,
        )
    ).bool()

    sim_matrix = sim_matrix.masked_select(mask).view(
        2 * batch_size,
        -1,
    )

    pos_sim = torch.exp(
        torch.sum(
            z1 * z2,
            dim=-1,
        )
        / temperature
    )

    pos_sim = torch.cat(
        [pos_sim, pos_sim],
        dim=0,
    )

    loss = -torch.log(
        pos_sim
        / sim_matrix.sum(dim=-1)
    )

    return loss.mean()
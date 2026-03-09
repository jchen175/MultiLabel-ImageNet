import torch.nn as nn
import torch.nn.functional as F

class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, apply_input_norm, apply_output_norm):
        super(LinearHead, self).__init__()
        self.dropout_rate = dropout_rate
        self.apply_input_norm = apply_input_norm
        self.apply_output_norm = apply_output_norm

        self.layers = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(input_dim, output_dim)
        )
        self.layers[-1].weight.data.normal_(mean=0.0, std=0.01)
        self.ssl_info = None

    def forward(self, x):
        if self.apply_input_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.layers(x)
        if self.apply_output_norm:
            x = F.normalize(x, p=2, dim=1)
        return x


class MLPHead(nn.Module):
    """
    adapt from https://arxiv.org/pdf/2206.15369
    """
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, num_layers,
                 apply_input_norm=False, apply_output_norm=True,
                 norm_type=None, dropout_rate=0.0):
        """
        Args:
            input_dim (int): Dimension of input features (d)
            hidden_dim (int): Dimension of hidden layers (d_h)
            bottleneck_dim (int): Dimension of output bottleneck (d_b)
            num_layers (int): Number of hidden layers (L)
            apply_input_norm (bool): Whether to apply L2 norm to input x
            apply_output_norm (bool): Whether to apply L2 norm to output
        """
        super().__init__()
        self.apply_input_norm = apply_input_norm
        self.apply_output_norm = apply_output_norm

        layers = []

        # Build L hidden layers: (Linear -> BatchNorm -> GELU)
        for _ in range(num_layers):
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(input_dim, hidden_dim))
            if norm_type == 'batchnorm':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == 'layernorm':
                layers.append(nn.LayerNorm(hidden_dim))
            elif norm_type is not None:
                raise ValueError(f"Unknown normalization type: {norm_type}")
            layers.append(nn.GELU())
            input_dim = hidden_dim  # for next layer

        # Final projection to bottleneck_dim
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))

        self.net = nn.Sequential(*layers)
        self.ssl_info = None

    def forward(self, x):
        if self.apply_input_norm:
            x = F.normalize(x, p=2, dim=1)

        x = self.net(x)

        if self.apply_output_norm:
            x = F.normalize(x, p=2, dim=1)

        return x


def get_cls_head(arch, input_dim, output_dim, args):
    if arch == 'linear':
        return LinearHead(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout_rate=args.dropout,
            apply_input_norm=args.apply_input_norm,
            apply_output_norm=args.apply_output_norm
        )
    elif arch == 'mlp':
        return MLPHead(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=output_dim,
            num_layers=args.num_layers,  # Default to 2 hidden layers
            apply_input_norm=args.apply_input_norm,
            apply_output_norm=args.apply_output_norm,
            norm_type=args.norm,
            dropout_rate=args.dropout
        )

    else:
        raise ValueError(f"Unknown head type: {arch}")

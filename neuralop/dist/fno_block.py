from .collective import all_to_all
import torch.distributed as dist

def forward_with_postactivation(self, x, index=0, output_shape=None):

    group = dist.group.WORLD
    x_skip_fno = self.fno_skips[index](x)
    x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

    if self.mlp is not None:
        x_skip_mlp = self.mlp_skips[index](x)
        x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

    if self.stabilizer == "tanh":
        x = torch.tanh(x)

    x = all_to_all(x, group, scatter_dim=1, gather_dim=2)
    x_fno = self.convs(x, index, output_shape=output_shape)
    x_fno = all_to_all(x_fno, group, scatter_dim=2, gather_dim=1)
    
    if self.norm is not None:
        x_fno = self.norm[self.n_norms * index](x_fno)

    x = x_fno + x_skip_fno

    if (self.mlp is not None) or (index < (self.n_layers - 1)):
        x = self.non_linearity(x)

    if self.mlp is not None:
        x = self.mlp[index](x) + x_skip_mlp

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

    return x
from .collective import _ReduceFromModelParallelRegion
import torch
import torch.distributed as dist
from typing import List, Optional, Union, Tuple

def forward(
    self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
):
    """Generic forward pass for the Factorized Spectral Conv

    Parameters
    ----------
    x : torch.Tensor
        input activation of size (batch_size, channels, d1, ..., dN)
    indices : int, default is 0
        if joint_factorization, index of the layers for n_layers > 1

    Returns
    -------
    tensorized_spectral_conv(x)
    """
    group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    local_rank = dist.get_rank(group)
    
    batchsize, channels, *mode_sizes = x.shape

    fft_size = list(mode_sizes)
    fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
    fft_dims = list(range(-self.order, 0))

    if self.fno_block_precision == "half":
        x = x.half()

    x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
    if self.order > 1:
        x = torch.fft.fftshift(x, dim=fft_dims[:-1])

    if self.fno_block_precision == "mixed":
        # if 'mixed', the above fft runs in full precision, but the
        # following operations run at half precision
        x = x.chalf()

    if self.fno_block_precision in ["half", "mixed"]:
        out_dtype = torch.chalf
    else:
        out_dtype = torch.cfloat
    out_fft = torch.zeros([batchsize, self.out_channels, *fft_size],
                          device=x.device, dtype=out_dtype)

    weight = self._get_weight(indices)

    channels = x.shape[1]
    
    starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)]
    slices_w =  [slice(channels * local_rank, channels * (local_rank + 1)), slice(None)] # Batch_size, channels
    # slices_w =  [slice(None), slice(None)] # Batch_size, channels
    slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
    slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed

    weight = weight[slices_w]
    
    starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))]
    slices_x =  [slice(None), slice(None)] # Batch_size, channels
    slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
    slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed

    
    # out_fft[slices_x] = self._contract(x[slices_x], weight, separable=False)
    out_fft = self._contract(x[slices_x], weight, separable=False)

    # out_fft = torch.view_as_real(out_fft.contiguous())
    # print(out_fft.shape)
    # out_fft = _ReduceScatterToTensorParallelRegion.apply(out_fft)
    # print(out_fft.shape)
    # out_fft = torch.view_as_complex(out_fft)
    # out_fft = out_fft.contiguous()
    # dist.all_reduce(out_fft)
    # out_fft = out_fft[:, channels * self.local_rank : channels * (self.local_rank + 1), :, :]

    out_fft = _ReduceFromModelParallelRegion.apply(out_fft)

    # print(out_fft[0][0])
    # exit(0)


    if self.output_scaling_factor is not None and output_shape is None:
        mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.output_scaling_factor[indices])])

    if output_shape is not None:
        mode_sizes = output_shape

    if self.order > 1:
        out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])
    x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)

    if self.bias is not None:
        x = x + self.bias[indices, ...][channels * local_rank : channels * (local_rank + 1)]
        # x = x + self.bias[indices, ...]

    return x
import torch.distributed as dist
import torch
import math
from typing import List, Optional, Union

def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()
    
class _AllToAll(torch.autograd.Function):
    """All-to-all communication.
    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )

def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the first dimension."""
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=group
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=1).contiguous()
    return output
    
def _reduce_scatter_along_first_dim(input_, input_split_sizes=None):
    """Reduce-scatter the input tensor across model parallel group.
    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"
        dim_size[0] = dim_size[0] // world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed._reduce_scatter_base(
            output, input_.contiguous(), group=group
        )
    else:
        rank = torch.distributed.get_rank(group)
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))
        output = torch.empty_like(input_tensor_list[rank])
        torch.distributed.reduce_scatter(
            output, input_tensor_list, group=group
        )
    return output
    
def _reduce_scatter_along_last_dim(input_):
    """Reduce-scatter tensors on the last dimension."""
    
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    
    target_shape = list(input_.size())
    target_shape[1] = target_shape[1] // world_size
    input_ = input_.reshape(-1, input_.shape[1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = _reduce_scatter_along_first_dim(concat_tensor).reshape(target_shape)
    return output
    
class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_last_dim(input_)
    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _reduce_scatter_along_last_dim(input_)
    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_last_dim(grad_output)
        
def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    # All-reduce.
    input_ = input_.contiguous()
    torch.distributed.all_reduce(input_, group=group)
    return input_
def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    
def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
    
def split_tensor(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False, dim: int = -1
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.
        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    dim_size = divide(tensor.size()[dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, dim_size, dim=dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list
    
def _split_along_last_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    input_list = split_tensor(input_, world_size, dim=1)
    output = input_list[rank].contiguous()
    return output
    
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _split_along_last_dim(_reduce(input_))
    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _split_along_last_dim(_reduce(input_))
    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_last_dim(grad_output)
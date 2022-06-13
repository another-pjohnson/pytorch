import bisect
import contextlib
from enum import Enum, auto
import functools
import itertools
import math
from types import FunctionType
from typing import Any, Callable, Dict, Iterator, List, Tuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
)
from torch.utils._pytree import tree_map


def _sharding_spec_to_offsets(
    sharding_spec: ShardingSpec, tensor_numel: int, world_size: int
) -> List[int]:
    r"""
    Translates the sharding spec to a list of offsets along dim 0. If the
    sharding spec is ChunkShardingSpec, only the ``dim`` is used and the
    placement is not used.
    """
    offsets: List[int] = []
    if isinstance(sharding_spec, EnumerableShardingSpec):
        for shard in sharding_spec.shards:
            offsets.append(shard.shard_offsets[0])
    elif isinstance(sharding_spec, ChunkShardingSpec):
        assert sharding_spec.dim == 0
        chunk_size = math.ceil(tensor_numel / world_size)
        if chunk_size == 1:
            offsets = [
                rank if rank < tensor_numel else tensor_numel
                for rank in range(world_size)
            ]
        else:
            offsets = [chunk_size if rank > 0 else 0 for rank in range(world_size)]
            offsets = list(itertools.accumulate(offsets))
    else:
        raise ValueError(f"Un-recognized sharding spec type {type(sharding_spec)}.")

    return offsets


def _offsets_to_split_sizes(
    input_offsets: List[int],
    output_offsets: List[int],
    tensor_numel: int,
    world_size: int,
    my_rank: int,
) -> Tuple[List[int], List[int]]:
    r"""
    Given the shard offsets for each rank of the input tensor and output tensor,
    this API returns the corresponding split sizes that can be passed to
    all_to_all_single().
    """

    def _get_interval(offsets):
        if my_rank != world_size - 1:
            return offsets[my_rank], offsets[my_rank + 1] - 1
        else:
            return offsets[my_rank], tensor_numel - 1

    def _offsets_to_sizes(offsets, begin, end):
        sizes = []
        for i, offset in enumerate(offsets):
            next_offset = offsets[i + 1] if i < len(offsets) - 1 else end + 1
            sizes.append(
                (next_offset - offset)
                - max(begin - offset, 0)
                - max(next_offset - end - 1, 0)
            )
        return sizes

    def _convert(from_offsets, to_offsets, split_sizes):
        begin, end = _get_interval(from_offsets)
        to_begin_rank = bisect.bisect(to_offsets, begin) - 1
        to_end_rank = bisect.bisect(to_offsets, end) - 1
        _split_sizes = _offsets_to_sizes(
            to_offsets[to_begin_rank : to_end_rank + 1], begin, end
        )
        split_sizes[to_begin_rank : to_end_rank + 1] = _split_sizes

    input_split_sizes = [0 for _ in range(world_size)]
    output_split_sizes = [0 for _ in range(world_size)]
    _convert(input_offsets, output_offsets, input_split_sizes)
    _convert(output_offsets, input_offsets, output_split_sizes)

    return input_split_sizes, output_split_sizes


def _reshard_flatten_tensor(
    input_tensor: ShardedTensor,
    output_spec: ShardingSpec,
    world_size: int,
    my_rank: int,
    device: torch.device,
    process_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """
    Resharded a sharded flatten tensor, this is used by FSDP to do sharded
    state_dict. But the functionaility is not supported by ShardedTensor.
    This API is designed to be used for FSDP; therefore this API supports only
    1-D ShardedTensor (hence the naming, reshard_flatten_tensor).

    This API uses the ChunkShardingSpec and EnumerableShardingSpec from
    torch.distributed.sharding_spec but ignores the placement field in
    ChunkShardingSpec, as the placement requires the callees understand the
    number of GPUs per node. The API simply uses the semantics of the sharding
    specs.

    Args:
        input_tensor (ShardedTensor): the original ShardedTensor. Must be 1D.
        output_spec (ShardingSpec): the sharding spect for the output tensor.
        world_size (int): total trainer count.
        my_rank (int): the rank for this trainer.

    Returns:
        The local shard for the new ShardedTensor.
    """

    input_spec = input_tensor.sharding_spec()
    size = input_tensor.size()
    if isinstance(size, int):
        raise ValueError("The input tensor has no dimensions.")
    tensor_numel = size.numel()
    input_offsets = _sharding_spec_to_offsets(input_spec, tensor_numel, world_size)
    output_offsets = _sharding_spec_to_offsets(output_spec, tensor_numel, world_size)
    input_split_sizes, output_split_sizes = _offsets_to_split_sizes(
        input_offsets, output_offsets, tensor_numel, world_size, my_rank
    )
    output_size = sum(output_split_sizes)
    local_shard = torch.empty(output_size, dtype=input_tensor.dtype, device=device)
    dist.all_to_all_single(
        local_shard,
        input_tensor.local_shards()[0].tensor,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=process_group,
    )
    return local_shard


def _all_gather_sharded_tensor(
    sharded_tensor: ShardedTensor, pg: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    if pg is None:
        pg = distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    shards = sharded_tensor.local_shards()
    local_tensor = shards[0].tensor.flatten()
    dim_0_size = sharded_tensor.size()[0]  # type: ignore[index]
    tensor_numel = sharded_tensor.size().numel()  # type: ignore[union-attr]
    chunk_size = math.ceil(dim_0_size / world_size) * tensor_numel // dim_0_size
    num_padding = chunk_size - local_tensor.numel()
    if num_padding > 0:
        local_tensor = F.pad(local_tensor, [0, num_padding])
    tensor = torch.empty(chunk_size * world_size, dtype=local_tensor.dtype).cuda()
    dist._all_gather_base(tensor, local_tensor, group=pg)
    return tensor.narrow(0, 0, tensor_numel).reshape(sharded_tensor.size())


def _gather_state_dict(
    state_dict: Dict[str, Any],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Given a state_dict, this API gathers all the ShardedTensor in the state_dict
    to the output_rank, and creates a new state_dict which the values are either
    the gathered tensors (rank == output_rank) or None (rank != output_rank).
    """
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, ShardedTensor):
            """
            # TODO: It is unclear why the following implementation cause a
            # timeout in some unittests on AWS servers but not other environment.
            output_tensor = (
                torch.empty(tensor.shape, dtype=tensor.dtype).cuda()
                if curr_rank == output_rank
                else None
            )
            tensor.gather(output_rank, output_tensor)
            """
            output_tensor = _all_gather_sharded_tensor(tensor, pg)
            tensor = output_tensor
        new_state_dict[key] = tensor
    return new_state_dict

@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard

class BypassState(Enum):
    WRAPPER_ONLY = auto()
    WRAPPED_ONLY = auto()
    AUTOMATIC = auto()

class bypass_state(object):
    # only support @decorator, and not @decorator()
    def __init__(self, *args):
        self.new_state = BypassState.WRAPPER_ONLY
        self.old_value = None
        self.func = None
        if len(args) > 0:
            if isinstance(args[0], Callable):
                self.func = args[0]
                if len(args) == 2:
                    self.new_state = args[1]
            else:
                if len(args) != 1:
                    raise ValueError("Only expected a single argument of type BypassState")
                else:
                    self.new_state = args[0]

    def __enter__(self):
        self.old_value = ShadowTensor.bypass_state
        ShadowTensor.bypass_state = self.new_state

    def __exit__(self, *args):
        ShadowTensor.bypass_state = self.old_value

    def __call__(self, *args, **kwargs):
        def wrapper(*args, **kwargs):
            with self:
                result = self.func(*args, **kwargs)
                return result

        if self.func is None:
            self.func = args[0]
            return wrapper
        else:
            return wrapper(*args, **kwargs)

    def __get__(self, instance, owner):
        return functools.partial(self.__call__, instance)

class ShadowTensor(torch.Tensor):
    bypass_state: BypassState = BypassState.AUTOMATIC
    @staticmethod
    def __new__(
        cls, tensor: torch.Tensor
    ):
        r = super(ShadowTensor, cls)._make_wrapper_subclass(cls, tensor.shape, dtype=tensor.dtype, requires_grad=tensor.requires_grad, device=tensor.device)  # type: ignore
        object.__setattr__(r, "tensor", tensor)
        return r

    def __init__(self, *args, **kwargs):
        pass

    def __getattribute__(self, name: str) -> Any:
        #print(f"__getattribute__ for name: {name}")
        local_state = ShadowTensor.bypass_state
        # local state overrides global state
        if "_wrapper_" in name:
            name = name.replace("_wrapper_", "")
            local_state = BypassState.WRAPPER_ONLY
        elif "_wrapped_" in name:
            name = name.replace("_wrapped_", "")
            local_state = BypassState.WRAPPED_ONLY
        if local_state == BypassState.WRAPPER_ONLY:
            return object.__getattribute__(self, name)
        elif local_state == BypassState.WRAPPED_ONLY:
            return object.__getattribute__(self, "tensor").__getattribute__(name)
        #elif name in ["__torch_function__", "__torch_dispatch__", "_grad", "grad", "grad_fn"]:
        elif name in ["__torch_function__", "__torch_dispatch__", "grad_fn"]:
            return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self, "tensor").__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        #print(f"__setattr__ for name: {name}")
        local_state = ShadowTensor.bypass_state
        # local state overrides global state
        if "_wrapper_" in name:
            name = name.replace("_wrapper_", "")
            local_state = BypassState.WRAPPER_ONLY
        elif "_wrapped_" in name:
            name = name.replace("_wrapped_", "")
            local_state = BypassState.WRAPPED_ONLY
        if local_state == BypassState.WRAPPER_ONLY:
            object.__setattr__(self, name, value)
        elif local_state == BypassState.WRAPPED_ONLY:
            object.__getattribute__(self, "tensor").__setattr__(name, value)
        #elif name in ["_grad", "grad"]:
            #object.__setattr__(self, name, value)
        else:
            object.__getattribute__(self, "tensor").__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        #print(f"__delattr__ for name: {name}")
        local_state = ShadowTensor.bypass_state
        # local state overrides global state
        if "_wrapper_" in name:
            name = name.replace("_wrapper_", "")
            local_state = BypassState.WRAPPER_ONLY
        elif "_wrapped_" in name:
            name = name.replace("_wrapped_", "")
            local_state = BypassState.WRAPPED_ONLY
        if local_state == BypassState.WRAPPER_ONLY:
            object.__delattr__(self, name)
        elif local_state == BypassState.WRAPPED_ONLY:
            object.__getattribute__(self, "tensor").__delattr__(name)
        else:
            object.__getattribute__(self, "tensor").__delattr__(name)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        func_name = func.overloadpacket.__name__
        #print(f"ShadowTensor dispatch for func: {func_name}")
        def unwrap(e: Any) -> torch.Tensor:
            if isinstance(e, ShadowTensor):
                with bypass_state():
                    t = e.tensor
                return t
            else:
                return e
        def wrap(e):
            if func_name.startswith("split") or func_name.startswith("view"):
                return ShadowTensor(e)
            else:
                return e

        # no_dispatch is only needed if you use enable_python_mode.
        # It prevents infinite recursion.
        with no_dispatch():
            r = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        return r

class ShadowParameter(ShadowTensor, torch.nn.Parameter):
    def __new__(cls, tensor):
        param = tensor
        if not isinstance(tensor, torch.nn.Parameter):
            param = torch.nn.Parameter(tensor)
        return super(ShadowParameter, cls).__new__(cls, param)


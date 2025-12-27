# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""实用函数。"""

import functools
from typing import Any, Sequence, TypeVar

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import rlax

from disco_rl import types

_T = TypeVar('_T')
_SpecsT = TypeVar('_SpecsT')


def shard_across_devices(data: _T, devices: Sequence[jax.Device]) -> _T:
  """跨设备分片数据。

  Args:
    data: 要分片的数据。
    devices: 设备序列。

  Returns:
    分片后的数据。
  """
  num_shards = len(devices)
  leaves, treedef = jax.tree.flatten(data)
  split_leaves = [np.split(leaf, num_shards, axis=0) for leaf in leaves]
  flat_shards = ((leaf[i] for leaf in split_leaves) for i in range(num_shards))
  data_shards = [jax.tree.unflatten(treedef, shard) for shard in flat_shards]
  return jax.device_put_sharded(data_shards, devices)


def gather_from_devices(data: _T) -> _T:
  """从设备收集数据。

  Args:
    data: 分布在设备上的数据。

  Returns:
    收集到的数据。
  """
  return jax.tree.map(
      lambda x: x.reshape((-1, *x.shape[2:])), jax.device_get(data)
  )


def batch_lookup(
    table: chex.Array, index: chex.Array, num_dims: int = 2
) -> chex.Array:
  """批量查找表中的值。

  Args:
    table: 查找表。
    index: 索引。
    num_dims: 批处理维度数量。

  Returns:
    查找到的值。
  """

  def _lookup(table: chex.Array, index: chex.Array) -> chex.Array:
    return jax.vmap(lambda x, i: x[i])(table, index)

  if index is not None:
    index = index.astype(jnp.int32)
  return hk.BatchApply(_lookup, num_dims=num_dims)(table, index)


def broadcast_specs(specs: _SpecsT, n: int, replace: bool = False) -> _SpecsT:
  """将 `n` 预置到规范的形状中。

  Args:
    specs: 要广播的规范。
    n: 要预置到形状的值。
    replace: 是否替换或预置第一个维度。

  Returns:
    广播后的规范。

  Raises:
    ValueError: 如果不支持规范类型。
  """
  f_i = 1 if replace else 0

  def _prepend(s: types.ArraySpec | types.specs.Array):
    if isinstance(s, types.specs.Array):
      return s.replace(shape=(n,) + s.shape[f_i:])
    elif isinstance(s, types.ArraySpec):
      return types.ArraySpec(shape=(n,) + s.shape[f_i:], dtype=s.dtype)
    else:
      raise ValueError(f'Unsupported spec type: {type(s)}')

  return jax.tree.map(_prepend, specs)


def tree_stack(
    elems: Sequence[chex.ArrayTree], axis: int = 0
) -> chex.ArrayTree:
  """将树序列堆叠成单个树。

  Args:
    elems: 树序列。
    axis: 堆叠轴。

  Returns:
    堆叠后的树。
  """
  return jax.tree.map(lambda *xs: jnp.stack(xs, axis=axis), *elems)


def cast_to_single_precision(
    tree_like: _T, cast_ints: bool = True, host_data: bool = False
) -> _T:
  """将数据转换为单精度。

  Args:
    tree_like: 类似树的数据结构。
    cast_ints: 是否转换整数。
    host_data: 是否为主机数据。

  Returns:
    转换后的数据。
  """
  if host_data:

    def conditional_cast(x):
      if isinstance(x, (np.ndarray, jnp.ndarray)):
        if np.issubdtype(x.dtype, np.floating) or jnp.issubdtype(
            x.dtype, jnp.floating
        ):
          if x.dtype != np.float32:
            x = x.astype(np.float32)
        elif cast_ints and x.dtype == np.int64:
          x = x.astype(np.int32)
      return x

    return jax.tree.map(conditional_cast, tree_like)
  else:
    return jmp.cast_to_full(tree_like)


def get_num_actions_from_spec(spec: types.ActionSpec) -> int:
  """从动作规范返回动作数量。

  Args:
    spec: 动作规范。

  Returns:
    动作数量。
  """
  return spec.maximum - spec.minimum + 1


def get_logits_specs(
    spec: types.ActionSpec, with_batch_dim: bool = False
) -> types.ArraySpec:
  """为提供的规范提取 Logits 形状树。

  Args:
    spec: 动作规范。
    with_batch_dim: 是否包含批次维度。

  Returns:
    Logits 规范。
  """
  if with_batch_dim:
    spec = spec.replace(shape=spec.shape[1:])

  return types.ArraySpec((get_num_actions_from_spec(spec),), np.float32)


def zeros_like_spec(spec: Any, prepend_shape: tuple[int, ...] = ()):
  """从 `spec` 返回一个零树。

  Args:
    spec: `array_like` 或规范的树。
    prepend_shape: 要预置到形状的整数元组。

  Returns:
    零数组树。
  """
  return jax.tree.map(
      lambda spec: np.zeros(shape=prepend_shape + spec.shape, dtype=spec.dtype),
      spec,
  )


def differentiable_policy_gradient_loss(
    logits_t: chex.Array, a_t: chex.Array, adv_t: chex.Array, backprop: bool
) -> chex.Array:
  """计算具有可微优势的策略梯度损失。

  `rlax.policy_gradient_loss()` 的优化版本。

  Args:
    logits_t: 未归一化的动作偏好序列（形状: [..., |A|]）。
    a_t: 从偏好 `logits_t` 采样的动作序列。
    adv_t: 执行动作 `a_t` 观察到或估计的优势。
    backprop: 是否使损失可微。

  Returns:
    损失（每步），其梯度对应于策略梯度更新。
  """

  chex.assert_type([logits_t, a_t, adv_t], [float, int, float])

  log_pi_a = rlax.batched_index(jax.nn.log_softmax(logits_t), a_t)
  if backprop:
    loss_per_step = -log_pi_a * adv_t
  else:
    loss_per_step = -log_pi_a * jax.lax.stop_gradient(adv_t)
  return loss_per_step


class MovingAverage:
  """跟踪 EMA 并将其用于归一化的函数。"""

  def __init__(
      self,
      example_tree: chex.ArrayTree,
      decay: float = 0.999,
      eps: float = 1e-6,
  ):
    """初始化移动平均参数。

    Args:
      example_tree: 稍后传递给 `update_state` 的结构示例。
      decay: 矩的衰减。即学习率是 `1 - decay`。
      eps: 用于归一化的 Epsilon。
    """
    self._example_tree = example_tree
    self._decay = decay
    self._eps = eps

  def init_state(self) -> types.EmaState:
    """初始化状态。

    Returns:
      EMA 状态。
    """
    zeros = jax.tree.map(
        lambda x: jnp.zeros((), jnp.float32), self._example_tree
    )
    return types.EmaState(  # pytype: disable=wrong-arg-types  # jnp-type
        moment1=zeros,
        moment2=zeros,
        decay_product=jnp.ones([], dtype=jnp.float32),
    )

  def update_state(
      self,
      tree_like: chex.ArrayTree,
      state: types.EmaState,
      pmean_axis_name: str | None,
  ) -> types.EmaState:
    """更新移动平均统计信息。

    Args:
      tree_like: 类似树的数据。
      state: 当前 EMA 状态。
      pmean_axis_name: 并行平均轴名称。

    Returns:
      更新后的 EMA 状态。
    """
    squared_tree = jax.tree.map(jnp.square, tree_like)

    def _update(
        moment: chex.Array,
        value: chex.Array,
        pmean_axis_name: str | None = None,
    ) -> chex.Array:
      mean = jnp.mean(value)
      # Compute the mean across all learner devices involved in the `pmap`.
      if pmean_axis_name is not None:
        mean = jax.lax.pmean(mean, axis_name=pmean_axis_name)
      return self._decay * moment + (1.0 - self._decay) * mean

    update_fn = functools.partial(_update, pmean_axis_name=pmean_axis_name)
    moment1 = jax.tree.map(update_fn, state.moment1, tree_like)
    moment2 = jax.tree.map(update_fn, state.moment2, squared_tree)
    return types.EmaState(
        moment1=moment1,
        moment2=moment2,
        decay_product=state.decay_product * self._decay,
    )

  def _compute_moments(
      self, state: types.EmaState
  ) -> tuple[chex.ArrayTree, chex.ArrayTree]:
    """计算矩，应用 Adam 优化器中的 0 去偏差。

    Args:
      state: EMA 状态。

    Returns:
      均值和方差的元组。
    """

    # Factor to account for initializing moments with 0s.
    debias = 1.0 / (1 - state.decay_product)

    # Debias mean.
    mean = jax.tree.map(lambda m1: m1 * debias, state.moment1)

    # Estimate zero-centered debiased variance; clip negative values to
    # safeguard against numerical errors.
    variance = jax.tree.map(
        lambda m2, m: jnp.maximum(0.0, m2 * debias - jnp.square(m)),
        state.moment2,
        mean,
    )
    return mean, variance

  def normalize(
      self,
      value: chex.ArrayTree,
      state: types.EmaState,
      subtract_mean: bool = True,
      root_eps: float = 1e-12,
  ) -> chex.ArrayTree:
    """通过除以二阶矩并减去均值进行归一化。

    Args:
      value: 要归一化的值。
      state: EMA 状态。
      subtract_mean: 是否减去均值。
      root_eps: 用于根号稳定性的 epsilon。

    Returns:
      归一化后的值。
    """

    def _normalize(mean, var, val):
      # Two epsilons, instead of one, are used for numerical stability when
      # backpropagating through the normalization (as in optax.scale_by_adam).
      if subtract_mean:
        return (val - mean) / (jnp.sqrt(var + root_eps) + self._eps)
      else:
        return val / (jnp.sqrt(var + root_eps) + self._eps)

    mean, variance = self._compute_moments(state)
    return jax.tree.map(_normalize, mean, variance, value)

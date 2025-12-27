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

"""元网络输入的变换。"""

import functools
from typing import Callable, Sequence

import chex
import haiku as hk
import immutabledict
import jax
import jax.numpy as jnp
import rlax

from disco_rl import utils


class InputTransform:
  """输入变换的基类。"""

  def __call__(
      self, x, actions: chex.Array, policy: chex.Array, axis: str | None
  ) -> chex.Array:
    """应用变换。

    Args:
      x: 输入数组。
      actions: 动作。
      policy: 策略。
      axis: 轴。

    Returns:
      变换后的数组。
    """
    raise NotImplementedError


class SelectByAction(InputTransform):
  """通过动作选择输入的变换。"""

  def __call__(self, x, actions, policy, axis):
    """应用变换。

    Args:
      x: 输入数组。
      actions: 动作。
      policy: 策略。
      axis: 轴。

    Returns:
      变换后的数组。
    """
    del policy, axis
    chex.assert_rank(actions, 2)
    chex.assert_tree_shape_prefix([x, actions], actions.shape)
    return utils.batch_lookup(x, actions)


class PiWeightedAvg(InputTransform):
  """策略加权平均变换。"""

  def __call__(self, x, actions, policy, axis):
    """应用变换。

    Args:
      x: 输入数组。
      actions: 动作。
      policy: 策略。
      axis: 轴。

    Returns:
      变换后的数组。
    """
    del actions, axis
    chex.assert_rank(x, 4)
    chex.assert_rank(policy, 3)
    chex.assert_tree_shape_prefix([x, policy], policy.shape)
    return jnp.sum(x * jnp.expand_dims(policy, -1), axis=2)


class Normalize(InputTransform, hk.Module):
  """归一化变换。"""

  def __call__(self, x, actions, policy, axis):
    """应用变换。

    Args:
      x: 输入数组。
      actions: 动作。
      policy: 策略。
      axis: 轴。

    Returns:
      变换后的数组。
    """
    del actions, policy
    assert x.ndim >= 2  # Average over B, T
    return EmaNorm(
        decay_rate=0.99, eps=1e-6, axis=(0, 1), cross_replica_axis=axis
    )(x)


class EmaNorm(hk.Module):
  """通过均值和方差的 EMA 估计进行归一化。"""

  def __init__(
      self,
      decay_rate: float,
      eps: float = 1e-6,
      eps_root: float = 1e-12,
      axis: Sequence[int] | None = None,
      cross_replica_axis: str | Sequence[str] | None = None,
      cross_replica_axis_index_groups: Sequence[Sequence[int]] | None = None,
      data_format: str = 'channels_last',
      name: str | None = None,
  ):
    """构建 EmaNorm 模块。基于 hk.BatchNorm。

    Args:
      decay_rate: EMA 的衰减率。
      eps: 避免除以零方差的小 epsilon。默认为 ``1e-6``。
      eps_root: 辅助 metagrad 稳定性的小 epsilon。默认为 ``1e-12``。
      axis: 要归约的轴。默认值（``None``）表示除了通道轴之外的所有轴都应归一化。
        否则，这是一个将计算归一化统计信息的轴索引列表。
      cross_replica_axis: 如果不为 ``None``，则它应该是一个字符串（或字符串序列），
        表示在 jax 映射（例如 ``jax.pmap`` 或 ``jax.vmap``）中运行此模块的轴名称。
        提供此参数意味着将在命名轴上的所有副本之间计算批处理统计信息。
      cross_replica_axis_index_groups: 指定设备如何分组。仅在 ``jax.pmap`` 集合中有效。
      data_format: 输入的数据格式。可以是 ``channels_first``, ``channels_last``,
        ``N...C`` 或 ``NC...``。默认为 ``channels_last``。参见 :func:`get_channel_index`。
      name: 模块名称。
    """
    super().__init__(name=name)

    self.eps = eps
    self.eps_root = eps_root
    self.axis = axis
    self.cross_replica_axis = cross_replica_axis
    self.cross_replica_axis_index_groups = cross_replica_axis_index_groups
    self.channel_index = hk.get_channel_index(data_format)
    self.m1_ema = hk.ExponentialMovingAverage(decay_rate, name='m1_ema')
    self.m2_ema = hk.ExponentialMovingAverage(decay_rate, name='m2_ema')

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """更新 EMA 状态（始终在 `train` 模式下），返回归一化后的输入。

    Args:
      inputs: 输入数组。

    Returns:
      归一化后的数组。
    """

    channel_index = self.channel_index
    if channel_index < 0:
      channel_index += inputs.ndim

    if self.axis is not None:
      axis = self.axis
    else:
      axis = [i for i in range(inputs.ndim) if i != channel_index]

    mean = jnp.mean(inputs, axis, keepdims=True)
    mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True)

    if self.cross_replica_axis:
      mean = jax.lax.pmean(
          mean,
          axis_name=self.cross_replica_axis,
          axis_index_groups=self.cross_replica_axis_index_groups,
      )
      mean_of_squares = jax.lax.pmean(
          mean_of_squares,
          axis_name=self.cross_replica_axis,
          axis_index_groups=self.cross_replica_axis_index_groups,
      )
    self.m1_ema(mean)
    self.m2_ema(mean_of_squares)

    ema_m1 = self.m1_ema.average.astype(inputs.dtype)
    ema_m2 = self.m2_ema.average.astype(inputs.dtype)

    ema_var = jnp.maximum(ema_m2 - jnp.square(ema_m1), 0.0)

    eps = jax.lax.convert_element_type(self.eps, ema_var.dtype)
    eps_root = jax.lax.convert_element_type(self.eps_root, ema_var.dtype)
    return (inputs - ema_m1) / (jnp.sqrt(ema_var + eps_root) + eps)


def td_pair(x):
  """连接 t 和 t+1 时刻的输入，以便计算类似 TD 误差的量。

  Args:
    x: 输入数组。

  Returns:
    连接后的数组。
  """
  # Concat inputs at t and t+1 to ease calculation of TD-error-like quantities
  return jnp.concatenate([x[:-1], x[1:]], axis=-1)


def tx_factory(tx_call: Callable[[chex.Array], chex.Array]):
  """包装单参数变换，使它们都具有相同的接口。

  Args:
    tx_call: 单参数变换函数。

  Returns:
    包装后的变换函数构建器。
  """

  # The dummy and wrap allows the fn to be called with the same interface as the
  # modules: tx()(x, actions, policy, axis)
  def dummy_builder():
    def _wrap_transform(x, actions, policy, axis):
      del actions, policy, axis
      return tx_call(x)

    return _wrap_transform

  return dummy_builder


_TRANSFORM_FNS = immutabledict.immutabledict({
    'identity': lambda x: x,
    'softmax': jax.nn.softmax,
    'max_a': functools.partial(jnp.max, axis=2),
    'stop_grad': jax.lax.stop_gradient,
    'clip': functools.partial(jnp.clip, a_min=-2.0, a_max=2.0),
    'sign': jnp.sign,
    'drop_last': lambda x: x[:-1],
    'td_pair': td_pair,
    'sign_log': rlax.signed_logp1,
    'sign_hyp': rlax.signed_hyperbolic,
    'masks_to_discounts': lambda x: 1.0 - x,
})

TRANSFORMS = immutabledict.immutabledict({
    'select_a': SelectByAction,
    'pi_weighted_avg': PiWeightedAvg,
    'normalize': Normalize,
    **{name: tx_factory(tx) for name, tx in _TRANSFORM_FNS.items()},
})

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

"""更新规则使用的元网络。"""

import functools
from typing import Any, Callable, Mapping, Sequence

import chex
import haiku as hk
from haiku import initializers as hk_init
import jax
from jax import lax
from jax import numpy as jnp

from disco_rl import types
from disco_rl import utils
from disco_rl.update_rules import input_transforms


class MetaNet(hk.Module):
  """元网络基类。"""

  def __call__(
      self,
      inputs: types.UpdateRuleInputs,
      axis_name: str | None,
  ) -> types.UpdateRuleOuts:
    """生成更新规则代理损失所需的输出。

    Args:
      inputs: 更新规则输入。
      axis_name: 轴名称。

    Returns:
      更新规则输出。
    """
    raise NotImplementedError


class LSTM(MetaNet):
  """带有 LSTM 的元网络。"""

  def __init__(
      self,
      hidden_size: int,
      embedding_size: Sequence[int],
      prediction_size: int,
      meta_rnn_kwargs: Mapping[str, Any],
      input_option: types.MetaNetInputOption,
      policy_channels: Sequence[int] = (16, 2),
      policy_target_channels: Sequence[int] = (128,),
      policy_target_stddev: float | None = None,
      output_stddev: float | None = None,
      aux_stddev: float | None = None,
      state_stddev: float | None = None,
      name: str | None = None,
  ) -> None:
    """初始化 LSTM 元网络。

    Args:
      hidden_size: 隐藏层大小。
      embedding_size: 嵌入大小。
      prediction_size: 预测大小。
      meta_rnn_kwargs: 元 RNN 参数。
      input_option: 输入选项。
      policy_channels: 策略通道。
      policy_target_channels: 策略目标通道。
      policy_target_stddev: 策略目标标准差。
      output_stddev: 输出标准差。
      aux_stddev: 辅助标准差。
      state_stddev: 状态标准差。
      name: 网络名称。
    """
    super().__init__(name=name)
    self._hidden_size = hidden_size
    self._embedding_size = embedding_size
    self._prediction_size = prediction_size
    self._input_option = input_option
    self._output_init = _maybe_get_initializer(output_stddev)
    self._aux_init = _maybe_get_initializer(aux_stddev)
    self._state_init = _maybe_get_initializer(state_stddev)
    self._policy_target_init = _maybe_get_initializer(policy_target_stddev)
    self._policy_channels = policy_channels
    self._policy_target_channels = policy_target_channels
    self._meta_rnn_core: 'MetaLSTM' = MetaLSTM(
        **meta_rnn_kwargs, input_option=input_option
    )

  def __call__(
      self, inputs: types.UpdateRuleInputs, axis_name: str | None
  ) -> types.UpdateRuleOuts:
    """执行元网络。

    Args:
      inputs: 更新规则输入。
      axis_name: 轴名称。

    Returns:
      更新规则输出。
    """
    # Initialize or extract the meta RNN core state.
    initial_meta_rnn_state = self._meta_rnn_core.initial_state()
    meta_rnn_state = hk.get_state(
        'meta_rnn_state',
        shape=jax.tree.map(lambda t: t.shape, initial_meta_rnn_state),
        dtype=jax.tree.map(lambda t: t.dtype, initial_meta_rnn_state),
        init=lambda *_: initial_meta_rnn_state,
    )
    assert isinstance(meta_rnn_state, hk.LSTMState)

    # Inputs have shapes of [T+1, B, ...]
    logits = inputs.agent_out['logits']
    assert isinstance(logits, chex.Array)
    _, batch_size, num_actions = logits.shape

    # Construct inputs for the meta network.
    y_net = _batch_mlp(self._embedding_size, num_dims=2)
    z_net = _batch_mlp(self._embedding_size, num_dims=3)
    policy_net = _conv1d_net(self._policy_channels)
    x, policy_emb = _construct_input(
        inputs,
        y_net=y_net,
        z_net=z_net,
        policy_net=policy_net,
        input_option=self._input_option,
        axis_name=axis_name,
    )

    # Unroll the per-trajectory RNN core in reverse direction for bootstrapping.
    per_trajectory_rnn_core = hk.ResetCore(hk.LSTM(self._hidden_size))
    should_reset_bwd = inputs.should_reset_mask_bwd[:-1]  # [T, B]
    x, _ = hk.dynamic_unroll(  # [T, B, H]
        per_trajectory_rnn_core,
        (x, should_reset_bwd),
        per_trajectory_rnn_core.initial_state(batch_size=batch_size),
        reverse=True,
    )

    # Perform multipl-ve interaction with the (per-lifetime) meta RNN's outputs.
    x = _multiplicative_interaction(
        x=x,
        y=self._meta_rnn_core.output(meta_rnn_state),
        initializer=self._state_init,
    )

    # Compute an additional input embedding for the meta network unrolling.
    meta_input_emb = hk.BatchApply(hk.Linear(1, w_init=self._output_init))(
        x
    )  # [T, B, 1]

    # Compute the y, z targets.
    y_hat = hk.BatchApply(
        hk.Linear(self._prediction_size, w_init=self._aux_init)
    )(x)
    z_hat = hk.BatchApply(
        hk.Linear(self._prediction_size, w_init=self._aux_init)
    )(x)

    # Compute the policy target (pi).
    w = jnp.repeat(jnp.expand_dims(x, 2), num_actions, axis=2)  # [T, B, A, H]
    w = jnp.concatenate([w, policy_emb], axis=-1)  # [T, B, A, H+C]
    w = _conv1d_net(self._policy_target_channels)(w)  # [T, B, A, O]
    w = hk.BatchApply(hk.Linear(1, w_init=self._policy_target_init))(
        w
    )  # [T, B, A, 1]
    pi_hat = jnp.squeeze(w, -1)  # [T, B, A]

    # Set the meta network outputs.
    meta_out = dict(pi=pi_hat, y=y_hat, z=z_hat, meta_input_emb=meta_input_emb)

    # Unroll the meta RNN core and update its state.
    new_meta_rnn_state = self._meta_rnn_core.unroll(
        inputs, meta_out, meta_rnn_state, axis_name=axis_name
    )
    hk.set_state('meta_rnn_state', new_meta_rnn_state)

    return meta_out


class MetaLSTM(hk.Module):
  """处理轨迹和元目标的元 LSTM，贯穿代理的整个生命周期。"""

  def __init__(
      self,
      input_option: types.MetaNetInputOption,
      policy_channels: Sequence[int],
      embedding_size: Sequence[int],
      pred_embedding_size: Sequence[int],
      hidden_size: int,
  ):
    """初始化 MetaLSTM。

    Args:
      input_option: 输入选项。
      policy_channels: 策略通道。
      embedding_size: 嵌入大小。
      pred_embedding_size: 预测嵌入大小。
      hidden_size: 隐藏层大小。
    """
    super().__init__()
    self._input_option = input_option
    self._hidden_size = hidden_size
    self._embedding_size = embedding_size
    self._policy_channels = policy_channels
    self._pred_embedding_size = pred_embedding_size
    self._core_constructor = lambda: hk.LSTM(self._hidden_size)

  def unroll(
      self,
      inputs: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts,
      state: hk.LSTMState,
      axis_name: str | None,
  ) -> hk.LSTMState:
    """给定 rollout 和 rnn_state 更新 meta_state。

    Args:
      inputs: 更新规则输入。
      meta_out: 更新规则输出。
      state: LSTM 状态。
      axis_name: 轴名称。

    Returns:
      新的 LSTM 状态。
    """

    # Get meta inputs.
    y_net = _batch_mlp(self._pred_embedding_size, num_dims=2)
    z_net = _batch_mlp(self._pred_embedding_size, num_dims=3)
    policy_net = _conv1d_net(self._policy_channels)
    meta_inputs, _ = _construct_input(  # [T, B, ...]
        inputs,
        y_net=y_net,
        z_net=z_net,
        policy_net=policy_net,
        input_option=self._input_option,
        axis_name=axis_name,
    )
    input_list = [
        meta_inputs,
        meta_out['meta_input_emb'],
        y_net(jax.nn.softmax(meta_out['y'])),
    ]

    # Concatenate & embed all inputs.
    x = jnp.concatenate(input_list, axis=-1)  # [T, B, ...]
    x = _batch_mlp(self._embedding_size)(x)  # [T, B, E]

    # Apply average pooling over batch-time dimensions.
    x_avg = x.mean(axis=(0, 1))  # [E]
    if axis_name is not None:
      x_avg = jax.lax.pmean(x_avg, axis_name=axis_name)

    # Unroll the meta RNN core and update its state.
    core = self._core_constructor()
    _, new_state = core(x_avg, state)
    return new_state

  def initial_state(self) -> hk.LSTMState:
    """返回初始 rnn_state。

    Returns:
      初始 LSTM 状态。
    """
    return self._core_constructor().initial_state(batch_size=None)

  def output(self, state: hk.LSTMState) -> chex.Array:
    """从 rnn_state 中提取输出向量。

    Args:
      state: LSTM 状态。

    Returns:
      输出向量。
    """
    return state.hidden  # pytype: disable=attribute-error  # numpy-scalars


def _multi_level_extract_by_attr_or_key(x: Any, keys: str) -> Any:
  """返回 `x[k0][k1]...[kn]`，其中 `keys` 的形式为 `k0[/k1]`。

  Args:
    x: 嵌套结构。
    keys: 键路径。

  Returns:
    提取的值。

  Raises:
    ValueError: 如果中间值为 None。
    KeyError: 如果键不存在。
  """

  # Note that the keys can also be attributes of `x`.
  # A simple usage example: assert extract({'a': {'b': {'c': 3}}}, 'a/b/c') == 3

  def _get_attr_or_key(x: Any, key: str, keys: str) -> Any:
    if hasattr(x, key):
      return getattr(x, key)
    else:
      try:
        return x[key]
      except:
        raise KeyError(f'Input {x} has no attr or key {key}. {keys}') from None

  processed_keys = []
  for key in keys.split('/'):
    if x is None:
      raise ValueError(
          f'x/{"/".join(processed_keys)} is `None`, cannot recurse up to'
          f' x/{keys}'
      )
    x = _get_attr_or_key(x, key, keys)
    processed_keys.append(key)
  return x


def _construct_input(
    inputs: types.UpdateRuleInputs,
    input_option: types.MetaNetInputOption,
    y_net: Callable[[chex.Array], chex.Array],
    z_net: Callable[[chex.Array], chex.Array],
    policy_net: Callable[[chex.Array], chex.Array],
    axis_name: str | None = None,
) -> tuple[chex.Array, chex.Array | None]:
  """将更新规则输入映射到单个向量。

  Args:
    inputs: 更新规则输入。
    input_option: 输入选项。
    y_net: Y 网络函数。
    z_net: Z 网络函数。
    policy_net: 策略网络函数。
    axis_name: 轴名称。

  Returns:
    处理后的输入数组和动作条件嵌入（如果适用）的元组。
  """
  unroll_len, batch_size = inputs.is_terminal.shape

  actions = jax.tree.map(lambda x: x[:-1], inputs.actions)  # [T, B]
  policy = lax.stop_gradient(
      jax.nn.softmax(inputs.agent_out['logits'])
  )  # [T+1, B, A]
  num_actions = policy.shape[2]

  def preprocess_from_config(inputs, preproc_config, prefix_shape):
    inputs_t = []
    for input_config in preproc_config:
      # Extract inputs according to the config.
      x = _multi_level_extract_by_attr_or_key(inputs, input_config.source)

      # Align extra input dims.
      if (
          input_config.source.startswith('extra_from_rule')
          and 'target_out' not in input_config.source
      ) or input_config.source == 'extra_from_rule/target_out/q':
        x = jnp.expand_dims(x, axis=-1)

      # Apply transforms.
      for tx in input_config.transforms:
        if tx == 'y_net':
          x = y_net(x)
        elif tx == 'z_net':
          x = z_net(x)
        else:
          if tx not in input_transforms.TRANSFORMS:
            raise KeyError(
                f'Transform {tx} was not found in {input_config.transforms}.'
            )
          transform_fn = input_transforms.TRANSFORMS[tx]()
          x = transform_fn(x, actions, policy, axis_name)

      # Flatten to a vector.
      x = jnp.reshape(x, (*prefix_shape, -1))
      inputs_t.append(x)
    return inputs_t

  inputs_t = preprocess_from_config(
      inputs, input_option.base, prefix_shape=(unroll_len, batch_size)
  )

  # Get action-conditional inputs, if required.
  if input_option.action_conditional:
    act_cond_inputs = preprocess_from_config(
        inputs,
        input_option.action_conditional,
        prefix_shape=(unroll_len, batch_size, num_actions),
    )
    act_cond_inputs.append(
        jnp.expand_dims(
            jax.nn.one_hot(actions, num_actions, dtype=jnp.float32), axis=-1
        )
    )
    act_cond_inputs = jnp.concatenate(act_cond_inputs, axis=-1)
    act_cond_emb = policy_net(act_cond_inputs)  # [T, B, A, C]
    act_cond_emb_avg = jnp.mean(act_cond_emb, axis=2)  # [T, B, C]
    act_cond_emb_a = utils.batch_lookup(act_cond_emb, actions)  # [T, B, C]
    inputs_t += [act_cond_emb_avg, act_cond_emb_a]
  else:
    act_cond_emb = None

  chex.assert_rank(inputs_t, 3)
  chex.assert_tree_shape_prefix(inputs_t, (unroll_len, batch_size))
  return jnp.concatenate(inputs_t, axis=-1), act_cond_emb


def _maybe_get_initializer(
    stddev: float | None,
) -> hk.initializers.Initializer | None:
  """根据给定的标准差获取初始化器，如果 stddev 为 None 则返回 None。

  Args:
    stddev: 标准差。

  Returns:
    初始化器或 None。
  """
  return hk_init.TruncatedNormal(stddev=stddev) if stddev is not None else None


def _multiplicative_interaction(
    x: chex.Array, y: chex.Array, initializer: hk.initializers.Initializer
) -> chex.Array:
  """如果 y 不为 None，则返回 out = x * Linear(y)。否则返回 x。

  Args:
    x: 输入数组 x。
    y: 输入数组 y。
    initializer: 初始化器。

  Returns:
    交互后的数组。
  """
  if isinstance(y, chex.Array) and y.shape:  # not scalar
    # Condition on rnn_state via multiplicative interaction.
    y_embed = hk.Linear(x.shape[-1], w_init=initializer)(y)
    return jnp.multiply(x, y_embed)
  else:
    return x


def _batch_mlp(
    hiddens: Sequence[int], num_dims: int = 2
) -> Callable[[chex.Array], chex.Array]:
  """创建一个批处理 MLP。

  Args:
    hiddens: 隐藏层大小序列。
    num_dims: 批处理维度数量。

  Returns:
    批处理 MLP 函数。
  """
  return hk.BatchApply(hk.nets.MLP(hiddens), num_dims=num_dims)


def _conv1d_block(x: chex.Array, n_channels: int) -> chex.Array:
  """1D 卷积块。

  Args:
    x: 输入数组。
    n_channels: 输出通道数。

  Returns:
    处理后的数组。
  """
  x_avg = jnp.mean(x, axis=2, keepdims=True)  # [T, B, 1, C]
  x_avg = jnp.repeat(x_avg, x.shape[2], axis=2)  # [T, B, A, C]
  x = jnp.concatenate([x, x_avg], axis=-1)  # [T, B, A, 2C]
  x = hk.BatchApply(hk.Conv1D(output_channels=n_channels, kernel_shape=1))(x)
  x = jax.nn.relu(x)  # [T, B, A, C_new]
  return x


def _conv1d_net(channels: Sequence[int]) -> Callable[[chex.Array], chex.Array]:
  """创建一个 1D 卷积网络。

  Args:
    channels: 每个块的通道数序列。

  Returns:
    卷积网络函数。
  """
  return hk.Sequential(
      [functools.partial(_conv1d_block, n_channels=c) for c in channels]
  )

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

"""动作条件模型。"""

import chex
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

from disco_rl import types
from disco_rl import utils


def get_action_model(name: str, *args, **kwargs):
  """获取动作条件模型实例。

  Args:
    name: 模型名称 (例如 'lstm')。
    *args: 传递给模型构造函数的位置参数。
    **kwargs: 传递给模型构造函数的关键字参数。

  Returns:
    实例化的动作模型。

  Raises:
    ValueError: 如果模型名称无效。
  """
  if name == 'lstm':
    net = LSTMModel(*args, **kwargs)
  else:
    raise ValueError(f'Invalid model network name {name}.')
  return net


class LSTMModel:
  """基于 LSTM 的动作条件模型，灵感来自 Muesli/MuZero。"""

  def __init__(
      self,
      action_spec: types.ActionSpec,
      out_spec: types.Specs,
      head_mlp_hiddens: tuple[int, ...],
      lstm_size: int,
  ) -> None:
    """初始化 LSTM 模型。

    Args:
      action_spec: 动作规范。
      out_spec: 输出规范。
      head_mlp_hiddens: 头部 MLP 隐藏层大小。
      lstm_size: LSTM 大小。
    """
    self._out_spec = out_spec
    self._action_spec = action_spec
    self._head_mlp_hiddens = head_mlp_hiddens
    self._lstm_size = lstm_size

  def _model_transition_all_actions(
      self, embedding: hk.LSTMState
  ) -> chex.Array:
    """对所有动作执行模型转换传递。

    Args:
      embedding: LSTM 状态嵌入。

    Returns:
      LSTM 输出。
    """
    num_actions = utils.get_num_actions_from_spec(self._action_spec)
    batch_size = embedding.cell.shape[0]

    # Enumerate all action embeddings.
    one_hot_actions = jnp.eye(num_actions).astype(
        embedding.cell.dtype
    )  # [A, A]
    batched_one_hot_actions = jnp.tile(
        one_hot_actions, [batch_size, 1]
    )  # [BA, A]

    all_actions_embed = jax.tree.map(
        lambda x: jnp.repeat(x, repeats=num_actions, axis=0), embedding
    )  # [BA, *H]

    lstm_output, _ = hk.LSTM(self._lstm_size, name='action_cond')(
        batched_one_hot_actions, all_actions_embed
    )
    return lstm_output

  def _model_head_pass(
      self, transition_output: chex.Array
  ) -> dict[str, chex.Array]:
    """从 MLP 获取根据输出规范形状的输出。

    Args:
      transition_output: 转换输出。

    Returns:
      模型输出字典。
    """
    # transition_output has shape [BA, ...]
    num_actions = utils.get_num_actions_from_spec(self._action_spec)
    batch_size = transition_output.shape[0] // num_actions

    model_outputs = dict()
    for key, pred_spec in self._out_spec.items():
      pred = hk.nets.MLP(self._head_mlp_hiddens + (np.prod(pred_spec.shape),))(
          transition_output
      )
      model_outputs[key] = pred.reshape(
          (batch_size, num_actions, *pred_spec.shape)
      )

    return model_outputs

  def model_step(self, embedding: hk.LSTMState) -> dict[str, chex.Array]:
    """执行模型步骤。

    Args:
      embedding: LSTM 状态嵌入。

    Returns:
      模型输出字典。
    """
    transition_output = self._model_transition_all_actions(embedding)
    model_outputs = self._model_head_pass(transition_output)
    return model_outputs

  def root_embedding(self, state: chex.Array) -> hk.LSTMState:
    """从代理的状态构建根节点。

    Args:
      state: 代理状态。

    Returns:
      LSTM 状态。
    """
    flat_state = hk.Flatten()(state)
    cell = hk.Linear(self._lstm_size)(flat_state)
    return hk.LSTMState(hidden=jnp.tanh(cell), cell=cell)

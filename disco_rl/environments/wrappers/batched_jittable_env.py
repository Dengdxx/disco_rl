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

"""用于批处理 JIT 编译环境的包装器。"""

import chex
import dm_env
from dm_env import specs as dm_specs
import jax
import jax.numpy as jnp
from ml_collections import config_dict as configdict
import numpy as np

from disco_rl import types
from disco_rl.environments import base


def _to_env_timestep(
    obs: chex.Array, reward: chex.Array, is_terminal: chex.Array
) -> types.EnvironmentTimestep:
  """将原始环境输出转换为 EnvironmentTimestep。

  Args:
    obs: 观测值。
    reward: 奖励。
    is_terminal: 是否终止。

  Returns:
    EnvironmentTimestep 实例。
  """
  return types.EnvironmentTimestep(
      step_type=jax.lax.select(
          is_terminal, dm_env.StepType.LAST, dm_env.StepType.MID
      ),
      reward=reward.astype(jnp.float32),
      observation=obs.astype(jnp.float32),
  )


@chex.dataclass(mappable_dataclass=False)
class EnvState:
  """环境状态数据类。

  Attributes:
    state: 内部状态树。
    rng: 随机数生成器密钥。
  """
  state: chex.ArrayTree
  rng: chex.PRNGKey


class BatchedJittableEnvironment(base.Environment):
  """用于制作批处理 JIT 编译 disco 环境的包装器。"""

  def __init__(
      self,
      env_class,
      batch_size: int,
      env_settings: configdict.ConfigDict,
  ):
    """初始化批处理 JIT 环境。

    Args:
      env_class: 环境类。
      batch_size: 批次大小。
      env_settings: 环境设置。
    """
    self.batch_size = batch_size
    self._env = env_class(**env_settings.to_dict())
    self._single_action_spec = dm_specs.BoundedArray(
        (), np.int32, 0, self._env.num_actions - 1
    )

    dummy_state = self._env.initial_state(jax.random.PRNGKey(0))
    obs = self._env.render(dummy_state)
    self._single_observation_spec = {
        'observation': dm_specs.Array(shape=obs.shape, dtype=obs.dtype)
    }

    self._batched_env_step = jax.vmap(self._single_env_step)
    self._batched_env_reset = jax.vmap(self._single_env_reset)

  def _single_env_step(
      self, env_state: EnvState, action: chex.Array
  ) -> tuple[EnvState, types.EnvironmentTimestep]:
    """执行单个环境的步骤。

    Args:
      env_state: 当前环境状态。
      action: 动作。

    Returns:
      新的环境状态和时间步。
    """
    new_rng, rng_step = jax.random.split(env_state.rng)
    new_state = self._env.step(rng_step, env_state.state, action)
    is_terminal = self._env.is_terminal(new_state)
    reward = self._env.reward(new_state)

    # Use initial states for terminated episodes while keeping other data.
    init_state = self._env.episode_reset(rng_step, new_state)
    next_state = jax.tree.map(
        lambda reset_x, x: jax.lax.select(is_terminal, reset_x, x),
        init_state,
        new_state,
    )

    return EnvState(state=next_state, rng=new_rng), _to_env_timestep(
        self._env.render(next_state), reward, is_terminal
    )

  def _single_env_reset(
      self, rng_key: chex.PRNGKey
  ) -> tuple[EnvState, types.EnvironmentTimestep]:
    """重置单个环境。

    Args:
      rng_key: 随机数生成器密钥。

    Returns:
      环境状态和时间步。
    """
    new_rng, reset_rng = jax.random.split(rng_key)
    state = self._env.initial_state(reset_rng)
    return EnvState(state=state, rng=new_rng), _to_env_timestep(
        self._env.render(state),
        self._env.reward(state),
        self._env.is_terminal(state),
    )

  def step(  # pytype: disable=signature-mismatch  # numpy-scalars
      self, state: EnvState, actions: chex.Array
  ) -> tuple[EnvState, types.EnvironmentTimestep]:
    """执行批处理步骤。

    Args:
      state: 批处理环境状态。
      actions: 批处理动作。

    Returns:
      新的批处理环境状态和时间步。
    """
    return self._batched_env_step(state, actions)

  def reset(self, rng_key: chex.PRNGKey) -> tuple[EnvState, types.EnvironmentTimestep]:  # pytype: disable=signature-mismatch  # numpy-scalars
    """重置批处理环境。

    Args:
      rng_key: 随机数生成器密钥。

    Returns:
      批处理环境状态和时间步。
    """
    rngs = jax.random.split(rng_key, self.batch_size)
    return self._batched_env_reset(rngs)

  def single_action_spec(self) -> types.ActionSpec:
    """返回单个动作规范。

    Returns:
      动作规范。
    """
    return self._single_action_spec

  def single_observation_spec(self) -> types.Specs:
    """返回单个观测规范。

    Returns:
      观测规范。
    """
    return self._single_observation_spec

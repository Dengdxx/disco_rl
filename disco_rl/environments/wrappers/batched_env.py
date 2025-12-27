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

"""用于批处理单流环境的包装器。"""

import copy

import chex
import dm_env
import jax
from ml_collections import config_dict as configdict
import numpy as np
import rlax

from disco_rl import types
from disco_rl import utils
from disco_rl.environments import base


class UnusedEnvState:
  """未使用的环境状态占位符。"""
  pass


def _to_env_timestep(timestep: dm_env.TimeStep) -> types.EnvironmentTimestep:
  """将 dm_env.TimeStep 转换为 EnvironmentTimestep。

  Args:
    timestep: dm_env 时间步。

  Returns:
    EnvironmentTimestep 实例。
  """
  return types.EnvironmentTimestep(
      step_type=np.array(timestep.step_type, dtype=np.int32),
      reward=np.array(timestep.reward or 0.0),
      observation=timestep.observation,
  )


class BatchedSingleStreamEnvironment(base.Environment):
  """用于使单流环境批处理化的包装器。

  所有操作对批处理中的所有实例顺序执行。

  Attributes:
    batch_size: 批次大小。
  """

  def __init__(
      self,
      env_class,
      batch_size: int,
      env_settings: configdict.ConfigDict,
  ) -> None:
    """初始化批处理环境。

    Args:
      env_class: 环境类。
      batch_size: 批次大小。
      env_settings: 环境设置。
    """
    self.batch_size = batch_size
    self._num_envs = batch_size
    self._shape_prefix = (batch_size,)

    if 'random_seed' in env_settings:
      seed = env_settings.random_seed
      max_seed = np.iinfo(np.int32).max
      seeds = jax.random.randint(
          jax.random.PRNGKey(seed), (self._num_envs,), 0, max_seed
      )
      envs = []
      for i in range(self._num_envs):
        settings = copy.deepcopy(env_settings)
        settings.random_seed = int(seeds[i])
        envs.append(env_class(env_settings=settings))
      self._envs = tuple(envs)
    else:
      self._envs = tuple(
          env_class(env_settings=env_settings) for _ in range(self._num_envs)
      )

    self._single_action_spec = self._envs[0].single_action_spec()
    self._single_observation_spec = self._envs[0].single_observation_spec()
    self._states = [env.reset(jax.random.PRNGKey(0))[1] for env in self._envs]

  def _stack_states(self) -> types.EnvironmentTimestep:
    """堆叠各个环境的状态。

    Returns:
      批处理的环境时间步。
    """
    states = jax.tree.map(
        lambda *xs: np.stack(xs).reshape(self._shape_prefix + xs[0].shape),
        *self._states,
    )
    return utils.cast_to_single_precision(states, host_data=True)

  def step(
      self, state: UnusedEnvState | None, actions: chex.ArrayTree
  ) -> tuple[UnusedEnvState | None, types.EnvironmentTimestep]:
    """执行一步。

    Args:
      state: 状态（未使用）。
      actions: 动作。

    Returns:
      未使用状态和新的环境时间步的元组。
    """
    del state
    chex.assert_tree_shape_prefix(actions, self._shape_prefix)
    actions = jax.tree.map(
        lambda a: a.reshape((self._num_envs,) + a.shape[2:]), actions
    )
    actions_list = rlax.tree_split_leaves(actions)
    self._states = [
        env.step(None, a)[1] for env, a in zip(self._envs, actions_list)
    ]
    return UnusedEnvState(), self._stack_states()

  def reset(
      self, rng_key: chex.PRNGKey
  ) -> tuple[UnusedEnvState | None, types.EnvironmentTimestep]:
    """重置环境。

    Args:
      rng_key: 随机数生成器密钥。

    Returns:
      未使用状态和初始环境时间步的元组。
    """
    del rng_key
    self._states = [env.reset(jax.random.PRNGKey(0))[1] for env in self._envs]
    return UnusedEnvState(), self._stack_states()

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

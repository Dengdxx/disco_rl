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

"""环境的基本接口。"""

import abc
from typing import Any, TypeVar

import chex

from disco_rl import types
from disco_rl import utils

_EnvState = TypeVar('_EnvState')


class Environment(abc.ABC):
  """环境接口。

  所有环境都应该是批处理的。

  Attributes:
    batch_size: 批处理大小。
  """

  batch_size: int

  @abc.abstractmethod
  def step(
      self, states: _EnvState, actions: chex.ArrayTree
  ) -> tuple[_EnvState, types.EnvironmentTimestep]:
    """执行一步环境交互。

    Args:
      states: 当前环境状态。
      actions: 执行的动作。

    Returns:
      下一个环境状态和环境时间步的元组。
    """
    pass

  @abc.abstractmethod
  def reset(
      self, rng_key: chex.PRNGKey
  ) -> tuple[Any, types.EnvironmentTimestep]:
    """重置剧集。

    Args:
      rng_key: 随机数生成器密钥。

    Returns:
      初始环境状态和环境时间步的元组。
    """
    pass

  @abc.abstractmethod
  def single_observation_spec(self) -> types.Specs:
    """返回单个观测的规范。

    Returns:
      观测规范。
    """
    pass

  @abc.abstractmethod
  def single_action_spec(self) -> types.ActionSpec:
    """返回单个动作的规范。

    Returns:
      动作规范。
    """
    pass

  def batched_action_spec(self) -> types.ActionSpec:
    """返回批处理动作的规范。

    Returns:
      批处理动作规范。
    """
    return utils.broadcast_specs(self.single_action_spec(), self.batch_size)

  def batched_observation_spec(self) -> types.Specs:
    """返回批处理观测的规范。

    Returns:
      批处理观测规范。
    """
    return utils.broadcast_specs(
        self.single_observation_spec(), self.batch_size
    )

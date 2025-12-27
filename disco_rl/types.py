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

"""类型定义。"""

from typing import Any, Callable, Mapping, Sequence

import chex
import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
ArraySpec = jax.ShapeDtypeStruct
ActionSpec = specs.BoundedArray
Specs = dict[str, specs.Array | ArraySpec]
SpecsTree = ArraySpec | Sequence['SpecsTree'] | dict[str, 'SpecsTree']
MetaState = dict[str, chex.ArrayTree | None]
UpdateRuleLog = dict[str, chex.ArrayTree]

HaikuState = hk.State
AgentOuts = dict[str, chex.ArrayTree]
UpdateRuleOuts = dict[str, chex.ArrayTree]
HyperParams = dict[str, chex.Array | float]

OptState = chex.ArrayTree
AgentParams = chex.ArrayTree
MetaParams = chex.ArrayTree
MetaParamsEMA = dict[float, MetaParams]  # {decay: params}
# State that can be directly updated by update rule.
MetaState = dict[str, chex.ArrayTree | None]
LogDict = dict[str, chex.Array]
RNNState = chex.ArrayTree


# (params, state, observations, should_reset) -> (agent_out, new_state)
AgentStepFn = Callable[
    [AgentParams, HaikuState, chex.ArrayTree, chex.Array | None],
    tuple[AgentOuts, HaikuState],
]
AgentUnrollFn = Callable[
    [
        AgentParams,
        HaikuState,
        chex.ArrayTree,
        chex.Array | None,
    ],
    tuple[AgentOuts, HaikuState],
]
SampleActionFn = Callable[
    [chex.PRNGKey, MetaParams, AgentOuts],
    tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree],
]


@chex.dataclass
class ValueFnConfig:
  """价值函数配置。

  Attributes:
    net: 网络名称。
    net_args: 网络参数字典。
    learning_rate: 学习率。
    max_abs_update: 最大绝对更新值。
    discount_factor: 折扣因子。
    td_lambda: TD-lambda 参数。
    outer_value_cost: 外部价值代价。
    ema_decay: 指数移动平均衰减率。
    ema_eps: 指数移动平均 epsilon。
  """

  net: str
  net_args: dict[str, Any]
  learning_rate: float
  max_abs_update: float
  discount_factor: float
  td_lambda: float
  outer_value_cost: float
  ema_decay: float = 0.99
  ema_eps: float = 1e-6


@chex.dataclass
class ValueState:
  """价值函数状态。

  Attributes:
    params: 代理参数。
    state: Haiku 状态。
    opt_state: 优化器状态。
    adv_ema_state: 优势函数的指数移动平均状态。
    td_ema_state: 时序差分的指数移动平均状态。
  """
  params: AgentParams
  state: HaikuState
  opt_state: OptState
  adv_ema_state: 'EmaState'
  td_ema_state: 'EmaState'


@chex.dataclass
class TransformConfig:
  """变换配置。

  Attributes:
    source: 源名称。
    transforms: 变换函数或名称序列。
  """
  source: str
  transforms: Sequence[str | Callable[[Any], chex.Array]]


@chex.dataclass
class MetaNetInputOption:
  """元网络输入选项。

  Attributes:
    base: 基础变换配置序列。
    action_conditional: 动作条件变换配置序列。
  """

  base: Sequence[TransformConfig]
  action_conditional: Sequence[TransformConfig]


@chex.dataclass(mappable_dataclass=False, frozen=True)
class PolicyNetwork:
  """收集底层代理网络的有用可调用变换。

  Attributes:
    init: 初始化函数，接受 rng, obs, should_reset 并返回 (params, state)。
    one_step: 单步执行函数。
    unroll: 展开执行函数。
  """

  # hk-transformed functions.
  init: Callable[
      [
          chex.PRNGKey,  # rng for params
          chex.ArrayTree,  # obs
          chex.Array | None,  # should_reset
      ],
      tuple[AgentParams, HaikuState],
  ]
  one_step: AgentStepFn
  unroll: AgentUnrollFn


@chex.dataclass
class EmaState:
  """指数移动平均状态。

  Attributes:
    moment1: 一阶矩树。
    moment2: 二阶矩树。
    decay_product: 从累积开始的所有衰减的乘积。
  """
  # The tree of first moments.
  moment1: chex.ArrayTree
  # The tree of second moments.
  moment2: chex.ArrayTree
  # The product of the all decays from the start of accumulating.
  decay_product: float


@chex.dataclass
class EnvironmentTimestep:
  """环境时间步。

  Attributes:
    observation: 观测值映射。
    step_type: 步骤类型。
    reward: 奖励。
  """
  observation: Mapping[str, chex.ArrayTree]
  step_type: chex.Array
  reward: chex.Array


@chex.dataclass
class ActorTimestep:
  """演员时间步。

  Attributes:
    observations: 观测值。
    actions: 动作。
    rewards: 奖励。
    discounts: 折扣。
    agent_outs: 代理输出。
    states: Haiku 状态。
    logits: Logits。
  """

  observations: chex.ArrayTree
  actions: Any
  rewards: Any
  discounts: Any
  agent_outs: AgentOuts
  states: HaikuState
  logits: Any

  @classmethod
  def from_rollout(cls, rollout: 'ActorRollout') -> 'ActorTimestep':
    """从 rollout 创建 ActorTimestep。

    Args:
      rollout: ActorRollout 实例。

    Returns:
      ActorTimestep 实例。
    """
    return cls(
        observations=rollout.observations,
        actions=rollout.actions,
        rewards=rollout.rewards,
        discounts=rollout.discounts,
        agent_outs=rollout.agent_outs,
        states=rollout.states,
        logits=rollout.logits,
    )

  def to_env_timestep(self) -> 'EnvironmentTimestep':
    """转换为 EnvironmentTimestep。

    Returns:
      EnvironmentTimestep 实例。
    """
    return EnvironmentTimestep(
        observation=self.observations,
        step_type=jnp.where(
            self.discounts > 0, dm_env.StepType.MID, dm_env.StepType.LAST
        ),
        reward=self.rewards,
    )


@chex.dataclass
class ActorRollout(ActorTimestep):
  """堆叠的演员时间步。

  形状: [D, O, T, B, ...] (默认情况下; 可以在代码中更改)。

  其中:
    D: 学习器设备数量
    O: 外部 rollout 长度 (即 meta rollout 长度)
    T: 轨迹长度
    B: 批次大小

  Attributes:
    observations: 观测值。
    actions: 动作。
    rewards: 奖励。
    discounts: 折扣。
    agent_outs: 代理输出。
    states: Haiku 状态。
    logits: Logits。
  """

  @classmethod
  def from_timestep(cls, timestep: ActorTimestep) -> 'ActorRollout':
    """从时间步创建 ActorRollout。

    Args:
      timestep: ActorTimestep 实例。

    Returns:
      ActorRollout 实例。
    """
    return cls(**timestep)

  def first_state(self, time_axis: int) -> HaikuState:
    """获取时间轴上的第一个状态。

    Args:
      time_axis: 时间轴的索引。

    Returns:
      HaikuState: 第一个状态。
    """
    index = tuple([np.s_[:]] * (time_axis - 1) + [0])
    return jax.tree.map(lambda x: x[index], self.states)


@chex.dataclass
class ValueOuts:
  """价值函数输出。

  Attributes:
    value: 标量价值。
    target_value: 标量目标价值。
    rho: 重要性权重。
    adv: 优势。
    normalized_adv: 归一化优势。
    value_target: 价值目标。
    td: 时序差分 (value_target - value)。
    normalized_td: 归一化时序差分。
    qv_adv: Q - V。
    normalized_qv_adv: 归一化 Q - V。
    q_value: 标量 Q 值。
    target_q_value: 标量目标 Q 值。
    q_target: Q 值目标。
    q_td: Q 时序差分 (q_target - q_a)。
    normalized_q_td: 归一化 Q 时序差分。
  """

  value: jax.typing.ArrayLike = 0.0  # Scalar value
  target_value: jax.typing.ArrayLike = 0.0  # Scalar target value
  rho: jax.typing.ArrayLike = 0.0  # Importance weight
  adv: jax.typing.ArrayLike = 0.0  # Advantage
  normalized_adv: jax.typing.ArrayLike = 0.0  # Normalised advantage
  value_target: jax.typing.ArrayLike = 0.0  # Value target
  td: jax.typing.ArrayLike = 0.0  # value_target - value
  normalized_td: jax.typing.ArrayLike = 0.0  # Normalised TD
  qv_adv: chex.ArrayTree | None = None  # Q - V
  normalized_qv_adv: chex.ArrayTree | None = None  # Normalised Q - V
  q_value: chex.ArrayTree | None = None  # Scalar Q-value
  target_q_value: chex.ArrayTree | None = None  # Scalar target Q-value
  q_target: chex.ArrayTree | None = None  # Q-value target
  q_td: chex.ArrayTree | None = None  # q_target - q_a
  normalized_q_td: chex.ArrayTree | None = None  # Normalised q_td


@chex.dataclass
class UpdateRuleInputs:
  """更新规则输入。

  Attributes:
    observations: 观测值。
    actions: 动作。
    rewards: 奖励。
    is_terminal: 动作是否为终止动作。
    agent_out: 代理输出。
    behaviour_agent_out: 行为代理输出。
    value_out: 价值函数输出。
    extra_from_rule: 更新规则中元网络之前的预处理输入（例如优势）。
  """

  observations: chex.ArrayTree
  actions: chex.Array
  rewards: chex.Array
  is_terminal: chex.Array  # whether the action was terminal
  agent_out: chex.ArrayTree
  behaviour_agent_out: AgentOuts | None = None
  value_out: ValueOuts | None = None
  # Inputs with pre-processing in update rule before meta-net (e.g. advantages)
  extra_from_rule: chex.ArrayTree | None = None

  @property
  def should_reset_mask_fwd(self) -> chex.Array:
    """返回前向 RNN 的 `should_reset` 掩码。

    将 is_terminal 向右移动一步，模仿 step_type.is_first()。

    Returns:
      chex.Array: 前向掩码。
    """
    # Shifts is_terminal to the right by a step, mimicking step_type.is_first().
    prepend_non_terminal = jnp.zeros_like(self.is_terminal[:1])
    return jnp.concatenate(
        (prepend_non_terminal, self.is_terminal),
        axis=0,
        dtype=self.is_terminal.dtype,
    )

  @property
  def should_reset_mask_bwd(self) -> chex.Array:
    """返回后向 RNN 的 `should_reset` 掩码。

    附加一个非终止步骤，用于自举。

    Returns:
      chex.Array: 后向掩码。
    """
    # Appends one non-terminal step, for bootstrapping.
    append_non_terminal = jnp.zeros_like(self.is_terminal[:1])
    return jnp.concatenate(
        (self.is_terminal, append_non_terminal),
        axis=0,
        dtype=self.is_terminal.dtype,
    )

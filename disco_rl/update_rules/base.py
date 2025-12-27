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

"""更新规则的基类。参见 `UpdateRule` 的文档字符串。"""

import chex
from dm_env import specs as dm_env_specs
import jax
import jax.numpy as jnp

from disco_rl import types
from disco_rl import utils

ArraySpec = types.ArraySpec


def get_agent_out_spec(
    action_spec: types.ActionSpec,
    flat_out_spec: types.Specs,
    model_out_spec: types.Specs,
) -> types.Specs:
  """为提供的规范构建输出形状树。

  示例:
      >> action_spec = specs.BoundedArray((), int, minimum=0, maximum=A-1)
      >> unconditional_output_spec = {'logits': (A,),
                                      'y': (Y,),
                                      }
      >> conditional_output_spec = {'z': (Z,),
                                    'aux_pi': (P,),
                                    }
      >> agent_output_spec = {
            'logits': (A,),
            'y': (Y,),

            'z': (A, Z),
            'aux_pi': (A, P),
         }

  Args:
      action_spec: 一个动作规范。
      flat_out_spec: 代理平面输出形状的嵌套字典。
      model_out_spec: 代理模型输出形状的嵌套字典。

  Returns:
      一个嵌套字典，其中形状指定了代理的总输出形状。
  """
  if set(flat_out_spec.keys()).intersection(set(model_out_spec.keys())):
    raise ValueError(
        'Keys overlap between flat_out_spec and model_out_spec.'
        f'Given: {flat_out_spec} and {model_out_spec}'
    )

  num_actions = utils.get_num_actions_from_spec(action_spec)
  agent_out_spec = {key: val for key, val in flat_out_spec.items()}
  for key, val in model_out_spec.items():
    agent_out_spec[key] = ArraySpec((num_actions, *val.shape), val.dtype)
  return agent_out_spec


class UpdateRule:
  """更新规则的基类。"""

  def _get_dummy_input(
      self,
      include_behaviour_out: bool = True,
      include_value_out: bool = False,
      include_agent_adv: bool = False,
  ) -> types.UpdateRuleInputs:
    """生成用于初始化的虚拟元网络输入。

    Args:
      include_behaviour_out: 是否包含行为输出。
      include_value_out: 是否包含价值输出。
      include_agent_adv: 是否包含代理优势。

    Returns:
      虚拟更新规则输入。
    """
    b = 1
    t = 2
    unroll_batch_shape = (t, b)
    bootstrapped_shape = (t + 1, b)
    dummy_action_spec = dm_env_specs.BoundedArray(
        shape=(3,), dtype=int, minimum=0, maximum=3
    )
    num_actions = utils.get_num_actions_from_spec(dummy_action_spec)
    dummy_actions = jnp.zeros(bootstrapped_shape, dtype=jnp.int32)
    agent_out_shapes = self.agent_output_spec(dummy_action_spec)

    agent_out = jax.tree.map(
        lambda s: jnp.zeros(bootstrapped_shape + s.shape), agent_out_shapes
    )

    dummy_input = types.UpdateRuleInputs(
        observations=jnp.zeros(bootstrapped_shape),
        actions=dummy_actions,
        rewards=jnp.zeros(unroll_batch_shape),
        is_terminal=jnp.ones(unroll_batch_shape, dtype=jnp.bool_),
        agent_out=agent_out,
    )

    if include_behaviour_out:
      dummy_input.behaviour_agent_out = {}
      dummy_input.behaviour_agent_out.update(agent_out)

    target_out = agent_out
    dummy_input.extra_from_rule = dict(target_out=target_out)

    if include_agent_adv:
      value_unroll_batch_shape = (t, b, 1)
      value_bootstrapped_shape = (t + 1, b, 1)
      q_bootstrapped_shape = (t + 1, b, num_actions, 1)

      dummy_input.extra_from_rule = dict(
          adv=jnp.zeros(value_unroll_batch_shape),
          normalized_adv=jnp.zeros(value_unroll_batch_shape),
          v_scalar=jnp.zeros(value_bootstrapped_shape),
          q=jnp.zeros(q_bootstrapped_shape),
          qv_adv=jnp.zeros(q_bootstrapped_shape),
          normalized_qv_adv=jnp.zeros(q_bootstrapped_shape),
          target_out=target_out,
      )

    if include_value_out:
      num_discounts = 1
      value_unroll_batch_shape = (t, b, num_discounts)
      q_shape = (t, b, num_actions, num_discounts)
      bootstrapped_q_shape = (t + 1, b, num_actions, num_discounts)
      value_bootstrapped_shape = (t + 1, b, num_discounts)
      dummy_input.value_out = types.ValueOuts(
          value=jnp.ones(value_bootstrapped_shape),
          target_value=jnp.ones(value_bootstrapped_shape),
          rho=jnp.ones(unroll_batch_shape),
          adv=jnp.ones(value_unroll_batch_shape),
          normalized_adv=jnp.ones(value_unroll_batch_shape),
          td=jnp.ones(value_unroll_batch_shape),
          normalized_td=jnp.ones(value_unroll_batch_shape),
          value_target=jnp.ones(value_unroll_batch_shape),
          qv_adv=jax.tree.map(jnp.ones, bootstrapped_q_shape),
          normalized_qv_adv=jax.tree.map(jnp.ones, bootstrapped_q_shape),
          q_target=jax.tree.map(jnp.ones, q_shape),
          q_value=jax.tree.map(jnp.ones, q_shape),
          target_q_value=jax.tree.map(jnp.ones, q_shape),
          q_td=jax.tree.map(jnp.ones, q_shape),
          normalized_q_td=jax.tree.map(jnp.ones, q_shape),
      )

    return dummy_input

  def init_params(
      self, rng: chex.PRNGKey
  ) -> tuple[types.MetaParams, chex.ArrayTree]:
    """初始化元参数。

    Args:
      rng: 随机密钥。

    Returns:
      元参数和初始元网络状态的元组。
    """
    raise NotImplementedError

  def flat_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    """返回代理的无条件输出规范。

    Args:
      action_spec: 一个动作规范。

    Returns:
      一个指定输出规范的元组嵌套字典。
    """
    del action_spec
    return dict()

  def model_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    """返回代理的动作条件输出规范。

    Args:
      action_spec: 一个动作规范。

    Returns:
      一个指定模型输出规范的元组嵌套字典。
    """
    del action_spec
    return dict()

  def agent_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    """返回代理总输出的规范。

    Args:
      action_spec: 一个动作规范。

    Returns:
      一对指定代理总输出规范的字典。
    """
    return get_agent_out_spec(
        action_spec=action_spec,
        flat_out_spec=self.flat_output_spec(action_spec),
        model_out_spec=self.model_output_spec(action_spec),
    )

  def init_meta_state(
      self,
      rng: chex.PRNGKey,
      params: types.AgentParams,
  ) -> types.MetaState:
    """代理初始元状态。

    Args:
      rng: 随机密钥。
      params: 代理参数。

    Returns:
      具有元状态的数组树。
    """
    raise NotImplementedError

  def unroll_meta_net(
      self,
      meta_params: types.MetaParams,
      params: types.AgentParams,
      state: types.HaikuState,
      meta_state: types.MetaState,
      rollout: types.UpdateRuleInputs,
      hyper_params: types.HyperParams,
      unroll_policy_fn: types.AgentStepFn,
      rng: chex.PRNGKey,
      axis_name: str | None,
  ) -> tuple[types.UpdateRuleOuts, types.MetaState]:
    """展开元网络以准备代理的损失。

    Args:
      meta_params: 元参数。
      params: 代理参数。
      state: 代理状态。
      meta_state: 代理元状态。
      rollout: rollout。 [T, B, ...] 和 [T+1, B, ...] 用于 `agent_out`。
      hyper_params: 代理损失的超参数。
      unroll_policy_fn: 代理的策略展开函数。
      rng: 随机密钥。
      axis_name: 在集合操作中使用的轴名称（如果在 `pmap` 下运行）。

    Returns:
      元网络的输出 [T, B, ...] 和更新后的元状态。
    """
    raise NotImplementedError

  def agent_loss(
      self,
      rollout: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts,
      hyper_params: types.HyperParams,
      backprop: bool,
  ) -> tuple[chex.Array, types.UpdateRuleLog]:
    """代理损失。

    Args:
      rollout: 带有奖励、折扣等的 rollout。[T, B, ...]
      meta_out: 沿 rollout 的元网络输出。[T, B, ...]
      hyper_params: 代理损失的超参数。
      backprop: 是否使损失可微。

    Returns:
      每步损失（张量）和日志。
    """
    raise NotImplementedError

  def agent_loss_no_meta(
      self,
      rollout: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts | None,
      hyper_params: types.HyperParams,
  ) -> tuple[chex.Array, types.UpdateRuleLog]:
    """代理损失的可选部分，不应接收元梯度。

    Args:
      rollout: 带有奖励、折扣等的 rollout。[T, B, ...]
      meta_out: 沿 rollout 的元网络输出。[T, B, ...]
      hyper_params: 代理损失的超参数。

    Returns:
      每步损失（张量）和日志。
    """
    del meta_out, hyper_params
    return jnp.zeros_like(rollout.rewards), {}

  def __call__(
      self,
      meta_params: types.MetaParams,
      params: types.AgentParams,
      state: types.HaikuState,
      rollout: types.UpdateRuleInputs,
      hyper_params: types.HyperParams,
      meta_state: types.MetaState,
      unroll_policy_fn: types.AgentUnrollFn,
      rng: chex.PRNGKey,
      axis_name: str | None,
      backprop: bool = False,
  ) -> tuple[chex.Array, types.MetaState, types.UpdateRuleLog]:
    """从 rollout 和代理输出计算代理损失。

    Args:
      meta_params: 元参数。
      params: 代理参数。
      state: 代理状态。
      rollout: 带有奖励、折扣等的 rollout。[T+1, B, ...]
      hyper_params: 代理损失的标量超参数。
      meta_state: 代理元状态。
      unroll_policy_fn: 代理的策略展开函数。
      rng: 随机密钥。
      axis_name: 在集合操作中使用的轴名称（如果在 `pmap` 下运行）。
      backprop: 是否使损失相对于 meta_params 可微。

    Returns:
      一个元组 (每步损失, 元状态, 日志)。
    """
    meta_out, new_meta_state = self.unroll_meta_net(
        meta_params=meta_params,
        params=params,
        state=state,
        meta_state=meta_state,
        rollout=rollout,
        hyper_params=hyper_params,
        unroll_policy_fn=unroll_policy_fn,
        rng=rng,
        axis_name=axis_name,
    )
    loss_per_step, log_with_meta = self.agent_loss(
        rollout, meta_out, hyper_params, backprop=backprop
    )
    loss_per_step_no_meta, log_no_meta = self.agent_loss_no_meta(
        rollout, meta_out, hyper_params
    )

    loss_per_step = loss_per_step + loss_per_step_no_meta
    logs = log_with_meta | log_no_meta

    return loss_per_step, new_meta_state, logs

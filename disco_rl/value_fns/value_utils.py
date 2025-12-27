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

"""价值函数工具。"""

import functools

import chex
import distrax
import jax
import jax.numpy as jnp
from optax import losses
import rlax

from disco_rl import types
from disco_rl import utils


DEFAULT_DISCOUNT = 0.995
DEFAULT_TD_LAMBDA = 0.95


def get_value_outs(
    value_net_out: chex.Array | None,
    q_net_out: chex.ArrayTree | None,
    target_value_net_out: chex.Array | None,
    target_q_net_out: chex.ArrayTree | None,
    rollout: types.ActorRollout | types.UpdateRuleInputs,
    pi_logits: chex.Array,
    discount: jax.typing.ArrayLike,
    lambda_: jax.typing.ArrayLike = DEFAULT_TD_LAMBDA,
    nonlinear_transform: bool = False,
    categorical_value: bool = False,
    max_abs_value: float | None = None,
    drop_last: bool = True,
    adv_ema_state: types.EmaState | chex.ArrayTree | None = None,
    adv_ema_fn: utils.MovingAverage | None = None,
    td_ema_state: types.EmaState | chex.ArrayTree | None = None,
    td_ema_fn: utils.MovingAverage | None = None,
    axis_name: str | None = None,
) -> tuple[types.ValueOuts, types.EmaState | None, types.EmaState | None]:
  """获取价值输出并更新 EMA 状态，处理多折扣值。

  当仅给出状态值时计算 V-trace 目标，当给出动作值（或两者都给出）时计算 Retrace 目标。

  Args:
    value_net_out: 标量值或 logits，可能具有对应于多个折扣因子的额外最终维度。[T, B, 1 or num_bins, (num_gamma)]。
    q_net_out: 每个动作维度的标量值或 logits 树，可能具有对应于多个折扣因子的额外最终维度。[T, B, A, 1 or num_bins, (num_gamma)]。
    target_value_net_out: 用于自举的目标价值输出。[T, B, 1 or num_bins, (num_gamma)]。
    target_q_net_out: 用于自举的目标 Q 值输出。[T, B, A, 1 or num_bins, (num_gamma)]。
    rollout: 来自演员的 rollout 或为更新规则预处理的 rollout。
    pi_logits: 在线策略的 logits。
    discount: 标量或折扣因子向量。
    lambda_: 跟踪参数。
    nonlinear_transform: 如果为真，则对值应用带符号的双曲变换。
    categorical_value: 如果为真，则期望价值网络输出为 logits。
    max_abs_value: 分类价值函数可表示的最大绝对值；如果 `categorical_value` 为 False，则未使用；当 `categorical_value` 为 True 时必须设置。
    drop_last: 如果为真，则丢弃最后的奖励、终止、动作。
    adv_ema_state: 用于归一化的优势 EMA 状态。
    adv_ema_fn: 用于优势归一化的 EMA 状态。
    td_ema_state: 用于归一化的 TD EMA 状态。
    td_ema_fn: 用于 TD 归一化的 EMA 状态。
    axis_name: 归约轴，用于 EMA 统计聚合。

  Returns:
    一个元组，包含：
      value_outs (包含价值和优势估计、价值损失)
      更新后的优势 EMA 状态
      更新后的 TD EMA 状态
  """
  if adv_ema_state is not None:
    assert isinstance(adv_ema_state, types.EmaState)  # for pytypes
  if td_ema_state is not None:
    assert isinstance(td_ema_state, types.EmaState)  # for pytypes

  ema_arg_names = ('adv_ema_state', 'adv_ema_fn')
  ema_arg_undef = [name for name in ema_arg_names if name not in locals()]
  if len(ema_arg_undef) not in (0, len(ema_arg_names)):
    raise ValueError(
        f'Received some but not all ema args, undef: {ema_arg_undef}.'
    )

  if value_net_out is None and q_net_out is None:
    raise ValueError('Both value_net and q_value_net are None.')

  if categorical_value and max_abs_value is None:
    raise ValueError(
        'The max absolute value (`max_abs_value`) representable'
        'by the categorical value fn must be set when using the '
        'categorical value fn (`categorical_value` = True).'
    )

  if value_net_out is not None:
    t, b = value_net_out.shape[:2]
  else:
    t, b = jax.tree_util.tree_leaves(q_net_out)[0].shape[:2]

  rewards, actions = rollout.rewards, rollout.actions
  if isinstance(rollout, types.ActorRollout):
    env_discounts, mu_logits = rollout.discounts, rollout.logits
  else:
    assert isinstance(rollout, types.UpdateRuleInputs)
    env_discounts = 1.0 - rollout.is_terminal
    mu_logits = rollout.behaviour_agent_out['logits']

  # Check shapes.
  chex.assert_tree_shape_prefix((pi_logits, actions), (t, b))
  if drop_last:
    chex.assert_tree_shape_prefix((rewards, env_discounts, actions), (t, b))
  else:
    chex.assert_tree_shape_prefix((rewards, env_discounts), (t - 1, b))

  # Extract scalar values from network outputs
  values, q_values = extract_scalar_values_from_net_out(
      value_net_out=value_net_out,
      q_net_out=q_net_out,
      pi_logits=pi_logits,
      categorical_value=categorical_value,
      max_abs_value=max_abs_value,
      nonlinear_transform=nonlinear_transform,
  )

  if target_value_net_out is None and target_q_net_out is None:
    target_values = values
    target_q_values = q_values
  else:
    target_values, target_q_values = extract_scalar_values_from_net_out(
        value_net_out=target_value_net_out,
        q_net_out=target_q_net_out,
        pi_logits=pi_logits,
        categorical_value=categorical_value,
        max_abs_value=max_abs_value,
        nonlinear_transform=nonlinear_transform,
    )

  # Preprocess rollouts.
  if drop_last:
    rewards = rewards[:-1]
    env_discounts = env_discounts[:-1]

  # Always drop the last action (update rule and value fn both get all actions.)
  actions = jax.tree.map(lambda x: x[:-1], actions)

  # Actor rollouts are from mu.
  rho = importance_weight(
      jax.tree.map(lambda x: x[:-1], pi_logits),
      jax.tree.map(lambda x: x[:-1], mu_logits),
      actions,
  )

  # Get value targets and value outs
  if q_values is not None:
    # Estimate state-action values.
    chex.assert_rank(q_values, 3)
    value_outs = estimate_q_values(
        rewards,
        actions,
        env_discounts,
        rho,
        values,
        target_values,
        q_values,
        target_q_values,
        discount,
        lambda_,
    )
  else:
    # Estimate state values, returning dummy q values.
    assert value_net_out is not None
    value_outs = estimate_values(
        rewards,
        actions,
        env_discounts,
        rho,
        values,
        target_values,
        discount,
        lambda_,
    )

  # Calculate normalized advantages.
  advantages = value_outs.adv
  qv_advantages = value_outs.qv_adv
  tds = value_outs.td
  q_tds = value_outs.q_td

  if adv_ema_state is not None and adv_ema_fn is not None:
    new_adv_ema_state = adv_ema_fn.update_state(
        advantages, adv_ema_state, axis_name
    )
    normalized_adv = adv_ema_fn.normalize(advantages, new_adv_ema_state)
    # Traverse only up to actions to avoid going inside adv_to_dict structure.
    normalized_qv_adv = jax.tree.map(
        lambda _, x: adv_ema_fn.normalize(x, new_adv_ema_state),  # pytype: disable=wrong-arg-types
        rollout.actions,
        qv_advantages,
    )
  else:
    new_adv_ema_state = None
    normalized_adv = jax.tree.map(jnp.zeros_like, tds)
    normalized_qv_adv = jax.tree.map(jnp.zeros_like, qv_advantages)

  if td_ema_state is not None and td_ema_fn is not None:
    # Update stats using q_td if q_net is provided. Otherwise, use value td.
    if q_net_out is None:
      # Normalize TD for state values.
      new_td_ema_state = td_ema_fn.update_state(tds, td_ema_state, axis_name)
      normalized_td = td_ema_fn.normalize(
          tds, new_td_ema_state, subtract_mean=False
      )
      normalized_q_td = jax.tree.map(jnp.zeros_like, q_tds)
    else:
      # Normalize TD for state action-values.
      new_td_ema_state = td_ema_fn.update_state(q_tds, td_ema_state, axis_name)
      normalized_q_td = td_ema_fn.normalize(
          q_tds, new_td_ema_state, subtract_mean=False
      )
      normalized_td = jax.tree.map(jnp.zeros_like, tds)
  else:
    new_td_ema_state = None
    normalized_td = jax.tree.map(jnp.zeros_like, tds)
    normalized_q_td = jax.tree.map(jnp.zeros_like, q_tds)

  value_outs.normalized_adv = normalized_adv
  value_outs.normalized_qv_adv = normalized_qv_adv
  value_outs.normalized_td = normalized_td
  value_outs.normalized_q_td = normalized_q_td
  return value_outs, new_adv_ema_state, new_td_ema_state


def estimate_values(
    rewards: chex.Array,
    actions: chex.ArrayTree,
    env_discounts: chex.Array,
    rho: chex.Array,
    values: chex.Array,
    target_values: chex.Array,
    discount: float = DEFAULT_DISCOUNT,
    lambda_: float = DEFAULT_TD_LAMBDA,
) -> types.ValueOuts:
  """通过 v-trace 获取状态值和 rollout 的价值估计。

  Args:
    rewards: 包含奖励的 [T, B] 数组。
    actions: 包含动作的 [T, B] 形状的数组树。
    env_discounts: 包含剧集终止标志的 [T, B] 数组。
    rho: 包含重要性权重的 [T, B] 数组。
    values: 包含状态值的 [T+1, B] 数组。
    target_values: 包含状态目标值的 [T+1, B] 数组。
    discount: 标量折扣因子。
    lambda_: v-trace 的跟踪参数。

  Returns:
    包含价值和（未归一化）优势估计的 value_outs。
  """
  chex.assert_rank([values, target_values], [2, 2])  # [T, B]
  chex.assert_equal_shape(
      [values[:-1], target_values[:-1], rho, rewards, env_discounts]
  )
  # Vmap for batch dimension.
  batch_vtrace_fn = jax.vmap(
      functools.partial(rlax.vtrace_td_error_and_advantage, lambda_=lambda_),
      in_axes=1,
      out_axes=1,
  )

  discounts = env_discounts * discount
  chex.assert_equal_shape([discounts, values[1:], target_values[1:]])

  vtrace_return = batch_vtrace_fn(
      target_values[:-1],
      target_values[1:],
      rewards,
      discounts,
      rho,
  )
  value_target = vtrace_return.errors + target_values[:-1]

  # Assign dummy qv_adv/q_target due to the lack of Q-values
  dummy_q_value = jax.tree.map(jnp.zeros_like, actions)
  dummy_qv_adv = jax.tree.map(jnp.zeros_like, actions)
  dummy_q_target = jax.tree.map(lambda _: jnp.zeros_like(value_target), actions)
  dummy_q_td = jax.tree.map(lambda _: jnp.zeros_like(value_target), actions)

  value_out = types.ValueOuts(
      adv=vtrace_return.pg_advantage,
      value=values,
      target_value=target_values,
      rho=rho,
      value_target=value_target,
      td=value_target - values[:-1],
      qv_adv=dummy_qv_adv,
      q_target=dummy_q_target,
      q_value=dummy_q_value,
      target_q_value=dummy_q_value,
      q_td=dummy_q_td,
  )

  return value_out


def estimate_q_values(
    rewards: chex.Array,
    actions: chex.ArrayTree,
    env_discounts: chex.Array,
    rho: chex.Array,
    values: chex.Array,
    target_values: chex.Array,
    q_values: chex.ArrayTree,
    target_q_values: chex.ArrayTree,
    discount: float = DEFAULT_DISCOUNT,
    lambda_: float = DEFAULT_TD_LAMBDA,
) -> types.ValueOuts:
  """通过 Retrace 从值和 rollout 中获取动作值估计。

  Args:
    rewards: 包含奖励的 [T, B] 数组。
    actions: 包含动作的 [T, B] 形状的数组树。
    env_discounts: 包含剧集终止标志的 [T, B] 数组。
    rho: 包含重要性权重的 [T, B] 数组。
    values: 包含状态值的 [T+1, B] 数组。
    target_values: 包含状态值的 [T+1, B] 数组。
    q_values: 包含动作值的 [T+1, B, A] 数组树。
    target_q_values: 包含目标动作值的 [T+1, B, A] 数组树。
    discount: 标量折扣因子。
    lambda_: Retrace 的跟踪参数。

  Returns:
    包含价值和（未归一化）优势估计的 value_outs。
  """

  chex.assert_rank([values], [2])  # [T, B]
  chex.assert_rank([jax.tree_util.tree_leaves(q_values)[0]], [3])  # [T, B, A]
  chex.assert_equal_shape([values[:-1], rho, rewards, env_discounts])
  chex.assert_equal_shape_prefix(
      (values, jax.tree_util.tree_leaves(q_values)[0]), 2
  )
  q_a = jax.tree.map(lambda x: utils.batch_lookup(x[:-1], actions), q_values)
  target_q_a = jax.tree.map(
      lambda x: utils.batch_lookup(x[:-1], actions), target_q_values
  )

  # Vmap for batch dimension.
  batch_retrace_fn = jax.vmap(
      functools.partial(
          rlax.general_off_policy_returns_from_q_and_v,
          stop_target_gradients=True,
      ),
      in_axes=1,
      out_axes=1,
  )

  discounts = env_discounts * discount
  chex.assert_equal_shape([discounts, values[1:]])

  def _add_first(x):
    return jnp.concatenate([jnp.zeros_like(x[:1]), x], axis=0)

  # Add a dummy first step to format correctly for Retrace.
  r = _add_first(rewards)
  d = _add_first(discounts)
  clipped_rho = jnp.minimum(rho, 1.0)
  lambda_rho = lambda_ * clipped_rho
  c_t = lambda_rho
  q_target = jax.tree.map(
      lambda q: batch_retrace_fn(q, target_values, r, d, c_t),
      target_q_a,
  )

  # Drop the dummy first step.
  q_target = jax.tree.map(lambda x: x[1:], q_target)
  chex.assert_equal_shape([q_a, q_target])

  # Calculate advantage from Q-values
  qv_adv = jax.tree.map(
      lambda x: x - jnp.expand_dims(target_values, axis=2), target_q_values
  )

  v_target = target_values[:-1] + clipped_rho * (
      jax.tree_util.tree_leaves(q_target)[0] - target_values[:-1]
  )
  adv = jax.tree_util.tree_leaves(q_target)[0] - target_values[:-1]

  q_td = jax.tree.map(lambda target, q: target - q, q_target, q_a)

  value_out = types.ValueOuts(
      adv=adv,
      value=values,
      target_value=target_values,
      rho=rho,
      value_target=v_target,
      td=v_target - values[:-1],
      qv_adv=qv_adv,
      q_target=q_target,
      q_value=q_values,
      target_q_value=target_q_values,
      q_td=q_td,
  )

  return value_out


def get_values_from_net_outs(
    x: chex.Array,
    categorical_value: bool,
    max_abs_value: float | None,
    nonlinear_transform: bool,
) -> chex.Array:
  """从网络输出中提取标量值。

  Args:
    x: 网络输出。
    categorical_value: 是否为分类值。
    max_abs_value: 最大绝对值。
    nonlinear_transform: 是否应用非线性变换。

  Returns:
    标量值。
  """
  chex.assert_rank(x, 3)  # [T, B, 1 or num_bins]
  if categorical_value and max_abs_value is None:
    raise ValueError(
        'The max absolute value (`max_abs_value`) representable'
        'by the categorical value fn must be set when using the '
        'categorical value fn (`categorical_value` = True).'
    )

  if categorical_value:
    v = value_logits_to_scalar(x, max_abs_value)
  else:
    v = jnp.squeeze(x, axis=2)
  if nonlinear_transform:
    return rlax.SIGNED_HYPERBOLIC_PAIR[1](v)
  else:
    return v


def extract_scalar_values_from_net_out(
    value_net_out: chex.Array,
    q_net_out: chex.ArrayTree | None,
    pi_logits: chex.ArrayTree,
    categorical_value: bool,
    max_abs_value: float | None,
    nonlinear_transform: bool,
) -> tuple[chex.Array, chex.ArrayTree | None]:
  """从网络输出中提取标量值。

  Args:
    value_net_out: 价值网络输出。
    q_net_out: Q 网络输出。
    pi_logits: 策略 logits。
    categorical_value: 是否为分类值。
    max_abs_value: 最大绝对值。
    nonlinear_transform: 是否应用非线性变换。

  Returns:
    包含标量值和 Q 值的元组。
  """
  if categorical_value and max_abs_value is None:
    raise ValueError(
        'The max absolute value (`max_abs_value`) representable'
        'by the categorical value fn must be set when using the '
        'categorical value fn (`categorical_value` = True).'
    )

  get_value_fn = functools.partial(
      get_values_from_net_outs,
      categorical_value=categorical_value,
      max_abs_value=max_abs_value,
      nonlinear_transform=nonlinear_transform,
  )

  if value_net_out is not None:
    values = get_value_fn(value_net_out)
  else:
    values = None

  if q_net_out is not None:
    # vmap over action dims [T, B, "A", ...]
    chex.assert_rank(jax.tree.leaves(q_net_out), 4)
    get_q_values_from_net_outs = jax.vmap(get_value_fn, in_axes=2, out_axes=2)
    q_values = jax.tree.map(get_q_values_from_net_outs, q_net_out)
  else:
    q_values = None

  if values is None:
    # Get state values from Q-values if values are not explicitly given
    pi_tree = jax.tree.map(jax.nn.softmax, pi_logits)
    values = jax.tree.map(
        lambda p, q: jnp.sum(p * q, axis=2), pi_tree, q_values
    )

  return values, q_values


def importance_weight(
    pi_logits: chex.ArrayTree,
    mu_logits: chex.ArrayTree,
    actions: chex.ArrayTree,
) -> chex.Array:
  """根据 logits 计算重要性权重。

  Args:
    pi_logits: 目标策略 logits。
    mu_logits: 行为策略 logits。
    actions: 动作。

  Returns:
    重要性权重。
  """
  log_prob_fn = lambda t, a: distrax.Softmax(t).log_prob(a)
  log_pi_a_tree = jax.tree.map(log_prob_fn, pi_logits, actions)
  log_mu_a_tree = jax.tree.map(log_prob_fn, mu_logits, actions)

  # Joint probs.
  log_pi_a = sum(jax.tree_util.tree_leaves(log_pi_a_tree))
  log_mu_a = sum(jax.tree_util.tree_leaves(log_mu_a_tree))
  rho = jax.lax.stop_gradient(jnp.exp(log_pi_a - log_mu_a))
  return rho


def value_logits_to_scalar(
    value_logits: chex.Array, max_abs_value: float
) -> chex.Array:
  """假设以零为中心的整数箱，将 logits 转换为标量。

  Args:
    value_logits: 价值 logits。
    max_abs_value: 最大绝对值。

  Returns:
    期望值。
  """
  if max_abs_value <= 0.0:
    raise ValueError(
        f'Max abs value must be greater than 0: {max_abs_value} <= 0.'
    )
  num_bins = value_logits.shape[-1]
  expected_values = rlax.transform_from_2hot(
      jax.nn.softmax(value_logits),
      min_value=-max_abs_value,
      max_value=max_abs_value,
      num_bins=num_bins,
  )
  return expected_values


def scalar_to_categorical_value(
    num_bins: int, value: chex.Array, max_abs_value: float
) -> chex.Array:
  """假设以零为中心的整数箱，将标量转换为 2-hot 概率。

  Args:
    num_bins: 箱的数量。
    value: 标量值。
    max_abs_value: 最大绝对值。

  Returns:
    概率。
  """
  if max_abs_value <= 0.0:
    raise ValueError(
        f'Maximum abs value must be greater than 0: {max_abs_value} <= 0.'
    )
  value_probs = rlax.transform_to_2hot(
      value,
      min_value=-max_abs_value,
      max_value=max_abs_value,
      num_bins=num_bins,
  )
  return value_probs


def value_loss_from_target(
    value_net_out: chex.Array,
    value_target: chex.Array,
    nonlinear_transform: bool = False,
    categorical_value: bool = False,
    max_abs_value: float | None = None,
) -> chex.Array:
  """计算标量每步目标的每步价值损失。

  Args:
    value_net_out: 价值网络输出。
    value_target: 价值目标。
    nonlinear_transform: 是否应用非线性变换。
    categorical_value: 是否为分类值。
    max_abs_value: 最大绝对值。

  Returns:
    每步损失。
  """
  if categorical_value and max_abs_value is None:
    raise ValueError(
        'The max absolute value (`max_abs_value`) representable'
        'by the categorical value fn must be set when using the '
        'categorical value fn (`categorical_value` = True).'
    )

  # No stop-gradient on target here, add before call if necessary.
  if nonlinear_transform:
    down_fn, _ = rlax.SIGNED_HYPERBOLIC_PAIR
    value_target = down_fn(value_target)

  if categorical_value:
    num_bins = value_net_out.shape[-1]
    value_target_probs = scalar_to_categorical_value(
        num_bins, value_target, max_abs_value
    )
    chex.assert_equal_shape([value_net_out, value_target_probs])
    loss_per_step = losses.softmax_cross_entropy(
        value_net_out, value_target_probs
    )
  else:
    values = jnp.squeeze(value_net_out, axis=-1)
    loss_per_step = 0.5 * jnp.square(values - value_target)
  return loss_per_step


def value_loss_from_td(
    value_net_out: chex.Array,
    td: chex.Array,
    nonlinear_transform: bool = False,
    categorical_value: bool = False,
    max_abs_value: float | None = None,
) -> chex.Array:
  """计算标量每步 TD 的每步价值损失。

  Args:
    value_net_out: 价值网络输出。
    td: 时序差分。
    nonlinear_transform: 是否应用非线性变换。
    categorical_value: 是否为分类值。
    max_abs_value: 最大绝对值。

  Returns:
    每步损失。
  """
  if categorical_value and max_abs_value is None:
    raise ValueError(
        'The max absolute value (`max_abs_value`) representable'
        'by the categorical value fn must be set when using the '
        'categorical value fn (`categorical_value` = True).'
    )

  values = get_values_from_net_outs(
      value_net_out,
      categorical_value=categorical_value,
      max_abs_value=max_abs_value,
      nonlinear_transform=nonlinear_transform,
  )

  chex.assert_rank(values, 2)  # [T, B]
  chex.assert_equal_shape([values, td])  # [T, B]

  # Construct a value target from (potentially normalized) TD.
  # loss = 0.5 * (value - stop_grad[value + TD])^2
  # Equivalently, loss = - value * stop_grad[TD]
  value_target = jax.lax.stop_gradient(values + td)
  return value_loss_from_target(
      value_net_out,
      value_target,
      nonlinear_transform=nonlinear_transform,
      categorical_value=categorical_value,
      max_abs_value=max_abs_value,
  )

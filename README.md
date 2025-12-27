# DiscoRL：发现最先进的强化学习算法

本仓库包含 Nature 出版物 *["Discovering State-of-the-art Reinforcement Learning Algorithms"](https://www.nature.com/articles/s41586-025-09761-x)* 的配套代码。

它提供了一个用于 DiscoRL 设置的最小 JAX 工具包，以及 *Disco103* 发现的更新规则的原始元学习权重。

该工具包支持：

-   **元评估 (Meta-evaluation)**: 使用 *Disco103* 发现的 RL 更新规则训练代理，使用 `colabs/eval.ipynb` 笔记本 [![Open In](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/disco_rl/blob/master/colabs/eval.ipynb)。

-   **元训练 (Meta-training)**: 从头开始或从现有的检查点元学习 RL 更新规则，使用 `colabs/meta_train.ipynb` 笔记本 [![Open In](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/disco_rl/blob/master/colabs/meta_train.ipynb)。

请注意，后续将不会积极维护此项目。

## 安装

设置 Python 虚拟环境并安装包：

```bash
python3 -m venv disco_rl_venv
source disco_rl_venv/bin/activate
pip install git+https://github.com/google-deepmind/disco_rl.git
```

该包也可以从 colab 安装：

```bash
!pip install git+https://github.com/google-deepmind/disco_rl.git
```

## 用法

代码结构如下：

* `environments/` 包含可与提供的工具包一起使用的环境的通用接口，以及 `Catch` 的两种实现：基于 CPU 的和 jittable 的；

* `networks/` 包括一个简单的 MLP 网络和 DiscoRL 模型的基于 LSTM 的组件，均在 Haiku 中实现；

* `update_rules/` 包含已发现规则、actor-critic 和策略梯度的实现；

* `value_fns/` 包含价值函数相关的实用程序；

* `types.py`、`utils.py`、`optimizers.py` 实现了工具包的基本功能；

* `agent.py` 是 RL 代理的通用实现，它使用更新规则的 API 进行训练，因此它与 `update_rules/` 中的所有规则兼容。

用法的详细示例可以在上面的 colab 中找到。

## 引用

请引用原始 Nature 论文：

```
@Article{DiscoRL2025,
  author  = {Oh, Junhyuk and Farquhar, Greg and Kemaev, Iurii and Calian, Dan A. and Hessel, Matteo and Zintgraf, Luisa and Singh, Satinder and van Hasselt, Hado and Silver, David},
  journal = {Nature},
  title   = {Discovering State-of-the-art Reinforcement Learning Algorithms},
  year    = {2025},
  doi     = {10.1038/s41586-025-09761-x}
}
```

## 许可证和免责声明

Copyright 2025 Google LLC

所有软件均根据 Apache 许可证 2.0 版 (Apache 2.0) 获得许可；
除非遵守 Apache 2.0 许可，否则您不得使用此文件。
您可以在以下网址获取 Apache 2.0 许可的副本：
https://www.apache.org/licenses/LICENSE-2.0

所有其他材料均根据知识共享署名 4.0 国际许可证 (CC-BY) 获得许可。
您可以在以下网址获取 CC-BY 许可的副本：
https://creativecommons.org/licenses/by/4.0/legalcode

除非适用法律要求或书面同意，否则根据 Apache 2.0 或 CC-BY 许可分发的软件和材料均按“原样”分发，不提供任何明示或暗示的保证或条件。
请参阅许可证以了解管理这些许可证下的权限和限制的特定语言。

这不是 Google 的官方产品。

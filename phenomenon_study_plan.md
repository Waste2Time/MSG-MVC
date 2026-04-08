# Phenomenon Study 设计文档（面向 Codex 实现）

## 0. 目的

本文档用于实现一个 **phenomenon study**，验证下面这条 motivation / problem statement 是否真实存在：

> 在多视图聚类中，真正影响性能的不是“所有困难样本”，而是“跨视图语义不一致且靠近 prototype 决策边界的样本”。  
> 这类样本一方面决定 cluster-level alignment 的边界质量，另一方面又最容易受到视图冲突和伪标签噪声影响。现有方法要么依赖全局一致性正则（如 RCI）稳定共享表示，要么粗糙地基于中心方向生成困难样本，无法区分“可信的边界困难样本”和“噪声导致的伪困难样本”。因此，需要一种基于 prototype 边界和跨视图一致性的可信困难样本挖掘机制，在不稳定样本中筛出真正能优化边界的 hard samples。

本 study **不是主实验**，目标不是追最终 SOTA 指标，而是验证“问题确实存在”。

---

## 1. 回答核心问题：phenomenon study 过程中是否需要 RCI？

### 结论
**需要分两层来看：**

### 1.1 核心 phenomenon existence study：**不应依赖 RCI**
原因：
- 你的目标是证明“boundary-conflict samples 这个问题本身存在”
- 如果一开始就用 RCI，RCI 会缓解视图冲突，可能把现象“抹平”
- 这样会削弱 problem statement 的说服力

因此，**核心现象证明实验建议在 `without RCI` 设置下完成**，或者至少把 `without RCI` 作为主 setting。

### 1.2 与你的 backbone 相关的 system-level relevance：**应加入 RCI 作为对照**
原因：
- 你还需要证明：即便有 RCI，这类样本仍然是关键样本，只是被部分缓解，不是被完全消除
- 这样才能说明你的新模块不是“重复造轮子”，而是在 RCI 之外补上“trusted boundary hard mining”

因此最终建议做 **三组 setting**：

1. **No-RCI backbone**：用于暴露现象  
2. **With-RCI backbone**：用于说明 RCI 只能缓解，不能区分 trusted-hard 和 pseudo-hard  
3. **With-lightweight-RPC (optional)**：如果你后续要替代 RCI，可加入轻量一致性模块作为补充对照

### 1.3 推荐策略
phenomenon study 主体优先跑下面两组：
- `Backbone w/o RCI`
- `Backbone w/ RCI`

写作时表述为：
- 无 RCI：现象更明显
- 有 RCI：现象仍存在，但程度减弱
- 说明问题不是“RCI 缺失的人为产物”，而是多视图聚类本身就存在的结构性问题

---

## 2. 与现有框架的关系

已知当前架构与 `MSG-MVC` repo 一致（由用户说明），本 study 不要求修改主 repo 的大体结构，只要求：

- 在现有 `Z -> K-means -> Q -> Enhance -> P` 流程上，导出中间统计量
- 在 cluster-level alignment 前后插入分析代码
- 做少量 mask / scoring / frozen-analysis 实验

**本 study 不需要先实现新的 hard mining 模块**。  
先证明问题存在，再设计 TPBM / trusted-hard 模块。

---

## 3. 要验证的三个命题

本 phenomenon study 需要验证三个命题：

### P1. Boundary-conflict samples 的确存在
即：在 learned prototype 边界附近，存在一类样本：
- prototype margin 小
- 跨视图 assignment disagreement 大
- 但局部密度不低（不是简单 outlier）

### P2. 真正影响 cluster-level alignment 的不是所有 hard samples，而是 boundary-conflict samples
即：相比普通 easy samples、普通 boundary samples、pseudo-hard noise，  
**boundary-conflict samples 的缺失/修正会更显著影响边界质量和聚类稳定性**

### P3. Naive hard mining 会混入 pseudo-hard noise
即：如果只按 hardness（如低 margin）选 hard samples，会把：
- 真正有价值的 boundary-conflict samples
- 以及无价值甚至有害的 low-density pseudo-hard noise
混在一起

因此需要 trusted boundary hard mining。

---

## 4. 总体实验设计概览

本 study 由一个 **controlled synthetic multi-view demo** + 一个 **MSG-MVC backbone analysis demo** 组成。

### Part A：Controlled Synthetic Demo
优点：
- 可控
- 有 oracle group label
- 能直观证明“问题存在”

### Part B：Real Backbone Analysis on Current Pipeline
优点：
- 与你的真实框架直接相关
- 能说明这个现象不是 synthetic-only artifact

建议论文里：
- 主现象证明：Part A
- 与真实模型关联：Part B

---

# Part A. Controlled Synthetic Demo

## 5. Synthetic 数据构造

### 5.1 共享语义空间
在 2D 空间中生成 3 个簇：

- class 1 center: `(-2.0, 0.0)`
- class 2 center: `( 2.0, 0.0)`
- class 3 center: `( 0.0, 2.8)`

每类采样 500 个点，共 1500 个样本：

```python
s_i ~ N(mu_yi, sigma^2 I), sigma = 0.7
```

### 5.2 定义 oracle boundary band
对每个语义点 `s_i`，计算到最近和次近真实中心的距离：

- `d1 = nearest_center_distance`
- `d2 = second_nearest_center_distance`

定义 oracle boundary margin:

```python
delta_gt = d2 - d1
```

若 `delta_gt < 0.5`，则视为 `oracle boundary sample`。

### 5.3 View 1 生成
```python
x1 = A1 @ s + eps1
```

其中：
- `A1`：旋转 + 缩放矩阵
- `eps1 ~ N(0, 0.1^2 I)`

### 5.4 View 2 生成
```python
x2 = A2 @ s + eps2 + r
```

其中：
- `A2`：与 `A1` 不同的线性变换
- `eps2 ~ N(0, 0.1^2 I)`

对于位于 boundary band 的样本，以概率 `p_conflict = 0.4` 注入 conflict 扰动：
- 找到其竞争类方向法向
- 沿竞争类方向推进一个小位移 `alpha in [0.4, 0.8]`

这样制造：
- 跨视图语义不一致
- 且这种不一致主要发生在边界带附近

### 5.5 注入 pseudo-hard noise
再生成 5% 到 8% 的 outlier：
- 从几个中心之间的低密度空洞区域均匀采样
- 经过相同 view transform

这类样本不是边界样本，但常常：
- margin 小
- 视图分歧大
- 本质是低密度噪声

---

## 6. Oracle 样本分组

对 synthetic 数据给每个样本标 group label：

### Group A: Easy-consistent
满足：
- 不在 boundary band
- 无 injected conflict
- 非 outlier

### Group B: Boundary-consistent
满足：
- 在 boundary band
- 无 injected conflict
- 非 outlier

### Group C: Boundary-conflict
满足：
- 在 boundary band
- 有 injected conflict
- 非 outlier

### Group D: Pseudo-hard noise
满足：
- outlier / low-density noise

这是整个 phenomenon study 的核心四组。

---

## 7. 模型运行设置

### 7.1 不跑完整大实验
目标是分析现象，不追最优性能。

### 7.2 推荐训练流程
用你当前 backbone 的简化版：

- 保留 encoder / clustering / K-means / Enhance / prototype P
- CCA 保留
- RCI 做两种 setting：
  - `w/o RCI`
  - `w/ RCI`

训练方式：
1. warm-up 20~30 epochs
2. 导出：
   - shared latent `z`
   - view latent `z^1`, `z^2`
   - assignments `q, q^1, q^2`
   - prototypes `P`
3. 冻结模型或做 very short analysis fine-tune

---

## 8. 核心统计量定义

### 8.1 Prototype margin
对每个样本的 fused assignment `q_i`：

```python
m_i = top1(q_i) - top2(q_i)
```

`m_i` 越小，表示越接近 prototype decision boundary。

### 8.2 Cross-view disagreement
定义 view disagreement：

```python
d_i = JS(q_i^1, q_i^2, q_i)
```

其中 JS 为 Jensen-Shannon divergence。

### 8.3 Local density
在 shared latent `z` 上计算 kNN 密度：

```python
rho_i = 1 / mean_distance_to_kNN(z_i, k=10)
```

或使用核密度估计，简化起见推荐 kNN 版本。

### 8.4 Assignment flip rate
跨 epoch 记录 assignment 变化：

```python
flip_i(t) = 1[argmax(q_i^t) != argmax(q_i^(t-delta))]
```

---

## 9. 子实验 A1：Existence Study

### 目标
证明四类样本在统计上不同，尤其：
- Group C = low margin + high disagreement + high/medium density
- Group D = low margin + high disagreement + low density

### 操作
1. 训练 backbone 到 warm-up 阶段
2. 计算所有样本：
   - `m_i`
   - `d_i`
   - `rho_i`
3. 按 oracle group 分组
4. 画箱线图 / violin 图

### 应输出的图
- `margin_by_group.png`
- `disagreement_by_group.png`
- `density_by_group.png`

### 期待结论
- A: 高 margin，低 disagreement，高 density
- B: 低 margin，低 disagreement，高 density
- C: 低 margin，高 disagreement，高 density
- D: 低 margin，高 disagreement，低 density

### 这一步要证明什么
低 margin 的 hard samples 不是同质的。  
真正对应你的 motivation 的，是 **Group C**，而不是所有低 margin 样本。

---

## 10. 子实验 A2：Importance Study

### 目标
证明真正影响 cluster-level alignment 的是 Group C，而不是所有 hard samples。

### 方法：Masking Study
对 cluster-level alignment loss，仅对某一组样本做 masking。

#### 具体操作
从四组中分别随机取相同比例样本（例如 20%）进行 mask：
- mask Group A
- mask Group B
- mask Group C
- mask Group D

然后对每种 mask：
- 从 warm-up checkpoint 开始
- 做 5~10 epoch 短暂 fine-tune
- 比较边界区域聚类质量变化

### 应记录指标
#### 10.1 Boundary-band ACC/NMI/ARI
只在 oracle boundary samples 上计算。

#### 10.2 Boundary disagreement
```python
D_B = mean(JS(q_i^1, q_i^2, q_i)) over oracle boundary samples
```

#### 10.3 Boundary prototype margin
```python
M_B = mean(top1(q_i)-top2(q_i)) over oracle boundary samples
```

### 期待现象
- mask Group C：边界质量下降最大
- mask Group D：影响小，甚至可能更好
- mask Group A：影响最小
- mask Group B：有影响，但通常不如 Group C

### 这一步要证明什么
真正决定边界质量的不是所有 hard samples，而是 **boundary-conflict samples**。

---

## 11. 子实验 A3：Selection Failure Study

### 目标
证明 naive hard mining 会选错对象。

### 三种样本选择策略
#### Strategy 1: Naive Hard
```python
score_naive(i) = 1 - m_i
```

#### Strategy 2: Boundary-Conflict Aware
```python
score_bc(i) = (1 - m_i) * d_i
```

#### Strategy 3: Trusted Boundary-Conflict
```python
score_trusted(i) = (1 - m_i) * d_i * normalize(rho_i)
```

### 评估方式
因为 synthetic 有 oracle group label，可直接评估 top-k selection 质量。

#### 11.1 Precision@k for Group C
top-k 中属于 Group C 的比例

#### 11.2 Recall@k for Group C
Group C 被选中的比例

#### 11.3 Contamination Rate by Group D
top-k 中属于 Group D 的比例

### 期待现象
- naive hard：可能 recall 尚可，但 contamination 高
- bc-aware：precision 提升
- trusted：precision 最高，contamination 最低

### 这一步要证明什么
如果只按 hardness 选样本，会混入大量 pseudo-hard noise。  
因此需要 trusted boundary hard mining，而不是 naive hard mining。

---

# Part B. MSG-MVC / Current Backbone 上的 Analysis Demo

## 12. 目的
说明上面的现象不是 synthetic-only artifact，而是在你的真实 pipeline 中也能观察到。

---

## 13. 数据与训练设置
选择 1~2 个最小公开多视图数据集即可，例如：
- Caltech-2V / 3V
- Scene-15 风格小数据集
- 或 repo 默认最容易跑的小数据集

不要求跑完整大实验，只要求跑：
- `w/o RCI`
- `w/ RCI`

每组 3~5 个 seed 即可。

---

## 14. 在真实 backbone 中构造 proxy groups

真实数据没有 oracle 四组标签，因此用 learned statistics 构造 proxy groups。

### 14.1 计算三类分数
对每个样本计算：
- margin `m_i`
- disagreement `d_i`
- density `rho_i`

### 14.2 分组规则
#### Proxy Easy-consistent
- margin 高于 70% 分位数
- disagreement 低于 30% 分位数

#### Proxy Boundary-consistent
- margin 低于 30% 分位数
- disagreement 低于 30% 分位数
- density 高于中位数

#### Proxy Boundary-conflict
- margin 低于 30% 分位数
- disagreement 高于 70% 分位数
- density 高于中位数

#### Proxy Pseudo-hard
- margin 低于 30% 分位数
- disagreement 高于 70% 分位数
- density 低于 30% 分位数

---

## 15. 子实验 B1：Real-Backbone Distribution Study

### 目标
在真实 pipeline 中验证 proxy groups 的分布和行为差异。

### 看什么
#### 15.1 Assignment flip rate by group
统计不同 proxy group 在相邻 epoch 间的 label flip rate。

预期：
- proxy boundary-conflict flip rate 最高
- proxy pseudo-hard 也高，但密度更低

#### 15.2 Cluster-level alignment contribution proxy
记录每组样本在 cluster-level loss 上的平均 loss / gradient norm。

预期：
- proxy boundary-conflict 对 cluster-level loss 的影响最大
- easy-consistent 最小

---

## 16. 子实验 B2：Masking / Reweighting Study on Real Backbone

### 方法
对 proxy groups 做小规模 intervention：

- 去掉一部分 proxy boundary-conflict
- 去掉一部分 proxy pseudo-hard
- 或对这两组分别加权

### 看什么
- boundary-related cluster margin
- view disagreement
- overall clustering trend（只做辅助，不是核心）

### 预期
- 减弱 proxy boundary-conflict，会让边界建模变差
- 减弱 proxy pseudo-hard，不一定坏，可能还更稳

---

## 17. RCI 在 backbone analysis 中怎么用

### 必跑两组
- `without RCI`
- `with RCI`

### 你要写出的结论
- 无 RCI：boundary-conflict 现象更明显
- 有 RCI：该现象有所缓解，但并未消失
- 说明 RCI 解决的是“全局一致性稳定”，不是“可信 hard sample 区分”
- 因而你的后续模块与 RCI 是互补关系，而不是重复关系

---

# 18. 实现细节建议

## 18.1 代码组织建议

建议新建目录：

```text
analysis/
  synthetic_demo.py
  backbone_demo.py
  metrics.py
  plotting.py
  grouping.py
  masking.py
  config_synth.yaml
  config_backbone.yaml
```

### `metrics.py`
实现：
- `compute_margin(q)`
- `compute_js_divergence(q1, q2, q)`
- `compute_knn_density(z, k=10)`
- `compute_flip_rate(assign_prev, assign_now)`

### `grouping.py`
实现：
- synthetic oracle grouping
- real proxy grouping

### `masking.py`
实现：
- random mask by group
- weighted loss by group

### `plotting.py`
实现：
- boxplot / violin plot
- scatter plot in latent space
- bar chart for drop comparison
- precision/recall/contamination curves

---

## 18.2 随机性控制
所有实验固定：
- numpy random seed
- torch random seed
- cudnn deterministic

每组至少跑 3~5 次，报告 mean ± std。

---

## 18.3 最小可视化清单

### Synthetic 部分
1. `synthetic_groups_scatter.png`
2. `margin_disagreement_density_boxplots.png`
3. `masking_drop_barplot.png`
4. `selection_precision_contamination.png`

### Real backbone 部分
1. `proxy_group_distribution.png`
2. `flip_rate_by_group.png`
3. `loss_or_grad_by_group.png`
4. `with_without_rci_comparison.png`

---

# 19. 论文里如何写现象结论

可以写成三条：

### Phenomenon 1
Boundary-conflict samples exist in multi-view clustering and are statistically distinct from both easy samples and pseudo-hard noisy samples.

### Phenomenon 2
Boundary-conflict samples contribute disproportionately to boundary quality and cluster-level alignment, while removing generic hard samples does not cause the same degradation.

### Phenomenon 3
Naive hardness-based mining is contaminated by low-density pseudo-hard samples, which explains the instability of direct hard-sample exploitation.

---

# 20. 最终你要从 phenomenon study 得出的核心论据

最终请确保 study 能支撑下面这段话：

1. 低 margin 样本不是同质集合  
2. 真正关键的是：低 margin + 高跨视图分歧 + 非低密度噪声  
3. 这类样本最影响 boundary quality  
4. RCI 只能缓解全局一致性问题，不能完成 trusted-hard selection  
5. 因此需要一种 **基于 prototype 边界和跨视图一致性的 trusted hard sample mining**

---

# 21. 建议先实现的优先级

## 第一优先级
- Part A / A1: Existence Study
- Part A / A3: Selection Failure Study

## 第二优先级
- Part A / A2: Importance Study

## 第三优先级
- Part B / backbone proxy analysis

如果时间紧，先把 A1 + A3 做出来，就足够支撑 motivation。

---

# 22. 给 Codex 的一句执行指令

可以把下面这句直接丢给 Codex：

> Please implement the phenomenon study described in this markdown file. Start with the synthetic demo (A1/A3), including data generation, oracle grouping, margin/disagreement/density metrics, and plotting. Then implement the real-backbone proxy analysis for the current MSG-MVC-style pipeline with and without RCI. Ensure reproducibility with fixed random seeds and save all figures and CSV summaries.


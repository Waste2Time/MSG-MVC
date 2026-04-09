# 下一步工作清单（给 Codex）

## 目标

先不要急着堆完整方法或追最终 ACC / NMI。当前最优先的目标是把 **motivation 对应的 phenomenon study 做扎实**，并且把 synthetic demo 中和故事冲突的部分修正掉。

核心主线只保留两句：

1. 在多视图聚类中，真正关键的不是所有 hard samples，而是 **靠近 prototype 决策边界且存在跨视图语义冲突的样本**。  
2. 现有 naive hard mining 会把这类样本与无效的伪困难样本混在一起，因此需要 **boundary-conflict-aware** 的样本选择；`trusted` 版本目前还没有 work，暂时不要强推。

---

## 一、当前结论：哪些已经成立，哪些还没有

### 已经部分成立
- `margin` 这条线是通的：  
  `C_boundary_conflict` 的 margin 最低，说明它确实更靠近 learned prototype 边界。
- `bc_aware` 在小 `k` 区域优于 `naive`：  
  说明“只看 hard 不够”，至少“boundary + conflict”这个方向是有信号的。
- scatter 图的几何位置基本合理：  
  `A` 在簇内部，`B/C/D` 主要在簇间过渡区。

### 还没有成立，甚至和预期冲突
- `disagreement` 没有支持故事：  
  现在反而 `A_easy_consistent` 的 disagreement 最高，这不应该出现。
- `D_pseudo_hard` 不像“低密度噪声”：  
  它现在更像“中心区域的混淆样本团”，不是 outlier/noise。
- `trusted` 版本比 `naive` 还差：  
  当前 trust 公式不能作为主卖点。

---

## 二、下一步优先级（必须按顺序做）

## P0：先修 synthetic phenomenon study，不要继续堆方法
在 synthetic demo 没讲通之前，不要再大规模调 backbone 或者继续叠 hard mining 模块。  
先把 “问题存在” 证明干净。

---

## P1：检查并修正 `disagreement` 的定义和实现

### 目的
让 `disagreement` 真正刻画“跨视图语义不一致”，并且让：
- `A_easy_consistent` 低
- `B_boundary_consistent` 中等或偏低
- `C_boundary_conflict` 高
- `D_pseudo_hard` 可以高，但不能主导全部现象

### Codex 要做的事
1. 检查 `disagreement` 的当前实现：
   - 输入到底是什么：`q^1, q^2, q` 还是 prototype similarity？
   - 是 JS divergence、KL、还是别的量？
   - 有没有 softmax / 温度 / 归一化不一致的问题？
2. 做一个最小 sanity check：
   - 人工构造两个完全一致的分布，确认 disagreement ≈ 0
   - 构造两个互相冲突的 one-hot 分布，确认 disagreement 高
3. 打印每组样本的若干 raw case：
   - 随机抽 `A/B/C/D` 各 5 个样本
   - 输出它们的 `q^1, q^2, q, margin, disagreement`
   - 手工确认数值和直觉是否一致
4. 如果当前定义不稳定，先换成最直接版本：
   - `disagreement = JS(q^1, q^2)`  
   暂时不要混 `q` 或额外复杂项

### 验收标准
修正后，组间统计至少满足：
- `mean(disagreement_C) > mean(disagreement_B)`
- `mean(disagreement_C) > mean(disagreement_A)`

如果还不满足，说明不是 metric 的问题，而是 synthetic generator 的问题。

---

## P2：重做 `D_pseudo_hard` 的生成逻辑

### 目的
让 `D` 真正代表“伪困难样本”，而不是“边界附近的一团正常混淆样本”。

### 当前问题
现在的 `D`：
- density 偏高
- 在 scatter 上成团且位置居中
- 更像 ambiguous hard，而不是 noisy hard

### Codex 要做的事
把 `D` 分成两种版本，分别生成并对比：

#### Version D1：低密度 outlier 型
- 从几个 prototype 之间的空洞区均匀采样
- 距离主流形有一定偏移
- 加入两个视图后仍保持低密度

#### Version D2：局部异常扰动型
- 从正常样本复制一份
- 只在一个 view 上加入强扰动，另一个 view 保持原样
- 让它看起来 hard，但不是真边界样本

### 建议
先同时保留 D1 和 D2，不要只押一个定义。  
后续可以统一称作：
- `D1_outlier_pseudo_hard`
- `D2_view_corrupted_pseudo_hard`

### 验收标准
至少有一个 `D` 版本满足：
- `density_D < density_C`
- scatter 上 visibly 偏离主簇或落在稀疏区
- 被 naive hard mining 选中的比例高于被 bc_aware 选中的比例

---

## P3：增大 `B/C` 的样本量，尤其是 `C`

### 目的
现在 `C=11` 太少，Precision@k 和箱线图都不稳。  
必须把 `C` 做到至少几十个，最好一百级别。

### Codex 要做的事
1. 放宽 boundary band：
   - 增大 `delta`
2. 提高 injected conflict 比例：
   - 在 boundary band 内，增加被扰动的概率
3. 可以只对最易混淆的类对注入冲突：
   - 例如 cluster 1 和 2 之间

### 建议目标
总样本数 1500 左右时，尽量做到：
- `B >= 80`
- `C >= 80`
- `D >= 80`

不是必须完全平衡，但不能再是个位数和十位数。

### 验收标准
重新统计后：
- `C` 不少于总样本的 5%
- `B + C` 足够支撑稳定箱线图和 top-k 曲线

---

## P4：重跑 phenomenon study，并只保留最可信的三组结论

修完 `P1-P3` 后，重做以下三类分析。

### Study 1：group-wise statistics
重新画：
- margin boxplot
- disagreement boxplot
- density boxplot

### 目标结论
- `A`：high margin, low disagreement
- `B`：low margin, relatively low disagreement
- `C`：low margin, high disagreement
- `D`：可能 low margin / high disagreement，但 low density 或明显偏离流形

---

### Study 2：selection quality
三种选择策略继续保留：
- `naive = 1 - margin`
- `bc_aware = (1 - margin) * disagreement`
- `trusted = 暂时可以不作为主结果，只保留探索版`

### 要看的指标
- `Precision@k for Group C`
- `Recall@k for Group C`
- `Contamination@k by Group D`

### 目标结论
先只要求：
- `bc_aware` 在小 `k` 上优于 `naive`
- `bc_aware` 的 contamination 小于 `naive`

注意：  
如果 `trusted` 还是不 work，不要强行写主结论。

---

### Study 3：intervention / masking
这是最关键但也最容易被忽略的一步。  
需要证明 `C` 确实更“重要”，而不只是“更容易被选中”。

#### 操作
在 cluster-level alignment 或类似损失上，对不同组分别做 masking：
- mask 一部分 `A`
- mask 一部分 `B`
- mask 一部分 `C`
- mask 一部分 `D`

每次 mask 相同数量，做短暂 fine-tune（例如 5~10 epochs）。

#### 看什么
- boundary-band performance
- boundary-band margin
- boundary-band disagreement

### 目标结论
如果 mask 掉 `C`，边界区域质量下降最多。  
这一步一旦成立，你的 motivation 才真正闭环。

---

## 三、关于 `trusted`，当前怎么处理

### 结论
当前 `trusted` 不要当主线。

### 建议策略
1. 主线先收缩成：
   - `naive hard` vs `boundary-conflict-aware hard`
2. `trusted` 先降级为：
   - appendix / exploratory result
   - 或者写成“current trust formulation is insufficient”

### 原因
从现有结果看：
- `trusted` 在 `precision_c` 上不优
- `trusted` 的 `contamination_d` 更高

强推只会拖垮叙事。

---

## 四、下一版论文叙事建议

在 Codex 继续写材料时，现阶段只允许用下面这个版本，不要写太满。

### 推荐表述
- 不是所有 hard samples 都 equally useful。
- 更关键的是 those near prototype boundaries and exhibiting cross-view disagreement。
- Naive hardness-based mining is contaminated by spurious ambiguous/noisy samples。
- Boundary-conflict-aware selection is therefore better motivated than plain hard mining。
- The trust formulation is still under investigation and is not yet the main contribution.

### 不要写的表述
- 不要写 “trusted boundary hard samples 已经被稳定区分”
- 不要写 “our trusted mining consistently outperforms naive”
- 不要写 “pseudo-hard 一定是低密度噪声”，除非 D 被重做并验证通过

---

## 五、建议输出文件（交给 Codex 的任务清单）

Codex 下一步建议生成这些文件：

1. `debug_disagreement.py`
   - 单独检查 disagreement 实现
   - 输出 raw case 样本

2. `regen_synthetic_groups.py`
   - 重写 synthetic generator
   - 支持 D1/D2 两种 pseudo-hard
   - 支持调 `delta` 和 conflict ratio

3. `analyze_group_stats.py`
   - 读取 synthetic 数据
   - 输出 group-wise summary
   - 画 boxplots

4. `analyze_selection_curves.py`
   - 比较 naive / bc_aware / trusted
   - 输出 precision / recall / contamination 曲线

5. `masking_study.py`
   - 实现 group masking short fine-tune
   - 输出 boundary-band 指标变化

6. `README_next_steps.md`
   - 记录参数、结论、未解决问题

---

## 六、验收顺序（必须按这个来）

### Round 1：只验 metric 和数据
- disagreement 修好
- D 重新定义
- B/C 数量足够

### Round 2：只验 phenomenon
- 统计图合理
- selection 曲线支持 `bc_aware`
- masking 支持 `C` 更关键

### Round 3：再考虑方法
只有 Round 2 成功之后，再回到 backbone 和 hard mining 模块设计。  
否则现在继续堆模型，只会把问题掩盖掉。

---

## 七、最小成功标准

只要下面三条同时成立，这条 motivation 就可以写进论文：

1. `C_boundary_conflict` 在 `margin` 上显著低于 `A`
2. `C_boundary_conflict` 在 `disagreement` 上显著高于 `A/B`
3. `bc_aware` 在小 `k` 上相对 `naive` 有更高 `precision_c`、更低 `contamination_d`
4. mask 掉 `C` 对边界区域质量影响最大

其中前 3 条是必须的，第 4 条是 strongest evidence。

---

## 八、一句话版 TODO

先别救 `trusted`，先把 **boundary-conflict-aware phenomenon** 做实。  
顺序就是：

**修 disagreement → 重做 pseudo-hard D → 放大 B/C → 重跑 selection → 做 masking 证明 C 最关键。**

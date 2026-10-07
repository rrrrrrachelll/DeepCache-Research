# 实验14A：DiT相邻step冗余剖析报告

日期：2026-10-07

## 1. 实验定位

实验13A.5已经终止SD1.5 U-Net上的cache-depth调度。本实验按其下一步建议，将研究对象转为DiT内部规则化的token和transformer block。14A只做只读冗余剖析，不实施缓存，以避免再次把代理误差直接当作最终图像收益。

## 2. 配置与完整性

- 模型：`facebook/DiT-XL-2-256`，权重SHA-256与官方值一致。
- 硬件：Tesla M60 8GB；FP16；256分辨率；batch 1。
- 原生DDIM 20步，CFG 4.0，seed 1701。
- 类别：golden retriever（207）、otter（360）、balloon（417）、volcano（980）。
- 28个transformer block；每类记录2,800个step×block×component点。
- 4/4类别的hook输出与baseline逐像素完全一致，最大绝对差均为0。
- baseline耗时为7.104–7.156秒，平均约7.124秒；峰值显存1,854.7MiB。

## 3. 预设判据

一个step×block坐标只有在4个类别上同时满足以下条件才视为稳健候选：

1. 相邻step余弦相似度不低于0.995；
2. 相邻step归一化L2变化不高于0.10。

静态whole-block缓存还需要覆盖至少三分之一的transformer计算，才有达到1.5×理论加速的可能。

## 4. 聚合结果

| 缓存对象 | 稳健坐标 | 总坐标 | 覆盖率 | 最长连续step |
|---|---:|---:|---:|---:|
| whole-block残差 | 133 | 532 | 25.0% | 19 |
| self-attention输出 | 13 | 532 | 2.44% | 4 |
| MLP输出 | 67 | 532 | 12.59% | 17 |

主要连续区域：

- block 0残差在step 1–19全部通过；block 18残差在step 1–17通过。
- block 1和2残差均在step 1–13通过。
- block 27的MLP在step 1–17通过；block 1和2的MLP在step 1–12通过。
- attention冗余很少，最长区域仅为block 27的step 15–18。

## 5. 发现

1. **DiT中存在跨类别稳定的相邻step冗余。** 特别是少数block的残差或MLP输出具有长连续区间，这与SD1.5 U-Net中高度不稳定的单step分支收益不同。
2. **冗余高度集中，而不是全模型均匀存在。** 全部残差坐标中只有25%通过严格门槛；attention只有2.44%，不适合作为首个独立缓存对象。
3. **严格静态whole-block缓存无法满足1.5×目标。** 即使忽略VAE、调度器和缓存开销，并假设28个block成本相同，25%覆盖率的理论上限也只有`1/(1-0.25)=1.333×`，实际端到端速度只会更低。
4. **MLP是比attention更有希望的细粒度对象，但单独覆盖仍不足。** MLP稳健覆盖率为12.59%，需要与token选择或更宽松的残差复用联合使用。
5. **模型与硬件路径可行。** 1.85GB峰值显存说明8GB显存足以继续做缓存干预和多策略闭环，不需要先换更小模型。

## 6. 分支决策

14A通过“存在稳健连续冗余”和“只读插桩完整性”门槛，但没有通过“静态whole-block覆盖足以支持1.5×”门槛。

因此不应直接构造只覆盖严格候选的静态缓存表，也不应停止DiT方向。下一步进入实验14B，直接用最终图像质量和真实耗时检验三档残差复用强度：

1. `conservative-25`：只复用14A严格通过的133个坐标，作为质量上界与速度下界；
2. `target-35`：按四类最坏归一化L2变化排序，扩展到35%坐标，首次具备1.5×理论覆盖；
3. `aggressive-50`：扩展到50%坐标，验证速度上界和质量崩溃点；
4. 对三档策略加入step-shuffled控制，区分“选对位置”与“仅减少计算”；
5. 固定20步、同class和seed，测端到端速度、SSIM、LPIPS、DISTS。若`target-35`不能形成至少1.5×且感知损失可控的点，则停止静态block缓存，转向输入相关的top-k token refresh。

## 7. 产物

- 实验计划：`EXPERIMENT14A_PLAN.md`
- 采集脚本：`profile_dit_redundancy.py`
- 聚合脚本：`analyze_profile.py`
- 正式结果：`outputs/experiment14a_formal/`
- 冒烟结果：`outputs/experiment14a_smoke/`

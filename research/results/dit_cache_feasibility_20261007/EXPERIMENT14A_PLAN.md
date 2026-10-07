# 实验14A：DiT相邻step冗余剖析

日期：2026-10-07

## 背景与定位

实验13A.5已经触发预设停止条件，因此不再继续SD1.5 U-Net上的branch调度、误差校正器或13B/13C。本实验是13A.5之后的第一项新实验，研究对象改为DiT内部规则化的token与transformer block。

## 研究问题

在固定20步采样下，DiT不同block的完整输出、block残差、self-attention输出和MLP输出，是否在相邻扩散step之间存在稳定且可利用的冗余？

本阶段只做只读剖析，不实施缓存。这样可以先验证可缓存对象和时间区间，再用最终图像指标闭环验证缓存策略，避免把代理相似度直接当作质量收益。

## 配置

- 模型：`facebook/DiT-XL-2-256`。
- 分辨率：256；FP16；batch 1。
- 调度器：模型原生DDIM；20个固定采样step。
- CFG：4.0，因此transformer内部有效batch为2。
- 冒烟样本：ImageNet class 207，seed 1701。
- 正式剖析候选：4个类别，固定同一seed；冒烟通过后再执行。

## 记录量

对每个扩散step和每个transformer block记录：

1. block输入、输出和残差相对上一step的token平均余弦相似度；
2. block输出与残差的归一化L2变化；
3. 残差相对输入的范数；
4. self-attention与MLP输出的相邻step余弦相似度、归一化L2变化和相对输入范数；
5. 无hook端到端耗时、峰值显存；
6. hook前后同seed输出的逐像素最大差，验证插桩只读。

## 进入干预实验14B的门槛

满足以下条件才实施whole-block、attention-only、MLP-only和top-k token refresh：

1. 至少一个连续的step×block区域，在不少于3/4类别上组件相邻余弦相似度达到0.995；
2. 同一区域归一化L2变化不高于0.10；
3. 该区域覆盖的理论组件计算量足以支持至少1.5×的目标加速；
4. 只读hook与baseline图像逐像素一致。

若只有零散高相似点，停止静态缓存表，下一步改做输入相关的token选择；若连组件相似度也不稳定，则不在该DiT检查点上继续缓存方向。

# 实验 13：Solver-Stage-Aware Cache-Depth Scheduling

日期：2026-10-01

## 1. 定位

SACC、全局文本条件、扩容和多尺度校正器均未取得可用恢复率，因此不再预测或修补最终noise输出。实验13直接改变DeepCache复用的计算深度：保持8次完整刷新不变，在不同扩散阶段选择不同`cache_branch_id`。

该实验不重复之前的Adaptive V1。Adaptive V1通过增加刷新次数提高质量，平均使用9.67次完整UNet调用且比fixed3慢28.6%；实验13固定完整调用数为8，只重新分配复用步的局部计算量。

它也不同于matched encoder–decoder oracle。该oracle证明一致地重算浅层编码器和解码器可以恢复41%–46%误差，但只比完整UNet省35–104 ms。实验13使用DeepCache原生branch边界寻找更便宜的中间点。

## 2. 核心假设

在DPM-Solver++ 20步中，不同复用步对缓存深度的敏感性不同。使用统一branch 0过于激进；统一使用更深branch又会损失过多速度。

在相同8次完整刷新、相近端到端延迟下，按扩散阶段选择branch深度，可以比：

1. balanced8-A + 固定branch 0；
2. balanced8-A + 任一固定branch；
3. uniform9 + branch 0；

取得更好的图像保真度。

## 3. 固定配置

- Stable Diffusion v1.5，512×512，FP16，CFG 7.5。
- DPM-Solver++ 2M midpoint，20 steps，linspace，无Karras。
- 刷新位置固定为balanced8-A：`[0, 2, 5, 7, 10, 13, 16, 19]`。
- 所有策略均为8次完整UNet调用；不允许通过额外刷新换质量。
- 不使用校正器，不使用文本分类器，不训练大网络。
- 策略选择只使用原train prompts；原validation prompts只用于最终确认；development prompts保持未查看。
- 初始latent、prompt和seed在同一比较组内严格配对。

## 4. 实验13A：静态branch Pareto扫描

### 数据

- 从120个train prompts中按12类各选1个，共12个prompt。
- 使用训练seed 1701，共12条配对轨迹。

### 模式

- DPM20完整基线。
- balanced8-A下扫描有效branch ID。先用1条轨迹验证0–11的合法性，再选择覆盖前沿的branch，例如`0, 2, 5, 8, 11`。
- 每个branch记录真实执行模块、复用步耗时、端到端延迟、峰值显存和最终latent。
- 同一latent上增加full shadow，仅用于统计每个复用步的guided-noise error；shadow耗时不计入部署延迟。

### 输出

- 每个`step × branch`的平均误差、p10/p90和增量耗时。
- 静态branch的端到端SSIM、LPIPS、DISTS、PSNR和速度。
- 删除被其他branch同时在质量和延迟上支配的点。

### 完整性检查

- 所有refresh步误差必须为0。
- branch切换关闭时必须逐位复现对应静态branch结果。
- branch 0必须复现balanced8-A已有速度/质量量级。
- DPM20输出作为成对reference，不使用旧图像文件混合比较。

## 5. 实验13B：阶段branch调度构造

使用13A的train数据，不查看validation结果。

### 策略生成

对12个复用步，为每个可用branch估计：

- 误差收益：相对branch 0减少的guided-noise error；
- 计算成本：相对branch 0增加的CUDA时间。

在总缓存计算预算下做离散knapsack/动态规划，产生三档静态阶段策略：

- `depth_fast`：端到端估计耗时不超过branch 0的105%；
- `depth_mid`：不超过branch 0的115%；
- `depth_quality`：不超过branch 0的125%。

策略只依赖step index、log-SNR阶段和cache age，不依赖prompt内容。每个复用步的branch表在进入验证前冻结。

### 必须包含的控制

- 固定branch 0。
- 与每档策略延迟最接近的最佳固定branch。
- 把同一组branch深度随机置换到不同step的控制，检验收益是否来自正确阶段，而不只是更多计算。

## 6. 实验13C：train内闭环筛选

### 数据

- 从未用于13A的train prompts中按12类各选2个，共24个prompt。
- seed 1701，共24条轨迹。

### 模式

- DPM20。
- balanced8-A + branch 0。
- uniform9 + branch 0。
- 最佳固定branch。
- `depth_fast`、`depth_mid`、`depth_quality`。
- 最佳阶段策略的step-shuffled控制。

### 进入独立验证的条件

最佳阶段策略必须同时满足：

- 相对DPM20速度至少1.50×；
- 与最佳固定branch的端到端延迟差不超过5%；
- 相对等延迟最佳固定branch，SSIM至少提高0.010；
- LPIPS至少降低5%，DISTS至少降低3%；
- 至少8/12类别的SSIM改善为正；
- prompt-grouped bootstrap 95%区间的SSIM差下界大于0；
- step-shuffled控制显著差于正确阶段表，SSIM至少低0.005。

若没有策略通过，停止cache-depth调度分支，不进行validation确认。

## 7. 实验13D：独立validation确认

仅当13C通过后执行。

### 数据

- 现有30个validation prompts × 2 seeds，共60条轨迹。
- 策略表、branch集合、预算和阈值全部冻结。

### 模式

- DPM20。
- balanced8-A + branch 0。
- 13C中等延迟最佳固定branch。
- 13C胜出的唯一阶段策略。
- step-shuffled阶段控制。

### 成功门槛

- 速度至少1.50× DPM20；
- 相对等延迟固定branch：SSIM提升至少0.010且95%区间下界大于0；
- LPIPS降低至少5%，DISTS降低至少3%；
- 至少8/12类别和至少40/60轨迹的SSIM改善为正；
- p10单轨迹SSIM差不低于-0.01，避免少数严重退化；
- CLIP score相对下降不超过0.5%。

通过后才允许在未查看的development split上做最终泛化和盲评。

## 8. 结果判读

- 阶段策略优于参数匹配的固定branch和step-shuffled控制：证明solver阶段与缓存深度存在可利用的结构，可形成新方法。
- 只优于branch 0、但不优于等延迟固定branch：收益只是增加计算，不构成方法贡献。
- 不优于step-shuffled控制：branch位置不重要，不支持stage-aware论点。
- 所有原生branch都接近完整UNet才能恢复质量：说明SD1.5 U-Net上的DeepCache缺少有用中间Pareto点，应停止该模型上的缓存优化，转向其他架构。

## 9. 资源估计

- 13A：约1–2小时，主要由shadow full probes决定。
- 13B：CPU优化，数分钟。
- 13C：约1–2小时。
- 13D：仅通过后执行，约2–3小时。
- 峰值显存预计与现有DeepCache实验接近，8GB Tesla M60足够。

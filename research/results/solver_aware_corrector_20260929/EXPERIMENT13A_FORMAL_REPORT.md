# 实验13A正式结果：跨类别静态branch前沿与代理有效性

日期：2026-10-03

## 1. 实验设置

- 数据：12个train提示词，每个类别1条，统一使用未参加先导扫描的`_01`提示词。
- seed：1701；同一提示词的6种模式严格复用相同初始latent。
- 模式：完整DPM20，以及balanced8-A下固定branch `0, 1, 3, 8, 10`。
- 每个缓存轨迹使用同latent full shadow统计逐step guided-noise error，shadow不计入部署耗时。
- 最终latent统一由同一SD1.5 VAE解码；SSIM、PSNR、LPIPS、DISTS均与本次配对DPM20图像比较。
- 模式执行顺序按类别做Latin rotation，降低GPU热状态与运行顺序混淆。

完整性检查全部通过：72/72轨迹完成，每个缓存轨迹均为8次刷新和12次复用，刷新误差严格为0；所有同组latent哈希一致，无非有限输出或缓存状态异常。峰值显存约2.65 GiB。

## 2. 静态branch结果

| branch | speedup | reuse error | SSIM | PSNR dB | LPIPS↓ | DISTS↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.078× | 0.08809 | 0.62913 | 16.902 | 0.33635 | 0.17422 |
| 1 | 1.517× | 0.07874 | 0.64019 | 16.874 | 0.31094 | 0.16945 |
| 3 | 1.156× | 0.07095 | 0.65099 | 16.759 | 0.30637 | 0.16352 |
| 8 | 1.022× | 0.02680 | 0.66884 | 16.844 | 0.28771 | 0.15371 |
| 10 | 1.011× | 0.02610 | 0.67300 | 17.130 | 0.27525 | 0.14410 |

相对branch 0：

- branch 1：SSIM `+0.01106`，LPIPS降低7.55%，DISTS降低2.74%；速度仅略高于1.50×门槛。
- branch 3：SSIM `+0.02185`，LPIPS降低8.91%，DISTS降低6.15%；速度已降至1.16×。
- branch 8：SSIM `+0.03971`，LPIPS降低14.46%，DISTS降低11.78%；速度仅1.02×。
- branch 10：SSIM `+0.04387`，LPIPS降低18.17%，DISTS降低17.29%；速度仅1.01×。

因此静态加深branch被否定：质量确实改善，但除branch 1外均不满足1.50×速度线；branch 1又只提供有限且类别依赖明显的收益。

## 3. 阶段敏感性

branch 8相对branch 0在12个复用step、12/12类别上都降低了same-latent noise误差，但收益高度集中：

| step | mean error reduction | 正收益类别 | 进入每类top-4 | 增量耗时 |
|---:|---:|---:|---:|---:|
| 9 | 0.08074 | 12/12 | 5/12 | 2.408 s |
| 12 | 0.07317 | 12/12 | 6/12 | 2.495 s |
| 15 | 0.08590 | 12/12 | 6/12 | 2.571 s |
| 18 | 0.11560 | 12/12 | 12/12 | 2.626 s |

step 18是最稳定且单位耗时收益最高的候选；step 12、15和9构成次级候选。先导实验发现的晚期敏感性得到跨类别支持，因此stage-aware假设没有被否定。

## 4. 代理有效性检查

跨60个静态branch图像，轨迹平均noise误差与最终指标的Spearman相关性为：

- SSIM：`ρ=-0.238`；
- LPIPS：`ρ=0.310`；
- DISTS：`ρ=0.224`；
- PSNR：`ρ=0.055`。

使用同提示词相对branch 0的改善量消除提示词难度后，相关性仍偏弱：

- noise改善 vs SSIM改善：`ρ=0.257`；
- noise改善 vs LPIPS降低：`ρ=0.343`；
- noise改善 vs DISTS降低：`ρ=0.304`。

均未达到预先建议的`|ρ| ≥ 0.5`代理有效性门槛。另一个直接证据是branch 8和10的平均noise误差几乎相同，但branch 10相对branch 8在SSIM和LPIPS上9/12获胜、DISTS上10/12获胜。

结论：guided-noise error可用于发现候选敏感step，但不足以作为最终图像收益的数值代理。原实验13B若直接用它做knapsack，存在选错step或branch的风险。

## 5. 结论

1. 固定深branch路线失败：质量恢复需要接近完整UNet的计算量。
2. 阶段结构路线仍成立：step 18表现出强跨类别一致性，9/12/15是次级候选。
3. 原13B需要修正：不能直接从noise误差表冻结调度策略。
4. 在进入24条train闭环筛选前，需要增加一次直接的单step闭环干预实验。

## 6. 下一步：实验13A.5单step闭环干预

仍使用本次12个train提示词及已有配对DPM20/branch 0 reference，只新增下列动态策略：

- 低成本动作：branch 1仅用于step 15或18；
- 深动作：branch 8仅用于step 1、9、12、15或18；
- 深度对照：branch 10仅用于step 9、12、15或18。

除指定复用step外，其余复用步全部使用branch 0；刷新表仍为balanced8-A，完整刷新次数仍为8。step 1是等计算量的早期负控制。

每个策略直接生成最终图像并测量SSIM、LPIPS、DISTS和实际延迟，不再用noise误差代替图像收益。单step候选进入组合调度的条件：

- 相对branch 0，至少9/12类别SSIM为正；
- 平均SSIM至少提高0.005；
- LPIPS至少降低3%，DISTS至少降低2%；
- 晚期候选必须优于等计算量step 1控制；
- 单step策略相对完整DPM20仍至少1.50×。

通过后再依据直接图像收益构造`depth_fast/mid/quality`组合表，并在未参与13A/13A.5的train prompts上执行原13C及step-shuffled控制。若没有单step候选通过，停止SD1.5上的cache-depth调度分支。

## 7. 产物

- 生成程序：`run_branch_frontier_train12.py`
- 分析程序：`analyze_branch_frontier_train12.py`
- 原始轨迹与latent：`outputs/experiment13a_train12/data/`
- 解码图像：`outputs/experiment13a_train12/images/`
- 逐图指标：`outputs/experiment13a_train12/metrics.json`
- 聚合指标：`outputs/experiment13a_train12/metric_summary.json`
- 逐step统计：`outputs/experiment13a_train12/step_statistics.json`
- 代理相关性：`outputs/experiment13a_train12/proxy_correlations.json`

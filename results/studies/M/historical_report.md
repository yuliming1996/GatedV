# V-only：26.16M参数有限确认结果

十二条完整训练（六条等预算LR选型、六条新seed确认）和新持出评价完成。最终结论需final_results_audit.json为PASS。

|设计|LR|20270211 NLL / PPL|20270212 NLL / PPL|20270213 NLL / PPL|平均PPL ± SD|平均NLL ± SD|
|---|---:|---:|---:|---:|---:|---:|
|baseline|0.0006|3.068839 / 21.5169|3.053903 / 21.1979|3.065069 / 21.4359|21.3836 ± 0.1658|3.062604 ± 0.007768|
|v_silu|0.0006|3.025692 / 20.6083|3.024261 / 20.5788|3.025909 / 20.6127|20.5999 ± 0.0184|3.025287 ± 0.000895|

V-only平均PPL变化：-3.6648%；逐seed变化：[-4.222998216913498, -2.920702909352868, -3.840388169354303]。
改善seed：3/3；预声明实用线索：True（不是显著性检验）。
配对NLL差：[-0.043147594726178795, -0.02964204567251727, -0.03916075191227719]；seed均值t(df2)95%近似区间：[-0.054554295050086236, -0.02007929982389594]。
条件于三个seed的域聚类NLL差95%区间：[-0.03858312934607966, -0.03601062788705652]。每域一文，文档敏感性区间相同，不能当两份独立证据。

## 有限选型
LR网格 .0006/.0012/.0024；各臂最佳：{'baseline': 0.0006, 'v_silu': 0.0006}。单seed调参、网格边界均须结合全部六条开发终点评价；不作后续扩展。

## 实测成本
前向 V/base 比值（三seed各七轮配对中位数后平均）：1.007003。
训练步 V/base 比值：1.011433。完整42slot原始时间/内存均保留。

## 限制
- Only one scale, one corpus family, three paired confirmation seeds; not general ability or fewer-layer evidence.
- Single-seed LR selection with equal finite grid; boundary winners are not known global optima.
- Data/source/length/domain exclusions shift evaluation distribution; approximate duplicate filters do not guarantee semantic nonoverlap.
- Domain bootstrap conditions on three seed pairs; t interval assumes near-normal paired effects with df2.
- One document per domain means bootstrap partitions coincide; not independent uncertainty evidence.
- Mean-PPL ratio differs from exponentiated pooled NLL; neither practical threshold is a significance test.
- Shared-host costs, sampled RSS and overlapping CUDA/host counters; not isolated latency or autoregressive decoding.
- Nonlinear V has prior art; this study cannot establish a unique mechanism or originality.

## 成本与内存明细
|配对seed|前向比值（七轮中位）|训练步比值（七轮中位）|
|---|---:|---:|
|20270211|1.005978|1.008845|
|20270212|1.007199|1.013174|
|20270213|1.007833|1.012281|

下表单位GiB；各slot内存观察最大值的中位数/最大值。计数存在重叠，不能相加，RSS并非精确峰值。
|设计|RSS median / max|CUDA allocated median / max|CUDA reserved median / max|CUDA peak allocated median / max|CUDA peak reserved median / max|
|---|---:|---:|---:|---:|---:|
|baseline|2.0188 / 2.0288|0.7460 / 0.7519|1.8672 / 1.9023|1.7525 / 1.7584|1.8672 / 1.9023|
|v_silu|2.0180 / 2.0347|0.7458 / 0.7529|1.9961 / 2.0156|1.8569 / 1.8656|1.9961 / 2.0156|

同步更新耗时跨恢复累计，不含数据准备、评估、checkpoint及资源等待，不是整轮墙钟时间。
|设计|确认seed|同步更新秒数|
|---|---:|---:|
|baseline|20270211|22707.147|
|v_silu|20270211|22842.256|
|v_silu|20270212|22894.780|
|baseline|20270212|22640.883|
|baseline|20270213|22676.328|
|v_silu|20270213|22875.510|

## 全部选型终点
|设计|选型seed|LR|开发NLL|开发PPL|
|---|---:|---:|---:|---:|
|baseline|20270201|0.0006|3.07076258|21.558336|
|v_silu|20270201|0.0006|3.04474817|21.004741|
|v_silu|20270201|0.0012|3.14246918|23.160985|
|baseline|20270201|0.0012|3.16577496|23.707109|
|baseline|20270201|0.0024|3.32111737|27.691275|
|v_silu|20270201|0.0024|3.40828440|30.213366|

选定LR是否网格边界：{'baseline': True, 'v_silu': True}。所有选择阶段结果均保留。

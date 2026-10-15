# V-only：学习率下界和矩控制的有限补充

2026-09-28，用户授权执行此前建议的两组补充实验。此协议在任何新增正式训练之前冻结。本轮独立目录v_only_supplement_20260928，既有研究全部只读。

## 两组新增任务与确切预算

第一组：baseline与v_silu各补峰值学习率0.0012，配对seed20270111、20270112、20270113，共6条。

第二组：v_affine_match与v_silu_standardized各补峰值学习率0.0024，相同三个seed，共6条。baseline/v_silu在0.0024的对照使用上一轮已完成的对应seed结果，不重训。

共12条，每条从头32768步、batch8、context256，即67108864目标曝光；合计393216次AdamW更新、805306368目标曝光。复用旧seed不增加独立训练重复数量；新旧合并仍仅3个seed。保留既有24条全部正负结果，合并36条共1179648次更新、2415919104目标曝光，其中24条是历史证据，不属于新增预算。

新队列按seed交错两组，每组两个arm的先后随seed轮换。具体顺序由固定计划生成并冻结，不依结果调整。无早停、不选择中途最佳checkpoint、不延长、不自动新增学习率或seed。补充0.0012后即停止网格扩展，即使最佳值仍在边界，也只报告有限网格内的结果。

## 同一训练条件与历史证据

共同配方与v_only_mechanism_20260928保持一致：6层、width128、4头、FFN hidden384、vocab8192、context256，RMSNorm/RoPE/SwiGLU、tied embedding，2328192个可训练参数。V变换位于fused QKV投影之后、head reshape之前，Q/K维持线性。固定仿射系数为非训练buffer，不增加可训练参数。

batch8，warmup512，32768步cosine衰减至峰值LR的10%，AdamW betas=.9/.95、eps=1e-8、weight_decay=.1，clip1，threads2。FP32、TF32关闭、math SDPA，无autocast/compile。每2048步开发评价，每256步checkpoint，每步更新日志。主终点固定32768步。

只读使用architecture_exploration_20260921/data/fineweb_edu_v1的train/dev及原tokenizer。开发评价固定512窗、131072目标。本轮不打开历史holdout，不创建或评价新盲测集。

model.py、helpers.py、train.py、runtime.py以及calibrate.py从上一轮逐字节复用；控制面只适配本轮确切队列和证据绑定。校准manifest、128窗train采样表及逐seed常数直接复制并核SHA，不重新校准。其原始定义为baseline初始化每层原V=Z，S=SiLU(Z)，总体均值/总体标准差在128×256×128元素上以FP64累计；V_affine=aZ+b，其中a=sdS/sdZ、b=muS-a*muZ；V_standardized=c*SiLU(Z)+d，其中c=sdZ/sdS、d=muZ-c*muS。运行系数为FP32固定非训练buffer。

reports/prior_evidence.json在新训练前绑定上一轮completion、计划与封存、校准、独立训练/结果审计、逐seed结果汇总以及科学实现SHA。旧完成状态及证据绑定必须有效。新生成的每seed训练schedule、开发offset/doc_id SHA必须与旧baseline@.0024相同；初始化参数身份、校准常数、数据SHA和CUDA环境也须一致。新运行保留自身源/计划身份，不能冒充旧运行或续接旧checkpoint。

## 预声明分析和决策

必须先独立审计全部12条完整预算、身份、Adam更新计数、最终checkpoint、校准buffer、训练日程和逐文档开发数组。任何缺失/失败/未满预算都不能宣告完成。

在核验历史锚点和原始开发数组后，合并12条新结果与24条历史结果，明确各自来源。对baseline/v_silu报告4个LR=.0012/.0024/.0048/.0096的逐seed终点token加权NLL/PPL、算术均值、样本SD；逐LR计算V-only减baseline的配对NLL差、逐seed PPL百分比变化及均值PPL比值。对两个矩控制在.0024/.0048分别对同LR的baseline和原SiLU作相同配对报告，保留全部负结果。

baseline和v_silu分别按4个LR中三seed平均终点开发NLL最低选取探索性最佳配方，精确平局取较低LR；报告各自最佳之间的差异和是否为边界值。选择与评价使用同一已反复使用的开发集，存在选择偏差，不能作为无偏确认或新增盲测。控制只有两个LR，不能声称充分调优或与主要arm相同调参预算。对所有设计的均值/SD、配对差和统计输入SHA作独立复算。

预定资源分配线索：若双方各自最佳配方比较中，V-only均值PPL下降至少0.5%且3/3seed同向，可建议下一步中等尺度验证；这只是实用线索，不是显著性检验，不自动启动后续研究。若优势消失/方向混乱，优先研究优化条件。如果简单仿射对照在新LR追平SiLU，应弱化“非线性本身必要”的解释，不以本轮证明唯一因果机制。

原始与标准化SiLU、仿射对照的匹配仅在baseline初始化参考输入上成立，不是逐通道/训练中动态匹配；后续层输入、梯度和优化轨迹不同，偏移与尺度也未分离。剩余差异不能唯一归因为表达能力。固定三seed、2.33M模型、一语料家族和开发集复用限制外推；NLL/PPL不等于通用能力，全部6层不证明更少层有效。相关非线性V已有先例，不声称首创。

## 执行、资源与停止规则

共享容量允许其它工作并行：启动可用内存至少16GiB、运行至少8GiB，swap<=4GiB、磁盘>=25GiB、NVIDIA查询可用，单训练进程CUDA allocator上限12GiB，每15秒guard。仅持本研究锁，禁止重复启动，不停止其它研究。

正式训练前完成控制面CPU检查、独立审查和绑定本轮计划/源码的有限CUDA形状、有限梯度及真实SIGTERM退出75/新进程恢复验收。旧模型实现相同，但新控制面仍验证。只通过新pipeline/launcher执行正式训练。资源受控暂停保留checkpoint并按相同身份恢复；未知异常/NaN/OOM保存证据停止自动推进，不无限重试、删除失败记录或静默改配方。后台管线最多七天，不增加科学预算。

最终保留完整曲线、逐步耗时、端到端耗时、RSS和CUDA allocated/reserved采样。共享主机耗时不是隔离性能基准，内存计数不能相加且采样最大值不等于精确峰值。训练完成后独立审计及统计审计全部通过才写完成清单。12条结束后管线退出，不自动进入中等尺度、第二语料或付费云端。

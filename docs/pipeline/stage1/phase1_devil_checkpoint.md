# Devil's Advocate Report - Checkpoint 1

## Verdict: REVISE

当前选题抓住了一个值得研究的闭环反常现象，也正确拒绝把 100 m 首次进圈写成对接或稳定会合；固定时域、分层终端事件和三维通道干预也是合理方向。但现有 RQ 与方法蓝图尚不能支持可发表的因果结论。以下问题必须在进入文献检索和确认性仿真之前修正。

## Critical Issues (Blocks Progression)

1. **目标控制并未在比较臂之间真正保持不变**
   - **Type**: Method / Confounding
   - **Location**: `methodology_blueprint.md` 3.2、3.4、E3；`aerospace/simulation/nerm_ekf_sdre.py:147-161`；`aerospace/control/sdre.py:191-196`
   - **Problem**: 虽然逃逸方反馈状态使用真实相对状态，但其控制增益 `P` 来自追踪方估计状态构造的 `A_SDC`。因此 P、O 以及 A1-A5 干预会同时改变追踪方控制和逃逸方控制。所谓“估计误差通道效应”包含了环境/对手响应变化，不能归因于追踪方的状态估计误差进入反馈律。
   - **Impact**: RQ-1 的核心因果识别失效；“速度通道主导”或“`A -> P -> u_p` 通道可忽略”都可能只是逃逸方控制被同步改变后的总效应。
   - **Required fix**: 将非机动目标或预先生成、跨臂完全重放的 `u_e(t)` 设为主要因果实验；把闭环博弈作为单独敏感性层。若坚持博弈为主，必须分别定义并实现 `P_p`、`P_e`，使目标策略只依赖目标自身固定的信息结构，并增加“冻结目标控制”和“允许目标响应”两套估计量。每个 A0-A8 干预必须声明 `u_e` 是否逐时刻相同。

2. **“EKF/SDC filter”数学对象未定，SJ 对照按当前接口会是错误滤波器**
   - **Type**: Method / Construct validity
   - **Location**: `rq_brief.md` 研究对象与证据缺口 10；`methodology_blueprint.md` 3.4、E0、Required implementation changes；`aerospace/estimation/ekf.py:132-156`
   - **Problem**: 当前预测同时用 `F=I+A_SDC dt` 传播均值和协方差，`Q` 直接逐步相加，更新也未使用 Joseph 形式。`A(x)x=f(x)` 不代表 `A(x)=df/dx`。若只把 SJ 的 `A_SDC` 换成非线性 Jacobian，而仍计算 `F x`，均值传播将不再代表原非线性动力学。步长改变时固定 `Q` 还会改变隐含过程噪声强度。
   - **Impact**: 观测到的“估计误差效应”可能主要是离散化、协方差传播或错误对照造成的算法效应，无法推广为 angles-only EKF 的机理。
   - **Required fix**: 在蓝图中二选一并写出完整方程：其一，把 P 明确定义为 SDC-based filter，并增加标准 EKF 基线；其二，改成真正 EKF，即非线性均值积分、沿轨迹 Jacobian/状态转移矩阵传播协方差、连续时间噪声谱密度离散化、Joseph 更新和 PSD/对称性检查。SJ 不得通过把 Jacobian 填入现有 `predict(A,...)` 实现。E0 必须先验证均值局部截断误差、协方差离散化和 NIS/NEES 校准，再运行机理实验。

3. **主要执行器模型仍未决定，现象可能是无约束高推力伪影**
   - **Type**: Feasibility / External validity
   - **Location**: `rq_brief.md` 证据缺口 6 与 H6；`methodology_blueprint.md` 3.3、Stage 1 exit criterion 5
   - **Problem**: 现有先导结果的峰值指令约为 `1.956 m/s^2`，Delta-V 约 `10 km/s`，Monte Carlo 中甚至超过 `100 km/s`。蓝图仍把“无约束算法研究”与“工程约束研究”留作待选项，而中心 RQ 已使用 proximity operations 的工程语境。
   - **Impact**: 首达更早、制动时序和驻留失败都可能在合理推力上限下消失或反转；在执行器模型确定前，主要假设不可检验，论文的工程意义也不可成立。
   - **Required fix**: 确认性主分析必须预先指定一个有任务依据的 `u_max,p`、`u_max,e`、饱和范数、推力施加方式和采样/保持规则。无约束结果只能作为诊断上界。先做小规模 go/no-go 试验，确认 O 和 P 在物理约束下至少存在可分析的接近轨迹；否则先重新设计初始距离、控制权重或任务时域，而不是直接扩大 Monte Carlo。

## Major Issues

1. **RQ 与实验主线不一致且范围过宽**
   - **Type**: Scope / Method alignment
   - **Location**: `rq_brief.md` RQ-1 与子问题；`methodology_blueprint.md` 1.2-1.3、E1-E5
   - **Problem**: RQ-1 问的是估计误差通道的因果作用，蓝图的主要问题却变成 P 对 O 的性能比较，同时还要求 common-SDC 架构对照、全运行域、目标机动和失效分类。这实际上混合了机理论文、架构论文和成功域论文。
   - **Required fix**: 将主论文锁定为“在预先限定的 3-D 场景族中，哪些估计反馈通道导致 first-passage 与 sustained-approach 指标分离”。E1-E3 为确认性核心；SJ/common-SDC 性能和 E4-E5 广域映射降为有限稳健性或后续工作。把 P-O 差异写成现象和基准，不要取代通道级主 estimand。

2. **“full-state oracle 是信息上界”这一表述不成立**
   - **Type**: Logical inference
   - **Location**: `methodology_blueprint.md:73`
   - **Problem**: 真状态只消除了估计误差；同一个固定权重、局部 SDRE 控制律并不保证在首达时间、Delta-V 或驻留概率上优于带估计误差的闭环。当前反常现象本身已经反证了“各指标上界”的暗示。
   - **Required fix**: 全文改称 **no-estimation-error SDRE reference**。只有在指定代价和可证明排序下才能使用 upper bound。若要评价控制性能，应另加有竞争力且同约束的仅测角 GNC 基线；否则论文只能主张机制识别，不主张方法优越性。

3. **主终点与确认性检验族尚未唯一化**
   - **Type**: Statistical design
   - **Location**: `methodology_blueprint.md` 5.1、8.1-8.6
   - **Problem**: “驻留成功概率、`T_sustain`、边界速度”被并列为三个 primary outcomes，H1 又同时要求首达更快、边界速度更高、驻留概率更低。没有说明是联合成功、层级检验还是多重共同主终点。边界速度只在发生 first passage 的试验中定义，直接分析成功者会产生条件选择偏差。
   - **Required fix**: 指定一个主 estimand，例如固定时域内 P-O 的 sustained-soft-approach 配对风险差；其余按预注册的 gatekeeping 顺序检验。对 `v(T_FP)` 预先规定：仅描述所有首达者、仅分析双臂都首达的配对样本，或使用带失败惩罚的复合结局，并明确其解释限制。给 H1-H5 建立 outcome-contrast-multiplicity 对照表。

4. **阈值、驻留时间和等效界值仍属可移动目标**
   - **Type**: Construct validity / Researcher degrees of freedom
   - **Location**: `methodology_blueprint.md` 4.2-4.3、8.5、Stage 1 exit criteria 6
   - **Problem**: `0.1 m/s`、`0.05 T_orbit`、600 s、`0.1 T_orbit` 以及 Delta-V +/-2% 等界值尚无任务或数值依据。若在看到新结果后再决定，极易形成 moving goalposts。
   - **Required fix**: 在任何确认性运行前，从明确任务类别或权威标准中选定一个主阈值组，并将其他值标记为敏感性分析；等效界值必须由工程最小重要差异或 E0 数值误差上界推导，不能仅写“recommended provisional”。

5. **过程噪声和真值扰动没有生成模型**
   - **Type**: Method / Confounding
   - **Location**: `rq_brief.md` In scope；`methodology_blueprint.md` E0、E3、randomness contract
   - **Problem**: 当前真值动力学是确定性的，`Q` 只在滤波协方差中逐步加入。蓝图要求独立控制 process noise 和进行 NIS/NEES 校准，却没有定义真值侧扰动、连续时间谱密度、离散化或匹配/失配条件。
   - **Required fix**: 明确 `Q` 是调参矩阵还是物理过程噪声。若为后者，定义连续时间加速度噪声、真值注入方式及 `Q_d(dt)`；若为前者，不得把变化解释为过程噪声效应，并将一致性分析限定为调参敏感性。

6. **若关键现象复现失败，论文没有预先定义的可发表 null 路径**
   - **Type**: Novelty / Answerability
   - **Location**: `rq_brief.md` Stage 1 结论；`methodology_blueprint.md` H1-H5、Stage 1 exit criteria
   - **Problem**: 新颖性依赖“指标反转存在且可被通道干预解释”。若固定时域、正确滤波器和推力约束使 H1 消失，H3/H4 的机制论文随之失去中心贡献；当前没有停止或转向规则。
   - **Required fix**: 增加预注册决策树：先用独立确认集复现 metric divergence；若不复现，则停止“机制”主张。只有在跨预定场景得到有精度保证的等效/无效应边界时，才可转成 bounded negative result；否则该结果应作为内部模型纠错，不强行包装成论文贡献。

7. **计算与数据规模缺少现实预算**
   - **Type**: Feasibility
   - **Location**: `methodology_blueprint.md` E0-E5、required run bundle
   - **Problem**: E2、E3 的全因子与 E4/E5 合计可达数千次 10-orbit、逐步 ARE 仿真，同时要求保存完整时间序列。现有早停单次运行已约 7-55 s；固定时域、多步长复核会显著增加 CPU、存储和失败重跑成本。
   - **Required fix**: 给出每阶段精确 cell 数、最大运行数、预计 CPU-hours、并行策略与存储量。采用 gate：E0 通过 -> 小规模 H1 复现 -> A0-A5 -> 必要时 A6-A8 -> 受限 E4。探索阶段只存稀疏/失败触发时间序列，确认集再按预定比例保存全轨迹。

## Minor Issues

- `RA` 臂尚未指定距离噪声、偏置、采样率和与角噪声的相关结构；补齐后才能称为 measurement-information control。
- 首达时应报告 signed range rate `dot(rho)=r_hat dot v_rel`；若使用“closing speed”，应定义为 `-dot(rho)`，避免符号混淆。
- E1 固定场景下 O 为确定轨迹；重复同一个 O 不能制造场景层面的样本量。其置信区间只代表测量噪声序列，而非轨道/几何总体，应在 estimand 中明确。
- A1-A8 是闭环策略干预的总效应，不是自然估计误差的可加性中介分解；报告中应避免“解释了百分之多少误差”一类表述。
- dwell 判定需要对区间内连续成员资格做数值保证；仅对 10 s 网格点线性插值可能漏掉短暂出界。应定义稠密输出或验证过的细采样规则。

## Observations

- 两份文件对旧数据的证据边界处理得较好：已明确先导 CSV 不可作为可复现确认性证据，也没有把 2-D 通道排序外推到 3-D。
- 将 first passage、instantaneous soft arrival 和 sustained occupancy 分开，是当前设计最坚实的部分；即使最终机理假设不成立，这套构念区分仍应保留。
- “共同 SDC”更适合作为实现选择或次要消融，而不是在缺少性能收益时承担论文创新性。

## Strongest Counter-Argument

> 这篇研究观察到的不是“仅测角估计误差改变了 SDRE 的终端接近机理”，而是一个特定的一阶 SDC 滤波器、无约束高推力控制器和由同一估计依赖 Riccati 增益驱动的逃逸方共同产生的数值现象。修正滤波传播、冻结目标策略并施加物理推力约束后，所谓指标反转可能完全消失；若消失，当前设计既没有新控制方法，也没有足够广的负结果来支撑 Acta Astronautica 论文。

## What's Missing

- 一个不受追踪方估计影响的主要目标/扰动策略。
- 一个数学上完整且可实现的标准 EKF 或明确命名的 SDC-filter 定义。
- 一个物理可辩护的主执行器配置和小规模可行性结果。
- 单一主 estimand、检验层级、阈值依据和等效界值依据。
- H1 不复现时的停止/转向规则。
- 与目标期刊贡献相匹配的范围收缩和计算预算。

## Stress Test Results

| Test | Result |
|---|---|
| Remove strongest pilot evidence - does the RQ remain answerable? | **Yes, conditionally**: 可通过新 3-D 固定时域干预回答，但必须先修复三个 Critical 问题。 |
| Flip the research question - could estimation error have no systematic effect after correction? | **Yes**: 推力约束、正确 EKF 和固定目标策略都可能消除当前趋势。 |
| Apply to different context - does the finding generalize beyond the 866-km MEO stress case? | **No evidence yet**: E4 尚无具体分布、范围和确认数据。 |
| “So what?” - is significance justified? | **Not yet**: 只有在物理约束下仍存在任务指标误判，或得到精确的无效应/失效边界，工程意义才成立。 |
| Is a publishable contribution preserved under null results? | **No, as currently written**: 需要预先定义 bounded negative-result 路径及足够的精度和适用域。 |

## Required Revision Gate

在进入下一阶段前，修订版至少应完成以下五项：

1. 选择并固定主要目标策略，消除 `P(A_hat)` 对 `u_e` 的跨臂混杂。
2. 写出滤波器的连续/离散方程，并给出 SJ/标准 EKF 的合法实现契约。
3. 选定主推力约束、任务场景和 go/no-go 可行性标准。
4. 将论文范围收缩到 E1-E3，并冻结主 estimand、阈值、等效界值与检验层级。
5. 增加 H1 不复现时的停止或 bounded-null 转向规则及分阶段计算预算。

完成上述修订后再提交 Checkpoint 1 复审；当前不应进入文献检索或确认性实验。

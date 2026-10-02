# R6：非线性池化、历史分辨率与 Bayes 风险反转的近邻门槛

日期：2026-10-02。阶段：**新科学主线立项 CONDITIONAL，可进入已限定的开发验证；不是投稿 GO，也不代表录用保证。** 独占本报告及同名 JSON；未修改旧稿、冻结代码或数据，没有拟合模型、运行新实验或读取保留确认期结局。

## 调研结论

**最强可辩护增量是一个具体的预测知识命题：原始细、粗历史的信息嵌套，在非线性适配器之后可能变成不可互相恢复的观察；绝对细节与均值的混合在一个明确生成族内产生可计算的任务依赖风险交叉，再用幅值／方向分离的成对干预检验其现实边界。** “首次发现池化会损失信息”“抗混叠新模块”“均值／细节新架构”均不能成立。任务依赖的池化优劣是经典问题，具体的未来成对平方损失阈值与预测干预才是本轮值得检验的增量。[Boureau et al., 2010](https://icml.cc/Conferences/2010/papers/638.pdf)、[Blackwell, 1951](https://digicoll.lib.berkeley.edu/record/112749/files/math_s2_article-08.pdf)

检索分为四个问题：① max 的均值／绝对细节混合及池化统计理论；② 时间序列抗混叠与多尺度处理；③ 均值／细节保留或自适应池化是否已有；④ 非嵌套观察的决策风险解释是否属于经典理论。使用 agent-reach 的 Exa、内置网页检索、Jina 及作者链接代码；结论依据官方会议／期刊原文、作者仓库与官方软件文档。按相关性择取最接近原文，**不是系统综述或全领域首创证明**。没有搜索到本项目精确生成族和阈值的同文命题，表述仅限本次已读范围。

Route 为 SCI / Neurocomputing，文章类型为神经预测的理论解释与受控实证研究。期刊范围允许理论、学习方法、架构和应用；无需硬造新网络或套用经济学因果识别。近期 ms-mamba、ARFNet、MPFGSN 和 AMSFormer 的身份与访问边界沿用已核验原始台账；它们表明多尺度／频率建模已有大量近邻，而非证明本稿达到发表水平。[Elsevier 官方范围](https://shop.elsevier.com/journals/neurocomputing/0925-2312)

## 最接近的六项原始研究

| 来源与正式身份 | 实际已覆盖能力与本轮区别 | 本次阅读层次 |
|---|---|---|
| Y-Lan Boureau, Jean Ponce, Yann LeCun. **A Theoretical Analysis of Feature Pooling in Visual Recognition**. ICML 2010, pp.111–118. | 已解析 max／average 的统计分布、稀疏度、池化基数及优劣变化；不能原创声称池化选择依赖数据。其任务是视觉分类与可分性，并非固定成对历史的未来 Bayes MSE 阈值。 | 官方全文，重点 §§2–4；[原文](https://icml.cc/Conferences/2010/papers/638.pdf)。未添加未核定 DOI。 |
| Chen-Yu Lee, Patrick W. Gallagher, Zhuowen Tu. **Generalizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated, and Tree**. AISTATS 2016, PMLR51:464–472. | 已有固定可学习混合、输入响应的 sigmoid gate、树池化。二点窗口的 `mean + gate × absolute detail` 与其 mixed/gated 形式直接等价；不能改名即宣称新方法。 | 官方全文 §§3.1–3.2；[正式记录](https://proceedings.mlr.press/v51/lee16a.html)。 |
| Richard Zhang. **Making Convolutional Networks Shift-Invariant Again**. ICML 2019, PMLR97:7324–7334. | 将 maxpool 拆成密集非线性 max 与采样，之间低通处理，形成 BlurPool。已有非线性后下采样及抗混叠问题；其主要证据是图像移位稳健性，不是本稿的预测风险交叉。 | 官方全文 §3，方法与范围；[正式记录](https://proceedings.mlr.press/v97/zhang19a.html)。 |
| Heng Zhou, Xin Sun, Chao Li. **Anti-Aliasing Matters: A Dynamic Network for Time Series Forecasting**. ICML 2026, PMLR306:165788–165830. | **最直接的新近时间序列近邻 DMANet。** 其 Fourier/Nyquist 混叠框架、采样前过滤、采样后频域插值与频谱／噪声干预已经存在。本轮应区分绝对值的幅值折叠与 Fourier 混叠，不能声称首先研究预测中的抗混叠。 | 官方43页全文；重点主文 §§3–4、附录 A/B/G/I，并做全文关键字核查。未定位到本轮成对 max 恒等式／非嵌套未来风险阈值；这不是不存在其他同类工作之证明。[正式记录](https://proceedings.mlr.press/v306/zhou26aq.html)。 |
| Md Mahmuddun Nabi Murad, Mehmet Aktukmak, Yasin Yilmaz. **WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting**. AAAI2025,39(18):19581–19588. DOI10.1609/aaai.v39i18.34156. | 多层 DWT 近似／细节分支分别预测系数后重建；均值／细节保留、分离频率分支本身不是空白。两点 Haar 只是其更简单的已知变换族，不能包装为新分解。 | 官方全文方法与消融；[正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/34156)。 |
| Cristian Challu et al. **NHITS: Neural Hierarchical Interpolation for Time Series Forecasting**. AAAI2023,37(6):6989–6997. DOI10.1609/aaai.v37i6.25854. | 原文使用多速率 MaxPool、分层插值与回溯残差；官方实现已支持 MaxPool1d／AvgPool1d。简单改成平均池化不是新预测算法。它是本轮局部干预的研究对象，不能把整个网络说成只观察最后适配器坐标。 | 原文多速率采样／残差结构及官方代码文档；[正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/25854)、[软件参数](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html)。 |

上下文近邻同样不能绕开：**TimeMixer** 已平均下采样、保留 finest 分支并跨尺度融合；**TimeMixer++** 正式 ICLR2025原文 Eq.(1) 使用 stride-2 卷积形成多尺度，不能沿用 TimeMixer 把它写成平均池化；其 §§3.1–3.2 再做时间图像、季节／趋势与多分辨率混合。**Pathformer** 已有多尺度 patch 和自适应路径，**FreDF** 已有变换域监督。因此本稿不能把尺度选择、频域损失或一般残差双路包装为原创。后两项本次复用既有全文核验记录，不重新声称完成它们的代码复现。[TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)、[TimeMixer++正式记录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html)、[Pathformer](https://openreview.net/pdf?id=lJkOCMP2aW)、[FreDF正式记录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1457fb1e5d72cdc4ecd88bc10f916095-Abstract-Conference.html)

## 理论增量的准确边界

以下是对 root 提供命题的独立代数判断，不是已经完成的实验结果。

取独立、正方差的中心 Gaussian `C,A`，令 `D=|A|−E|A|`，`V=Var(D)`；细历史为 `(C−A,C+A)`，粗历史重复均值为 `(C,C)`。max 观察分别是 `C+|A|` 与 `C`。减去已知常数不改变信息，因此细 max 可写为 `U=C+D`。原始细历史仍能计算粗历史；**后适配器的 `U` 与 `C` 不再是同一嵌套关系**。这没有违反更多原始信息不能提高无限制最优风险的经典结论，变化发生在决策者被迫使用的表示中。[Blackwell, 1951 §§2–5](https://digicoll.lib.berkeley.edu/record/112749/files/math_s2_article-08.pdf)

未来 pair 以 `C+αD` 为均值、`βA` 为带符号细节，定义两端误差平方的平均。符号对称给 `E[A|U]=0`，记 `W=E Var(D|U)`，于是

- 粗 max 的 Bayes 风险：`α²V+β²Var(A)`；
- 细 max 的 Bayes 风险：`(α−1)²W+β²Var(A)`；
- 细减粗：`(α−1)²W−α²V`。

在非退化条件下 `0<W<V`；在 **α≥0（例如 α∈[0,1]）** 域内存在唯一交叉 `α*=sqrt(W)/(sqrt(W)+sqrt(V))∈(0,1/2)`。若不限定非负 α，还有一个负根，不能称全实数域唯一。`Mean/absolute-detail` 观察 `(C,D)`，仍丢失 A 的方向；`Mean/signed-detail` 或 Direct 观察能恢复两原始坐标。在无额外未来噪声的这一本构造下，可逆观察的最优风险为零；添加独立同分布未来噪声只加入共同不可约项。退化 `Var(C)=0` 等情况须排除或单独陈述。

**这个阈值是经典条件方差与实验比较思想的具体预测应用，增量是问题特定的命题及可检验分解；不是新 Blackwell 定理、新信息论或完整网络学习理论。** 目前尚未完成证明审计、解析／数值阈值验证或现实拟合验证。上述平方损失命题也不能直接充当 MAE 风险排序定理。

Blackwell 1951《Comparison of Experiments》原始馆藏由 Jina 获得10页 OCR全文，已读定义、风险比较和 sufficiency；OCR符号有误，不据它转抄精确公式。1953《Equivalent Comparisons of Experiments》正式 Project Euclid 元数据／摘要已核定：Ann. Math. Statist.24(2):265–272，DOI10.1214/aoms/1177729032；直接 PDF 返回访问阻断，没有声称读到其全文。[1953官方来源](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.full)

## 不足以作为新算法的等价改写

1. `max−|d|=mean`：就是平均池化。`max−E|d|`：已知常数的 affine平移，可被后续自由线性层的偏置吸收。它们可作为干预，不能产生新的表示能力。
2. `mean+g|d|`：已是 mixed／gated max-average pooling。若提出学习 gate，必须超越 Lee et al. 的已知形式，不能以应用到负荷数据为核心增量。
3. `(mean,signed-detail)`：缩放 Haar 可逆坐标。其第一自由 Dense 层可吸收变换；新信息来自相对单标量池化保留了第二坐标，而不是相对 Direct 创造了额外信息。额外通道／可逆坐标也不是免费压缩。
4. last-block 的均值、方向或 rank 性质不扩展为整网定理：早先 backcasts、last-value路径、mask及归一化可能提供其他信息。成对干预必须在实际残差坐标定义，时间反转后的细节符号不能误当原始时间方向。[NHITS原文](https://ojs.aaai.org/index.php/AAAI/article/download/25854/25626)、[Lee et al.](https://proceedings.mlr.press/v51/lee16a.pdf)

## 最小且不重复的能力检验

本段是设计建议，尚未执行。优先走**知识／诊断贡献**，新 adapter 作为干预工具，不预设算法优越。

- 分别检验 amplitude 影响未来均值、signed orientation 影响未来细节及纯 nuisance 三种受控响应，报告完整 α 曲线与理论阈值，不只挑有利点。采用相同日历历史、未来 pair产品和共同掩码；合成数据明确标识。
- 固定 Direct、max-repeat、mean-repeat、mean/absolute-detail、mean/signed-detail，以及按 Zhang 定义顺序实现的 BlurPool-style 匹配参考。BlurPool必须是密集非线性 max → 低通 → 采样；仅平均输入后取 max 应另命名。固定核、边界、相位、返回宽度和成本后再拟合。
- 共享学习张量、参数、种子、批次、训练及选择预算。保留 Direct 与 Haar补偿初始化的等价检查，排除仅坐标重参数化的解释；相同参数数量并不等于同函数类。
- 真实预测的36拟合方案（六 adapter×两历史×三 seed）可检验具体局部机制在一个新开发时期是否表现，但不能单靠它宣称跨骨干／跨总体普遍性。既有 R4和R5结局属于问题形成证据，不能重命名为新确认。后续确认须在冻结方向与分析后按真实数据暴露状态执行。
- 局部诊断应同时保存池化前 residual 的 `m,d`、幅值／方向干预、整网误差对比与旁路边界。真实数据不能用总体方差、冬季标签或误差相关性替代上述机制验证。MSE和MAE均可报告，理论对应前者，业务端点使用后者时须分开解释。

DMANet 已有自己的卷积／FFT网络与标准预测实验，不必在**纯局部机制**论文中为了数量复现全套 benchmark。如果转为新抗混叠架构或性能论文，DMANet、WPMixer与简单平均池化是直接近邻；仅比旧 NHiTS不足。作者链接仓库实际可访问，commit `9d21bd2ba3180b090a3307b7b7e2579d643a9d3f`，MIT许可；README与模型代码已静态阅读。没有运行其环境、下载其 benchmark或确认作者成绩；不能把代码可访问说成完整可复现。[DMANet作者仓库](https://github.com/HenryZhou2024/DMANet)

若未来希望走算法主线，最可行候选是**目标条件下的幅值／方向受限路径**，但要先证明它产生 Direct、平均池化、已知 gated pooling或wavelet分支没有提供的实际能力或稳定成本收益；当前尚无这种证据。应先完成本轮受控知识命题，避免再用新名称重复失败的 retention gate。

## 门槛决策与可追溯性

- **NO-GO**：把 max 恒等式、Haar双路、常数去偏、平均池化、普通 gate或一般抗混叠包装为新方法。
- **CONDITIONAL，科学主线立项**：特定幅值折叠下的非嵌套观察与任务依赖 Bayes 预测风险交叉，加上成对幅值／方向隔离、残差访问控制、全部方向结果和适用边界。它与上述最接近原文存在可清楚陈述的区别；是否值得投稿还取决于真实新增证据和主线意义。
- **暂停或转向条件**：命题仅剩初等 identity、现实结果只反映 Haar初值差异、没有幅值与方向可区分证据，或用已经知道的 R4反转代替新验证。即使 adapter不普遍胜出，准确的失败边界仍可保留；不能预承诺某个错误改善比例或高接收概率。

原始来源存于 `F:\AcademicData\Neurocomputing_OnlineRetention\raw\literature_v1\R6_20261002_v1`，新增不可覆盖文件；开始捕获前检查 D/E/F空余分别约135/163/3632 GB，新增论文小于25 MB，未使用E。脚本、模型与实验未新增。本报告 JSON保存实际源文件大小、SHA256、版本、阅读层次、访问失败及对应问题；公式部分明确标为本项目代数审视。agent-reach v1.5.0版本检查显示已是当前最新版本，未安装／升级工具。

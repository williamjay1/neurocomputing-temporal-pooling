# 公开代码与数据权利审计（2026-10-02）

## 当前发行范围说明

本审计区分“许可上可以公开”与“本次实际公开”。按当前作者授权，GitHub 发布代码、数据、结果和复现说明；完整主文与补充材料在本机供作者审核，不纳入 Git 跟踪。下表对文稿的 INCLUDE 表示权利上允许，不代表本次已上传。当前完整科学核心含源训练缓存，实际大小以 PUBLIC_RELEASE_FILE_MANIFEST.json 为准。

## 调研结论

- **发布判定：GO，针对整理后的当前核心复现包。** 将 R7 与 R6 当前核心文件展开至同一仓库，保留真实 UCI 派生数据、全部当前模型/预测/失败记录与独立核验。不要把本地完整历史 ZIP 当成已经完成权利整理的公开发行包。GitHub 普通 Git 拒绝超过 100 MiB 的单文件；本地 R7/R6 ZIP 分别为 313,289,564/305,833,952 bytes，展开后的 R6 非 ZIP 文件最大为 44,882,048 bytes，无需通过 LFS 保存当前核心。依据：[GitHub 大文件规范](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
- **UCI 当前主数据允许公开派生版本，许可为 CC BY 4.0。** 使用 Artur Trindade (2015), ElectricityLoadDiagrams20112014, UCI Machine Learning Repository, DOI 10.24432/C58C86；数据署名与代码作者署名分开。说明 80 客户选择、启用前零值屏蔽、DST 处理、标签与掩码和单位转换，保留来源链接、许可链接及修改说明。官方单位是每 15 分钟的 kW，转换为区间 kWh 才除以 4。依据：[UCI 官方记录](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
- **当前核心确实依赖第三方 NeuralForecast，而非全套独立手写 N-HiTS。** 脚本 130 的模型工厂和脚本 133 的独立模型构造均导入 `neuralforecast.models.nhits.NHITSBlock` 与 `_IdentityBasis`；独立实现的是适配器、记录验证及评分等流程，不应将库块归为作者独创。实际安装的 neuralforecast 3.2.2 元数据与其 LICENSE 均为 **Apache-2.0**，与官方仓库一致；不是 MIT。运行库不随公开包重分发时，固定依赖与归属说明即可；若复制其源代码，另须保留 Apache 许可与适用通知、标明修改。依据：[NeuralForecast 官方许可](https://github.com/Nixtla/neuralforecast/blob/main/LICENSE)。
- **SciencePlots 为 MIT，版权 John Garrett (2018)。** 本地 146 脚本使用本地 Matplotlib 实现，未安装/打包 SciencePlots；它明确参考 Nature 参数。保留出处，若分发复制的样式文件/实质代码须附 MIT 许可。Nature 风格参数不代表期刊背书。依据：[SciencePlots 许可](https://github.com/garrettj403/SciencePlots/blob/master/LICENSE)。
- **Zenodo DOI 由作者实际创建后填写。** 当前不会把仓库 URL、local ZIP 或未发布草稿写成 DOI。发布的 GitHub 仓库可用 `CITATION.cff` 提供作者和引用元数据；如同时存在 `.zenodo.json`，Zenodo 使用后者而忽略前者，二者必须保持一致。依据：[Zenodo GitHub 存档流程](https://help.zenodo.org/docs/github/archive-software/github-upload/)、[Zenodo 元数据规则](https://help.zenodo.org/docs/github/describe-software/)、[混合许可说明](https://help.zenodo.org/docs/deposit/describe-records/licenses/)。

## 详细说明：公开清单

| 文件范围 | 决定 | 说明 |
|---|---|---|
| 当前主文、四模块补充、矢量图/表、图表数据 | INCLUDE | 公开当前版本；科研主张保持与实际输出一致；作者身份按人类提供信息由主代理填写。 |
| 脚本 125、130–137 与 138–147 中实际存在的研究/核验/绘图脚本 | INCLUDE | 固定原研究代码身份；安装依赖另列。保留出处；无需打包 venv、系统库或下载的第三方源码。 |
| `datasets/upgrade5_uci_v1` 的 frozen panel、valid、组别/时间标签、manifest、normalization、roster | INCLUDE | 这些是 UCI 派生数据；CC BY 4.0 与来源署名须落到 DATA_LICENSE/归属文件。当前最大 `panel_kw.npy` 44,882,048 bytes；`valid.npy` 11,220,608 bytes。 |
| 当前 R6 36 个真实模型、30 个合成模型的权重、预测、指标、独立检查 | INCLUDE | UCI 数据与合成数据来源明确；原始结果不挑选。原记录内原主机路径可作为历史身份，不宣称已完成全流程异机重跑。 |
| R7 120 个 source-only 拟合及完整失败门槛/独立审计 | INCLUDE | 保留失败结果与停止条件；不得生成/宣传这些失败模型的外部验证结果。 |
| 当前理论支持状态、合成生成器、固定随机种子、LP/投影核验、统计输入输出 | INCLUDE | 清楚标识合成/解析构造及 post-review 分析，不能称真实人类实验或事前预注册。 |
| 历史 S4 所引用的 own scripts 与汇总指标 | INCLUDE LIMITED | 可按下面的资源许可保留科学汇总；不需要公开所有旧稿、私下评论或重复历史包。历史事实性结果统计不等于原测量再分发。 |
| 原 UCI 全量 ZIP/txt、London/SGSC 大型原始档、ETT 改造测量/权重 | EXCLUDE | 当前复现有 frozen UCI 子集，原档可从提供者获得；ETT 明确 ND 许可，沿用不分发改造记录和历史权重的谨慎边界。 |
| `historical/code_and_evidence_upgrade5_20261002.zip` 与外层 R6/R7 ZIP 本身 | EXCLUDE FROM GIT | 不上传这些闭合本地档作为普通 Git 大文件；整理需要的内容后公开，原档本地保留不修改。 |
| 私下 review_request.txt、粘贴评论、运行库、凭据、第三方论文 PDF、非公开个人信息 | EXCLUDE | 科学复现不需要这些材料；保留评论如何落实的自有决策摘要即可。作者公开姓名/所属/ORCID/通信信息按当前授权办理，其他个人身份材料排除。 |

## 历史数据及歧义

1. **Low Carbon London 2013 refactored：CC BY 4.0 已核实。** 4TU 数据页与由提供者登记的 DataCite `rightsList` 均确认，DOI 为 10.4121/fbbe775b-48d8-469f-a39b-b64488bfd6fd.v1；作者 Simon Tindemans，原收集/发布者 UK Power Networks 与 Imperial College London，Delft University of Technology 做重构。记录历史转换为半小时平均 kW（原 kWh ×2）与处理差异。[4TU 记录](https://doi.org/10.4121/fbbe775b-48d8-469f-a39b-b64488bfd6fd.v1)、[DataCite 权利元数据](https://api.datacite.org/dois/10.4121/fbbe775b-48d8-469f-a39b-b64488bfd6fd.v1)。
2. **London 原版：官方页写 Creative Commons Attribution，未展示版本。** 可保留原提供者链接与其确切标注；不要把原版 2014 自动登记成另一个 4TU 2013 版本的许可。若公开具体原版观测而非仅汇总，应保留获得时的资源许可证据及变更说明。更严格的 UK Data Service 调查版不是本研究这个开放来源，不能混淆。[London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)。
3. **澳大利亚 SGSC：数据集许可 CC BY 3.0 Australia。** 官方 CKAN `package_show` 返回 `license_title` 与 `license_url` 明确为该许可；网页每个资源及 Additional Info 同样标注。网页底部 CC BY 4.0 属门户通用版权，不覆盖资源的更具体 CC BY 3.0 Australia 标注。历史 rosters 与抽样原始值需归属 Commonwealth of Australia/SGSC 及项目数据来源，不能统一成作者 CC BY4 原创数据。[数据集](https://www.data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data)、[官方元数据 API](https://www.data.gov.au/data/api/3/action/package_show?id=4e21dea3-9b87-4610-94c7-15a8a77907ef)、[CC BY 3.0 Australia](https://creativecommons.org/licenses/by/3.0/au/)。
4. **ETT：官方仓库 CC BY-ND 4.0。** 历史下载 manifest 锁定提交 `1d16c8f4f943005d613b5bc962e9eeb06058cf07`，许可 SHA256 `83cbb5c31415b1d0ee1d20d35dae3fda3a045b85fae9364cc57facc9cd6e0946`。递归旧包中 ETT 数据只发现 12 份 validation/test 汇总 CSV；未发现 ETT 观测数组/预测数组/权重。公开自有实验代码和非重构原测量的指标摘要，保留数据来源；不公开转换后的测量与历史权重。权重是否构成许可意义上的 Adapted Material 不在本审计中一概断言，排除它们是当前发行的谨慎选择。[ETT 原许可](https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE)、[CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/)。
5. **CER/ISSDA：未发现用于当前主线或包含在递归 R6/R5 包的 CER 文件。** 若另有 CER 原始记录不得顺手放入此仓库；官方仅开放申请非商业研究/教学，不因第三方 GitHub 镜像带 MIT 标志而获得原数据再分发权。[ISSDA 官方条款](https://doi.org/10.7929/ISSDA/BX59EU)。

## 发行建议与剩余操作

推荐为作者自有研究脚本选 MIT，为自有文稿/图表/合成结果选 CC BY 4.0；UCI/London/SGSC 保留各自 DATA_LICENSE 与归属，不能因顶层 MIT 改写其数据权利。NeuralForecast Apache-2.0 依赖和 SciencePlots MIT 参考列入 THIRD_PARTY_NOTICES；第三方依赖采用安装，不将已安装运行库重发。若选择顶层 Apache-2.0 作为自有代码许可亦可，核心是分清代码与数据。Zenodo 官方支持混合许可，作者导入时核对各范围。[Zenodo 许可说明](https://help.zenodo.org/docs/deposit/describe-records/licenses/)。

当前两个核心档展开约 369,034,758 bytes（未含另行添加的元数据/历史精选），最大单文件小于 GitHub 50 MiB 警告阈值。因此直接提交实际核心文件比仅添加外置 release asset 或 LFS 指针更适合此任务。若把数据放 release 附件或 LFS，不能假定 Zenodo 最终 ZIP 自动含真正载荷；作者应核对归档下载内容与 manifest。GitHub 自动源码档是否包含 LFS 对象由仓库管理设置决定；本方案无需 LFS。[GitHub release 规则](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)、[Zenodo 存档操作](https://help.zenodo.org/docs/github/archive-software/github-upload/)。

本审计只读核验本地档案和官方来源，不创建 GitHub 仓库、GitHub release 或 Zenodo 记录。实际公开包还需一次文件清单/许可/机密扫描与上传后内容核对；这不是要求增加新实验，也不改变现有科学结果。

## 本地核验身份

- R7 本地 ZIP：376 项，SHA256 `2c1f279c1d8f652a7b2e7c20ab52421287f637729fd43bddb6f15cc90bb44a9e`。
- R6 本地 ZIP：499 项，SHA256 `ba23f9d6cfd5d55dc12a03c40f77b4f1739d6def4ebcb14a81f82b6c596206ca`。
- R6 内 R5 historical ZIP：1063 项；本次递归只读检索，无 CER 文件，ETT 只含汇总 CSV；可见 London/SGSC 抽样原观测值，因此不能把旧包整体称为无原测量。
- 安装 NeuralForecast 3.2.2 LICENSE SHA256：`d392f3d2969e354d7e6aef8fc6df5590ef5c20eea4b946191bd76a6138b4e81c`，Apache-2.0。

## 来源列表

- [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
- [4TU Low Carbon London 版本 1](https://doi.org/10.4121/fbbe775b-48d8-469f-a39b-b64488bfd6fd.v1)
- [4TU DataCite 登记权利](https://api.datacite.org/dois/10.4121/fbbe775b-48d8-469f-a39b-b64488bfd6fd.v1)
- [London Datastore 原版](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)
- [SGSC 官方数据](https://www.data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data)
- [SGSC 官方 API](https://www.data.gov.au/data/api/3/action/package_show?id=4e21dea3-9b87-4610-94c7-15a8a77907ef)
- [ETDataset LICENSE](https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE)
- [CER/ISSDA 官方条款](https://doi.org/10.7929/ISSDA/BX59EU)
- [NeuralForecast Apache-2.0](https://github.com/Nixtla/neuralforecast/blob/main/LICENSE)
- [SciencePlots MIT](https://github.com/garrettj403/SciencePlots/blob/master/LICENSE)
- [GitHub 大文件](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
- [Zenodo GitHub 存档](https://help.zenodo.org/docs/github/archive-software/github-upload/)
- [Zenodo 元数据](https://help.zenodo.org/docs/github/describe-software/)
- [Zenodo 混合许可](https://help.zenodo.org/docs/deposit/describe-records/licenses/)

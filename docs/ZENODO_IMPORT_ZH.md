# 作者亲自导入 Zenodo

当前仓库：https://github.com/williamjay1/neurocomputing-temporal-pooling

1. 登录你自己的 Zenodo，连接 GitHub，并为本仓库启用归档。
2. 核对作者 Junjie Zhang、两项上外单位、ORCID 和软件/数据分项许可。仓库已准备 `CITATION.cff` 和 `.zenodo.json`；两者一致。若二者并存，Zenodo 使用 `.zenodo.json`。
3. 启用仓库后，由你亲自建立 GitHub release，例如 `v1.0.0`。本次仅发布 GitHub 代码/数据，不预先替你建立 Zenodo 记录或触发 release。
4. 等 Zenodo 处理该 release 后，打开记录，检查实际 ZIP 中包含数据数组、模型、预测和清单。本仓库将这些实际文件直接提交 Git，不采用 LFS 指针或只有 release 附件的方式。
5. 核对数据来源署名、版本、作者信息和 DOI 对应的实际文件。把 Zenodo 生成的真实 DOI 发回，以补入论文 Data and code availability 和引用元数据。不要把软件 DOI 当作期刊论文 DOI。

官方说明：
- https://help.zenodo.org/docs/github/enable-repository/
- https://help.zenodo.org/docs/github/describe-software/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
- https://help.zenodo.org/docs/deposit/describe-records/licenses/

完整论文稿在作者本机更新，供作者审核；GitHub 当前发布的是复现代码、数据、结果和方法文档。历史旧 ZIP 在本地保留，未经整理的历史原始测量不在这个当前核心发行包中。

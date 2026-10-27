# 论文版本、期刊来源与补充材料核查

检查日期2026-10-27；时区Asia/Shanghai；HTTP原始访问时间见PRIMARY_ACCESS_RECEIPTS.json。使用指定xiaoju解释器；通过sys.path只读取Codex捆绑pypdf库，没有更换解释器、安装包或改环境。

|来源|实际取得与校验|可以确认|仍不能确认|
|---|---|---|---|
|[arXiv abstract](https://arxiv.org/abs/2506.00625)|实时页面列v1 2025-05-31、v2 2026-07-18；journal ref为指定DOI|v2是本次页面所列最新版本|不能仅凭accepted注释证明与期刊VoR逐字相同|
|[arXiv v2 PDF](https://arxiv.org/pdf/2506.00625v2)|在线HTTP200，12,559,590 bytes；SHA256 b68e80b98b31ad2a317f6462cab93e59f15ac4f456b5bbd1d23386339f3d5dc8；与历史缓存一致；17页；metadata含/arXivID=v2、作者和CC BY4.0|标题PI-H2T: Enhancing Long-Tailed Visual Recognition with Permutation-Invariant and Head-to-Tail Feature Fusion；全文逐页读取，关键6页渲染复核|PDF页眉仍写SUBMITTED；不能把其页码作为期刊页码|
|[Crossref DOI元数据](https://api.crossref.org/works/10.1109/TPAMI.2026.3711911)|HTTP200，13,581 bytes，SHA256 d4971420834fe6fde80f572e38a1683f34a3f91d734f8c1e973dea0ae22b8292|同题/作者的TPAMI DOI；created 2026-07-09，issue 48(11),13910-13924，published-print 2026-11；主URL=document/11601054|元数据15页范围不是实得PDF页数，不验证全文内容或附件|
|[IEEE正式页](https://ieeexplore.ieee.org/document/11601054/)|工具不可读；直接HTTP202、body0字节；标准PDF请求HTTP418；DOI直接请求曾SSL超时|已经实际尝试正式渠道，留失败状态|VoR PDF标题/页数/内容SHA、Code Availability正文及独立补充附件均UNVERIFIED|
|[作者arXiv TeX包](https://arxiv.org/src/2506.00625v2)|HTTP200，13,139,106 bytes，SHA256 13e7667687020a5e54c03dbb0994cd53c4a5edcd7c515ea28b88ab7c22780eff；79个tar条目、21份tex|正文Sec3/4、算法tab/algorithm.tex与附录A-E可读；代码URL只指向Keke921/H2T；无训练可执行文件|它是论文排版源，不是训练程序或checkpoint发布；不能替代IEEE独立supplement|
|[作者主页](https://keke921.github.io/publications/)|当前PI-H2T条目指向arXiv；AAAI条目区分论文/code|论文与作者仓库关联有第一手依据|在已读页面未找到额外完整PI发布，不作全网未发布断言|

原始网络沙箱中的DNS失败已用获准的直接只读请求重试；不能把DNS失败写成服务器404/403。GitHub tags连接器不支持的端点，也用真实git ls-remote补核，没有将端点拒绝转换为“没有tag”的证据。

## 版本和补充材料的使用约束

本报告所有方法公式、参数数量和42.19/49.22/49.04都只引用已实际读取的arXiv v2。没有将旧v1标题/页码、AAAI2024公式、期刊元数据或第三方摘要混入同一版本证据。正式期刊与v2是否内容一致的最终状态为UNVERIFIED，既不宣称相同，也不宣称有差异。若期刊/Supplement日后可读，需要先逐项比对Eq3/6/7/8、初始化和算法、更正及Table6/Fig9，再更新门禁。

v2本身可以确认的补充范围是PDF pp16–17的Appendix A-E及TeX包相应正文/算法/图片。全部物理页阅读范围与6页视觉核验记录见PAPER_PAGE_LEDGER.csv。算法line5与正文的差异同时存在于实得PDF和tab/algorithm.tex:13，已保留为歧义，没有自动修正文稿。

本地PDF路径：D:/xiangmuxunlian/xin/H2T_repro_audit/research_gap_20261026/results/evidence_20261026_01/paper_cache/PI_H2T.pdf；仅只读复用。新TeX包和渲染PNG只在本次primary_cache/保留，不提交原文。没有下载作者模型或数据集，也没有读取CIFAR TEST图像/预测。

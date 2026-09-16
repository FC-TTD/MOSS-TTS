# MOSS-TTS Hub 接入

2026-09-16，正式接管与真实验收已完成。基线为当前正式发布 `10273b9f6a31c3784c02201093c3366130b0bbbd`，源 `FC-TTD/MOSS-TTS`（upstream OpenMOSS/MOSS-TTS）。原生权重与框架、`deploy/formal-worker/runtime` 的原文件均不改；其 SHA 与 worker 当前四个挂载文件核对一致。

新增 `hub_runtime` 使用统一 SDK 的动态 GPU 进程、活动与生命周期；API 复制正式 moss_formal，三路 `/generate`、`/api/tts`、`/v1/audio/speech` 的 Form 类型、默认值、四个上传别名、参数处理与响应头保持。旧 `/model/unload` 返回 409，释放由 Hub 运维路径协调。语速与 expected_duration 沿原后处理算法，从 SDK 取得原通用 speed helper；不再启动旧健康/webhook。

`ui.py` 复制正式 `clis/moss_tts_app.py` 的 UI/辅助回调，保留27个控件、9个 Gradio API（生成仍 `/lambda`）、ASR辅助、时长参数与原根路径。API/UI shared Runtime；parent无模型权重、无独立TTL。原主模型 BF16/cuda:0，audio tokenizer CPU；该混合分工来自原部署，未经任何容量调整。

部署从固定基础镜像 `registry.ttd/moss-tts/fusion@sha256:d32ae9314890ee0a2140a39cde4845453c6868c14c1388a4628e0b24cd1b0505` 安装固定SDK wheel，不升级 Python3.12/Torch2.9.1/Transformers5.0/Gradio6.18/FastAPI0.136。原clis overlay作为固定文件COPY入镜像，避免掉回基础镜像内旧版。模型/音频tokenizer权重只读挂载，数据/cache仍 `/TTD-Data/openmoss/moss-tts-v15`。

`deploy/hub/compose.yml`：worker三卡显式UUID集合，GPU进程由租约选一张；CPU parent CUDA隐藏，私有control `127.0.0.1:13916`，service `moss-tts`，actor独立状态目录。初始预算20GiB来自原CUDA实占约17.3GiB及工作区，须上线后标定。辅助域名 `moss-pool-validation` 只用于交接验证；`business.yml`接管原 `moss-tts` API/UI域名。Gateway既有模型ID仍 `moss`。

Hub侧 `scripts/moss_release.py` 负责固定制品、增量catalog、兄弟模型维护ACK及原journal重启，保留Index/SVC的lease/actor/policy。旧MOSS停止必须正常结束实际执行，检查退出/显存证据；API/UI真实通过后才移交业务域名。回退先关闭并排空新实例、确认退休，保留关闭的MOSS登记和token用于账本对账，再恢复原后端；不删除历史、不复活已退休actor，Gateway先恢复原未托管配置。

本地5项CPU测试通过：formal OpenAPI/schema/全部调优参数/上传别名/时长后处理、原UI控件与api_name对照、真实HTTP加载/推理/Gradio queue与文件下载/进程卸载；原文件SHA不变。CPU fixture不代替GPU、实际模型输出和Gateway用量验收。最终上线结果由Hub仓库对应接入记录保存。


正式运行版本 927389a 已通过原域名 API、原生 UI、实际卸载/重载及 Gateway 持久用量验收。当前持久预算 20 GiB，实际新租约已使用。完整固定镜像、设备实测和回退边界统一记录在 Hub `docs/proposals/model-compute-pool/worker-expansion-2026-09-16.md`；此目录的初期准备描述保留为实现基线，不能覆盖最终验收。源码仅本地提交，未 Git push。

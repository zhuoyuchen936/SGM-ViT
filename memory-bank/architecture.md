# 项目文件架构文档

**项目**: SGM-ViT 硬件加速器协同设计
**路径**: `/home/pdongaa/workspace/SGM-ViT`
**最后更新**: 2026-04-20 (Phase 8 Pareto 完成)

本文档记录项目每个文件/目录的作用，便于快速定位和理解代码。

---

## 顶层结构

```
SGM-ViT/
├── core/              # 核心算法模块（推理流水线）
├── scripts/           # 训练、评估、数据处理脚本
├── hardware/          # 硬件架构建模
├── simulator/         # Cycle-accurate 仿真器
├── SGM/               # 经典 Semi-Global Matching 实现
├── Depth-Anything-V2/ # DA2 源码（手动拷贝 + 修改）
├── third_party/       # Phase 6+ 引入的外部依赖（MIT EfficientViT）
├── paper/             # 论文草稿和实验日志
├── artifacts/         # 训练产物（cache、checkpoint）
├── results/           # 评估输出（metrics、demo 图）
├── asserts/           # 测试图像
├── data/              # 数据集根目录链接
├── memory-bank/       # 项目记忆库（设计文档、进度、架构）
├── demo.py            # 主推理 demo 入口（Phase 1-5 通路；Phase 6+ 用 demo_phase7.py）
├── Makefile           # 构建/运行快捷命令
├── pyproject.toml     # Python 项目配置
├── requirements.txt   # 依赖列表
├── README.md          # 项目说明
└── PROJECT_STATUS_CN.md  # 中文项目状态
```

---

## core/ — 核心算法模块

| 文件 | 作用 |
|------|------|
| `pipeline.py` | **主推理流水线**。`align_depth_to_sgm` 全局 affine mono→SGM 对齐（Huber loss，line 611）；DA2 封装；Token Merge；W-CAPS 整合 |
| `fusion.py` | **融合策略库**。所有启发式融合 (soft_blend, edge_aware_residual, region_stable_detail 等) + **RCF 四阶段实现** (`segment_mono_regions` / `calibrate_regions` with Huber+sanity check / `blend_region_boundaries` / `fuse_region_calibrated`)。`FUSION_STRATEGIES` 字典为分派入口 |
| `fusion_net.py` | **学习型融合网络（Phase 1-5 历史）**。所有旧版 FusionNet 架构：FusionResidualNet, MaskResidualLiteNet, DetailRestoreV2Net, DirectFusionNet, **RegionCalibratedRefineNet** (Phase 5, 196K params) + `predict_rcf_refine` + `compute_fusion_net_losses` + `MultiDirFusionCacheDataset` + `FusionCacheDataset`。**Phase 6+ 起 EffViTDepthNet 迁至独立 `effvit_depth.py`**，本文件主要作为 RCF 历史和 Phase 1-5 ckpt 兼容保留 |
| **`effvit_depth.py`** | **Phase 6+ EffViT 端到端融合网络**（Phase 7/8 主力）。`EffViTDepthNet` (MIT EfficientViT backbone B0/B1/B2 × head_ch 24/48 可配) + 自制 FPN decoder + residual-on-mono；`build_effvit_depth_inputs` (7 通道 = RGB+mono+sgm+conf+sgm_valid) + `predict_effvit_depth` + `compute_effvit_losses` (smooth_l1 + λ·soft_D1, **无平滑项**) + `TAU_BY_DATASET` |
| `sgm_wrapper.py` | SGM 算法的 Python 封装。Phase 8 Fix C 起增加 `--large-penalty` / `--window-size` 等 CLI 参数（原 hardcoded） |
| `decoder_adaptive_precision.py` | W-CAPS 解码器自适应精度量化 |
| `token_merge.py` | ViT token 合并策略 |
| `token_reassembly.py` | Token 重组工具 |
| `sparse_attention.py` | GAS 稀疏注意力 |
| `stereo_datasets.py` | 数据集加载器 |
| `eval_utils.py` | 评估指标工具（EPE, D1, boundary_epe, flat_region_noise 等） |
| `viz.py` | 视差可视化 |
| `_paths.py` | 内部路径常量 |

### core/fusion.py 关键函数（RCF 相关）

```python
FUSION_STRATEGIES = {
    'soft_blend', 'hard_switch', 'outlier_aware', 'two_threshold',
    'edge_aware_residual', 'region_stable_detail',
    'region_calibrated',  # Phase 4+: RCF
}

# RCF 四阶段
def segment_mono_regions(mono, image_bgr) -> region_labels    # Stage A
def calibrate_regions(mono, sgm, conf, labels)                 # Stage B (Huber+sanity)
def blend_region_boundaries(calibrated, labels, detail, rgb)   # Stage C
def fuse_region_calibrated(sgm, mono, conf, rgb, ...)          # Top-level

# 辅助
def compute_detail_guidance(mono, sgm, conf, rgb)
def _huber_affine_fit(m, s, c)  # inside calibrate_regions
```

### core/fusion_net.py 架构清单（Phase 1-5 历史）

| 架构名 | 参数量 | 输入通道 | 用途 |
|--------|--------|----------|------|
| `legacy_single_head` | 500K | 10 | 旧版单头残差 |
| `mask_residual_lite` | 260K | 10 | Phase 1-3 主力 |
| `mask_residual_lite_multiscale` | 270K | 10 | 多尺度变种 |
| `dual_branch_masked` | 280K | 10 | 双分支试验 |
| `pyramid_edge_masked` | 300K | 10 | 金字塔试验 |
| `detail_restore_v2` | 280K | 10 | detail-teacher 试验 |
| `direct_fusion` | 1.95M | 12 | Phase 4 直接 blend |
| `rcf_refine` | 196K | 13 | Phase 5 RCF 精修 |

### core/effvit_depth.py 架构清单（Phase 6-8 主力）

| 变体 | 参数量 | GFLOPs @ 384×768 | 角色 | ckpt 路径 |
|------|--------|------------------|------|-----------|
| `b0_h24` | **0.735M** | 4.95 | 🥉 极致小（FPGA 起点） | `artifacts/fusion_phase8_pareto/b0_h24/mixed_finetune/best.pt` |
| `b0_h48` | 0.879M | 15.89 | — | `artifacts/fusion_phase8_pareto/b0_h48/...` |
| `b1_h24` | 4.703M | **9.85** | 🥈 性价比最佳（推荐默认） | `artifacts/fusion_phase8_pareto/b1_h24/...` |
| `b1_h48` | 4.853M | 20.85 | Phase 7 Iter 1 baseline | `artifacts/fusion_phase7_iter1/mixed_finetune/best.pt`（或 `fusion_phase8_b5/mixed_finetune/best.pt` = Phase 8 重训 7ch 版） |
| `b2_h24` | 15.042M | 22.03 | ETH3D/Mid 单项最佳 | `artifacts/fusion_phase8_pareto/b2_h24/...` |
| `b2_h48` | 15.198M | 33.09 | 🥇 最强精度 | `artifacts/fusion_phase8_pareto/b2_h48/...` |

**全 6 变体 8/8 超 heuristic**（Phase 8 Pareto 结果）。统一输入 7 通道（Phase 8 B5 从 8→7 剪除冗余 `sgm_pos`），residual-on-mono 输出。

---

## scripts/ — 训练、评估、数据处理脚本

### Phase 1-5 历史脚本（fusion_net.py 架构）

| 文件 | 作用 |
|------|------|
| `train_fusion_net.py` | Phase 1-5 FusionNet 训练主脚本。支持 `--stage {sceneflow,kitti,both,mixed}` |
| `eval_fusion_net.py` | 4 数据集评估主脚本；含 `compute_method_metrics` (Phase 5 复用) |
| `eval_merge_adaptive.py` | Merge + W-CAPS 评估；含 `build_sample_list_{kitti,eth3d,sceneflow,middlebury}` |
| `eval_fusion.py` | 融合策略评估（6 种启发式对比） |
| `eval_token_merge.py` | Token merge 消融评估 |
| `eval_latency.py` | FLOPs 与吞吐量分析 |
| `build_fusion_cache.py` | v1 fusion_cache 构建（Phase 1-5 用） |
| `precompute_sgm_hole.py` | SGM 预计算（旧版，hardcoded params） |
| `run_four_dataset_demos.py` | 4 数据集 demo 生成（Phase 1-5） |
| `search_fusion_arch.py` | 自动化架构搜索 |
| `generate_paper_figures.py` | 论文图表生成 |
| `generate_arch_figures.py` / `generate_arch_pptx.py` | 硬件架构图/PPT 生成 |
| `common_config.py` | 全局默认配置 (dataset roots, DA2 weights 等) |
| `CLAUDE.md` | 子目录级别指引 |

### Phase 6-8 EffViT 脚本（主线）

| 文件 | 作用 |
|------|------|
| **`train_effvit.py`** | **EffViT 训练主脚本**。`EffViTCacheDataset` (读 v3 cache 含 sgm_valid) + 两 stage (`sf_pretrain` / `mixed_finetune`) + hole-aug 增强 + per-sample τ 混合 batch。CLI `--variant {b0,b1,b2} --head-ch {24,48} --in-channels 7 --hole-aug-prob 0.5` |
| **`eval_effvit.py`** | **EffViT 4 数据集 eval**。从 ckpt 读 variant/head_ch/in_channels，输出每数据集 `effvit / heuristic / mono` 三列指标 JSON；含 flat_region_noise |
| **`build_fusion_cache_v3.py`** | **v3 cache builder**（Phase 7+）。读 tuned SGM (`/tmp/sgm_tuned_all/`) + DA2 推理 + `align_depth_to_sgm` + `fuse_edge_aware_residual` 重算 `fused_base`；新增 `sgm_valid` bool 字段；SF driving 枚举全 8 splits = 4400 pair |
| **`rerun_sgm_tuned.py`** | **统一 SGM 重跑**（4 数据集）。`DATASET_PARAMS` 为每数据集固定 range/P2/Win；支持 `--start-index/--end-index` 分片并行；输出 `/tmp/sgm_tuned_all/<dataset>/<name>.npz` 含 `disp_raw / disp_filled / hole_mask / confidence / pkrn_raw` |
| **`profile_effvit.py`** | **Phase 8 Stage 3** — 测 params + GFLOPs。fvcore 首选，thop fallback；输出 JSON |
| **`pareto_analyze.py`** | **Phase 8 Stage 4** — 6 变体 profile+eval JSON 合并 → `summary_table.md/csv` + `pareto_plot.png` (双联图：params-EPE / GFLOPs-EPE) |
| **`run_phase8_pareto.sh`** | Phase 8 auto-chain：串行跑 6 变体 (SF pretrain + mixed finetune + profile + eval)，内建 skip-existing；B2 自动降 batch 至 6/3 |
| **`demo_phase7.py`** | Phase 7 最终 demo。竖向 6 面板 + EPE/bad 标题：left/gt/mono/sgm/heuristic/effvit |
| **`demo_effvit_tuned_sgm.py`** | Phase 8 Fix C demo（7 面板对比 cache-filled SGM vs tuned raw SGM+holes） |

### Phase 8 Fix B 实验遗留（可作为 ablation 参考）

| 文件 | 作用 |
|------|------|
| `inject_tuned_sgm_eth3d.py` | ETH3D cache 就地注入 tuned SGM（disp_raw + holes=0 + sgm_valid） |
| `inject_tuned_sgm_middlebury.py` / `_v2.py` / `_v3.py` | Middlebury 三变体实验（v1: hole=0 / v2: hole=mono / v3: keep old conf）— 全部回归，最终回退到 original |
| `eval_effvit_eth3d_tuned_input.py` | ETH3D 逐样本 A/B probe：old vs tuned SGM 输入 |
| `rerun_sgm_eth3d_tuned.py` / `rerun_sgm_middlebury_tuned.py` | `rerun_sgm_tuned.py` 的早期专用版（被统一版本取代，保留参考） |

### Phase 5 遗留脚本（位于 /tmp，未进主项目）

| 文件 | 作用 |
|------|------|
| `/tmp/run_phase5_rcf.py` | Phase 5 训练：RCFCacheDataset + 两阶段 + cal_preserve/detail_gate losses |
| `/tmp/eval_phase5.py` | Phase 5 评估 |
| `/tmp/phase5_viz.py` | Phase 5 视觉对比 |
| `/tmp/eval_rcf.py` / `/tmp/run_rcf_demo.py` | RCF 启发式评估/demo |

---

## hardware/ — 硬件架构建模

| 文件/目录 | 作用 |
|-----------|------|
| `architecture/top_level.py` | 顶层加速器（28nm/500MHz，32×32 PE 阵列） |
| `architecture/memory_hierarchy.py` | L0-L3 存储层次 |
| `architecture/dataflow.py` | 数据流调度 |
| `architecture/interconnect.py` | 片上互联 |
| `pe_array/mhsa_pe.py` | MHSA PE (INT8 MAC) |
| `pe_array/conv_pe.py` | 卷积 PE |
| `pe_array/upsample_pe.py` | 上采样 PE |
| `pe_array/unified_sa.py` | 统一脉动阵列 |
| `scu/adcu.py` | ADCU (Absolute Disparity Calibration Unit) |
| `scu/crm.py` | CRM (Confidence Router Module) |
| `scu/dpc.py` | DPC (Data Path Controller) |
| `scu/fu.py` | FU (Function Unit) |
| `scu/gsu.py` | GSU (Gather-Scatter Unit) |
| `base_module.py` | 硬件模块基类 |
| `interfaces.py` | 硬件接口定义 |

---

## simulator/ — Cycle-accurate 仿真器

| 文件/目录 | 作用 |
|-----------|------|
| `core/simulator.py` | 主仿真器 |
| `core/event_simulator.py` | 事件模拟 |
| `core/event_queue.py` | 事件队列 |
| `core/scheduler.py` | 调度器 |
| `core/memory_controller.py` | 存储控制器 |
| `core/workload_dag.py` | 工作负载 DAG |
| `analysis/performance.py` | 性能扫描 |
| `analysis/energy.py` | 能耗分析 |
| `analysis/area.py` | 面积估计 |
| `analysis/roofline.py` | Roofline |
| `config/` | YAML 配置 |
| `run_simulator.py` | 仿真器入口 |
| `run_sparsity_sweep.py` | 稀疏度扫描 |

---

## SGM/ — 经典 Semi-Global Matching

| 文件 | 作用 |
|------|------|
| `SGM.py` | SGM 完整实现（Numba JIT） |
| `gaussian.py` | 高斯平滑工具 |
| `gen_config_param.py` | 参数配置生成器 |
| `stereo_config.c` | C 版配置（遗留） |
| `run.sh` | 运行脚本 |

---

## Depth-Anything-V2/ — DA2 源码

手动拷贝 + 修改（token merge hook 集成）。

---

## third_party/ — Phase 6+ 外部依赖

| 目录 | 作用 |
|------|------|
| `efficientvit/` | **MIT EfficientViT 官方源码**（github.com/mit-han-lab/efficientvit，Apache-2.0）。**本地裁剪**：`efficientvit/models/efficientvit/__init__.py` 只 `from .backbone import *`，删除 SAM/DC-AE/seg 头的 import 避免拉入 `segment_anything`/`omegaconf` 等重依赖。`EffViTDepthNet` 用 `efficientvit_backbone_b{0,1,2}(in_channels=7)` 构造 backbone |

**已通过 pip install 的新依赖**：`timm`（EffViT 内部 augment 用）、`onnx/onnxsim/onnxruntime`（trainer apps 链式 import 需要）、`omegaconf`（DC-AE 配置）。

---

## paper/ — 论文和实验日志

| 文件 | 作用 |
|------|------|
| `EdgeStereoDAv2_ICCAD.md` | 主论文草稿 |
| `CAPS_MERGE_V1.md` | CAPS-Merge 技术文档 |
| `DECODER_CAPS_V1.md` | 解码器 W-CAPS 文档 |
| `fusion_net_experiment_log.md` | FusionNet 实验日志 |
| `prior_experiment.md` | 已否定方向存档 |
| `review/` | 审稿反馈 |
| `ref/` | 参考资料 |
| `figures/` | 论文图表 |

---

## artifacts/ — 训练产物

```
artifacts/
├── fusion_cache/                   # v1 NPZ 训练缓存（Phase 1-6 用；Phase 6 Fix B 注入 tuned ETH3D SGM）
│   ├── kitti/{train,val}/              # 354 + 40
│   ├── sceneflow/{train,val,test}/     # 仅 35mm/sf/fast 1 split = 240 + 30 + 30
│   ├── eth3d/eval/                     # 27 （Phase 6 Fix B 已注入 tuned SGM）
│   └── middlebury/eval/                # 15 （Fix B 实验已回滚到 original）
├── fusion_cache_v3/                # Phase 7+ v3 cache（tuned SGM + sgm_valid 字段）
│   ├── kitti/{train,val}/              # 354 + 40
│   ├── sceneflow/{train,val,test}/     # driving 全 8 splits = 3517 train + 440 val + 439 test
│   ├── eth3d/eval/                     # 27
│   └── middlebury/eval/                # 15
├── fusion_phase{1,2,3,4}_*/        # Phase 1-4 历史 ckpt
├── fusion_phase5_rcf/              # Phase 5 RCF refined ckpt
│   ├── sceneflow_pretrain/best.pt
│   └── kitti_finetune/best.pt
├── fusion_phase6_iter{1..5}/       # Phase 6 EffViT 1-5 轮迭代 (8ch + v1 cache)
│   └── {sf_pretrain,kitti_finetune,mixed_finetune,polish}/best.pt
├── fusion_phase7_iter1/            # Phase 7：8ch + v3 cache + full driving → 8/8 超越 current+heuristic
│   ├── sf_pretrain/best.pt
│   └── mixed_finetune/best.pt
├── fusion_phase8_b5/               # Phase 8 B5：7ch 验证 ckpt（Phase 7 recipe 重训）
├── fusion_phase8_pareto/           # Phase 8 Pareto 6 变体（7ch）
│   ├── b0_h24/{sf_pretrain,mixed_finetune}/best.pt    # 735K params（最小）
│   ├── b0_h48/...
│   ├── b1_h24/...                                     # 性价比最佳
│   ├── b1_h48/...                                     # = phase7_iter1 复用
│   ├── b2_h24/...
│   └── b2_h48/...                                     # 15.2M params（最强精度）
├── fusion_arch_search/             # 架构搜索结果
└── backup_*/                       # 重构前备份
```

**tuned SGM 中间产物**（不在 `artifacts/` 里，位于 `/tmp/sgm_tuned_all/<dataset>/<name>.npz`）：Phase 7 预计算，含 `disp_raw / disp_filled / hole_mask / confidence`。`build_fusion_cache_v3.py` 读这些构造 v3 cache。

---

## results/ — 评估输出

```
results/
├── eval_fusion_phase{1,2,3,4}_*/   # 各 Phase 评估结果（Phase 1-5 历史）
├── eval_rcf_heuristic/             # RCF 启发式评估
├── eval_phase5_rcf/                # Phase 5 RCF refined 评估
├── eval_phase6_iter{1..5}/         # Phase 6 迭代评估
├── eval_phase6_iter2_*/            # Phase 6 Fix B 各实验（_sgm_fixB, _after_revert, _mid_v3 等）
├── eval_phase7_iter1/              # Phase 7 主结果（8/8 全面超越）
├── eval_phase7_sanity/             # Phase 7 KITTI-only 5ep sanity 验证
├── eval_phase8_b5/                 # Phase 8 B5 7ch 验证
├── phase8_pareto/                  # Phase 8 Pareto 主目录
│   ├── profile/{b0_h24,...}.json       # 各变体 params+GFLOPs
│   ├── eval/{b0_h24,...}.json          # 各变体 4 数据集指标
│   ├── eval_{variant}/summary.json     # 原始 eval 输出
│   ├── pareto_plot.png                  # 双联 Pareto 图
│   ├── summary_table.md / .csv          # 汇总表
├── demo_phase{3,4,5,6,7}_*/        # 各阶段 demo（phase5_viz 格式：01_left..06_*）
├── demo_phase7_iter1/              # Phase 7 最终 demo（竖向 6 面板，带 EPE/bad 标题）
│   └── {kitti,sceneflow,eth3d,middlebury}/<scene>/00_summary.png
├── demo_phase6_iter2_eth3d_tuned_sgm/  # Phase 8 Fix C demo（对比 cache-filled vs tuned raw SGM）
└── eval_*/                         # 其他评估
```

### Demo 面板约定（Phase 7+）

`demo_phase7.py` 生成（竖向 6 面板，标题含 EPE + bad）：
- `01_left.png` / `02_gt.png` / `03_mono.png` / `04_sgm.png` / `05_heuristic.png` / `06_effvit.png`
- `00_summary.png` = 6 面板竖向拼接

`demo_effvit_tuned_sgm.py` 生成（竖向 7 面板，Fix C 对比用）：
- 增加 `05_sgm_tuned_raw.png`（对照 `04_sgm_cache_filled.png`）

### Demo 标准面板命名

**demo.py 生成的**（Phase 1-4 + RCF 启发式）:
- `00_summary.png` — 总览拼接
- `01_left_image.png`, `02_sgm_disparity.png`, `03_confidence.png`
- `04_gt_disparity.png` — 真值（需 --gt-disparity 参数）
- `05_dense_aligned.png` — **DA2 直出对齐后**
- `06_merge_aligned.png`, `07_wcaps_aligned.png`
- `08_fused_sgm_merge.png` — **当前 fusion-strategy 的融合输出** (含 RCF)
- `09_heuristic_fused_wcaps.png` — 启发式融合参考 (永远是 edge_aware)
- `12_fusion_net_refined.png` — 学习型融合输出
- `14_heuristic_vs_net_diff.png` — 差异图

**phase5_viz.py 生成的**（Phase 5 专用，避开 demo.py arch 兼容问题）:
- `01_left.png`, `02_gt.png`, `03_mono_aligned.png`
- `04_calibrated_rcf_heuristic.png` — Stage A-D 启发式输出
- `05_rcf_refined.png` — 学习型精修后
- `06_region_confidence.png` — 区域校准置信度可视化

---

## memory-bank/ — 项目记忆库

| 文件 | 作用 |
|------|------|
| `README.md` | **顶层索引**。新人 onboarding 入口；最强 ckpt 路径；维护约定 |
| `design-document.md` | **算法设计文档**。Phase 1-5 RCF 四阶段 + Phase 6-8 EffViT 端到端融合 |
| `architecture.md` | **本文档**：目录/文件地图（代码 + artifacts + results） |
| `progress.md` | **进度日志**：Phase 1-8 完整迭代记录（含 Fix B/C 分支实验）|
| `paper-progress.md` | **CAL letter 投稿跟踪**（4-page IEEE submission） |
| `hardware-architecture.md` | **硬件侧文件地图** + API 速查 |
| `hardware-design-document.md` | **硬件侧正式 spec**（Specs A-D + 11 模块 + 事件驱动仿真器） |
| `hardware-progress.md` | **硬件侧 Step 1-14 进度日志** |

---

## 配置与入口文件

| 文件 | 作用 |
|------|------|
| `demo.py` | **主 demo 入口**。支持 `--fusion-strategy` (含 region_calibrated), `--fusion-backend {heuristic,net}`, `--gt-disparity`。**注意：当前不支持 rcf_refine arch，用 phase5_viz.py 替代** |
| `Makefile` | `make install/demo/eval-full/lint/format/check` |
| `pyproject.toml` | Python 项目配置 |
| `requirements.txt` | pip 依赖 |
| `README.md` | 公开项目说明 |
| `PROJECT_STATUS_CN.md` | 中文项目状态 |

---

## 环境变量

`scripts/common_config.py` 读取：
- `SGMVIT_KITTI_ROOT`
- `SGMVIT_ETH3D_ROOT`
- `SGMVIT_SCENEFLOW_ROOT`
- `SGMVIT_MIDDLEBURY_ROOT`
- `SGMVIT_DA2_WEIGHTS`

---

## 典型工作流

### 训练 Phase 8 EffViT 新变体（标准路径）

```bash
# 前置：确保 artifacts/fusion_cache_v3/ 已构建（见下文）
cd /home/pdongaa/workspace/SGM-ViT
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

# Stage 1: SF driving pretrain
python scripts/train_effvit.py --stage sf_pretrain \
  --cache-root artifacts/fusion_cache_v3 --dataset sceneflow \
  --train-split train --val-split val --epochs 20 --lr 3e-4 --batch-size 8 \
  --variant b1 --head-ch 24 --in-channels 7 \
  --hole-aug-prob 0.5 --hole-aug-max-frac 0.3 \
  --out-dir artifacts/fusion_new_b1_h24/sf_pretrain

# Stage 2: mixed finetune (KITTI + SF)
python scripts/train_effvit.py --stage mixed_finetune \
  --cache-root artifacts/fusion_cache_v3 \
  --mixed-datasets kitti,sceneflow --mixed-weights 1.0,1.0 \
  --mixed-val-dataset kitti --val-split val \
  --epochs 15 --lr 5e-5 --batch-size 4 \
  --variant b1 --head-ch 24 --in-channels 7 \
  --init-ckpt artifacts/fusion_new_b1_h24/sf_pretrain/best.pt \
  --out-dir artifacts/fusion_new_b1_h24/mixed_finetune

# Eval + profile
python scripts/eval_effvit.py --ckpt artifacts/fusion_new_b1_h24/mixed_finetune/best.pt --cache-root artifacts/fusion_cache_v3 --out-dir results/eval_new
python scripts/profile_effvit.py --ckpt artifacts/fusion_new_b1_h24/mixed_finetune/best.pt
```

### 重建 Phase 7+ fusion_cache_v3

```bash
# Step 1: tuned SGM 重跑（4 数据集 × 可并行）
python scripts/rerun_sgm_tuned.py --dataset eth3d --skip-existing
python scripts/rerun_sgm_tuned.py --dataset middlebury --skip-existing
python scripts/rerun_sgm_tuned.py --dataset kitti --skip-existing
# SF 全量 4400 pair 建议分 4 片并行：--start-index 0/1100/2200/3300 --end-index 1100/2200/3300/4400

# Step 2: v3 cache（DA2 + align + heuristic）
python scripts/build_fusion_cache_v3.py --dataset {eth3d,middlebury,kitti,sceneflow} --skip-existing
```

### 跑 Phase 8 Pareto 6 变体（无人值守）

```bash
bash scripts/run_phase8_pareto.sh  # tmux 推荐
# 产物：artifacts/fusion_phase8_pareto/{variant}/ + results/phase8_pareto/
```

### 生成最终 demo（Phase 7）

```bash
python scripts/demo_phase7.py --ckpt artifacts/fusion_phase7_iter1/mixed_finetune/best.pt --per-dataset 3
# 输出 results/demo_phase7_iter1/<dataset>/<scene>/00_summary.png (6 面板)
```

### Phase 1-5 历史路径（保留兼容）

```bash
# RCF 启发式（无网络）
python demo.py --fusion-strategy region_calibrated --left ... --right ... --gt-disparity ...

# Phase 5 RCF refined 训练
python /tmp/run_phase5_rcf.py

# Phase 1-4 FusionNet 训练
python scripts/train_fusion_net.py --stage mixed --arch mask_residual_lite --out-root artifacts/xxx
```

---

## Phase 5 关键代码追踪（历史）

```
run_phase5_rcf.py
 ├─ RCFCacheDataset.__getitem__
 │   └─ compute_rcf_signals
 │       ├─ compute_detail_guidance         (core/fusion.py)
 │       ├─ segment_mono_regions            (core/fusion.py)
 │       ├─ calibrate_regions               (core/fusion.py, Huber+sanity)
 │       └─ blend_region_boundaries         (core/fusion.py)
 │
 ├─ RegionCalibratedRefineNet (13ch input) (core/fusion_net.py)
 │   └─ 4 heads: detail_mask, detail_gain, residual_mask, tiny_residual
 │
 ├─ compute_losses
 │   ├─ predict_rcf_refine                  (core/fusion_net.py)
 │   ├─ disp_loss (Smooth L1)
 │   ├─ grad_loss (x/y gradients)
 │   ├─ calibration_preservation_loss       (新)
 │   └─ detail_gating_loss                  (新)
 │
 └─ Two-stage training: SceneFlow 15 epochs → KITTI 10 epochs
```

---

## Phase 6-8 关键代码追踪（主线）

```
scripts/train_effvit.py
 ├─ EffViTCacheDataset.__getitem__       (v3 cache reader)
 │   ├─ load rgb / mono / sgm / conf / sgm_valid / gt / valid
 │   ├─ disp_jitter 0.9-1.1
 │   ├─ random crop + hflip
 │   └─ hole-aug (prob=0.5, 3-8 rectangles ≤ 30% area zeroed)
 │
 ├─ EffViTDepthNet                        (core/effvit_depth.py)
 │   ├─ _make_backbone(variant ∈ {b0,b1,b2}) in_channels=7
 │   │   └─ third_party/efficientvit backbone returns 5-stage features
 │   ├─ 4 × _UpBlock FPN decoder (head_ch ∈ {24,48})
 │   │   └─ upsample + 1x1 lateral + 3x3 fuse (BN + Hardswish)
 │   ├─ head_conv + residual_conv (zero-init → starts as identity)
 │   └─ residual_clamp ±32
 │
 ├─ compute_effvit_losses
 │   ├─ smooth_l1 over valid pixels
 │   └─ soft_D1 = sigmoid(k*(|err|-τ)), per-sample τ from TAU_BY_DATASET
 │       (kitti=3, sceneflow=1, eth3d=1, middlebury=2)
 │
 └─ Two-stage training:
    Stage 1: sf_pretrain (sceneflow v3 cache only), 20ep, lr 3e-4 cosine, batch 8
    Stage 2: mixed_finetune (ConcatDataset kitti+sceneflow + WeightedRandomSampler), 15ep, lr 5e-5, batch 4


scripts/build_fusion_cache_v3.py
 ├─ 4 enumerators: build_samples_{kitti,sceneflow_driving_full,eth3d,middlebury}
 │   └─ SF driving 枚举全 8 splits = 4400 pair (vs v1 only 1 split 300)
 │
 ├─ per-sample pipeline:
 │   ├─ load tuned SGM npz (/tmp/sgm_tuned_all/<dataset>/<name>.npz)
 │   ├─ DA2 inference (run_decoder_weight_caps_merged_da2, core/pipeline.py)
 │   ├─ align_depth_to_sgm (Huber, core/pipeline.py:531)
 │   ├─ fuse_edge_aware_residual (new heuristic, core/fusion.py)
 │   ├─ sgm_disp = disp_raw with holes=0
 │   ├─ sgm_valid = ~hole_mask (NEW field)
 │   └─ save npz → artifacts/fusion_cache_v3/<dataset>/<split>/<name>.npz


scripts/rerun_sgm_tuned.py
 ├─ DATASET_PARAMS: per-dataset (range, P2, P1, Win)
 │   eth3d=80/3.0/0.3/5, sceneflow=256/3.0/0.3/5, kitti=192/3.0/0.3/5, middlebury=192/3.0/0.3/5
 ├─ run_sgm_with_confidence(..., return_debug=True)  (core/sgm_wrapper.py)
 │   └─ saves disp_raw + hole_mask + raw PKRN conf (no Gaussian smooth)
 └─ supports --start-index/--end-index for 4-way parallel (56 CPU cores available)
```


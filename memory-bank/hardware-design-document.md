# SGM-ViT 硬件加速器设计文档

**项目**: EdgeStereoDAv2 算法-硬件协同设计 (ICCAD 2025)
**架构**: 统一 Systolic Array (SA) + 5 个 Special Compute Unit (SCU)
**目标工艺**: 28nm CMOS @ 500 MHz (主目标) / 7nm @ 1 GHz (对比)
**最后更新**: 2026-04-14 (Step 14 完成, 全部 Spec A-D 通过)

---

## 1. 项目背景

### 1.1 算法侧需求

SGM-ViT 流水线产生以下硬件需求:

```
立体图像对 -> SGM (外部引擎) -> 视差 + PKRN 置信度
                                        |
                                        v
左图 -> Patch Embed -> Encoder (12 层 ViT-S, 含 token merge) -> Decoder (DPT, 含 W-CAPS 双精度)
                                        |
                                        v
                                  绝对视差对齐 (ADCU)
                                        |
                                        v
                              Edge-aware Residual Fusion -> 输出视差
```

四条算法主线需要硬件支持:
1. **Token Merge** — 用 SGM 置信度选 representative token, ViT attention 只在 representative 上做
2. **Decoder W-CAPS** — 解码器空间自适应精度 (HP/LP 双路径 + 空间混合)
3. **Edge-aware Residual Fusion** — 双边滤波 / 梯度 / alpha 混合的像素级融合
4. **绝对视差对齐** — 稀疏 Harris 关键点 + 最小二乘求 scale-shift

### 1.2 设计原则

| 原则 | 落实 |
|------|------|
| **单一 SA 时分复用** | 32×32 systolic array 同时承担 encoder matmul 和 decoder conv, WS/OS 模式可切换 |
| **SCU 处理算法特定操作** | 5 个 SCU 各自负责一类不能让 SA 高效完成的工作 |
| **置信度作为外部输入** | PKRN 由外部 SGM 引擎提供, 不与 ADCU 耦合 (Spec A) |
| **完整 decoder 操作建模** | 11 个解码器阶段全部进入 DAG, 不抽象为粗粒度 (Spec B) |
| **L2 显式驻留契约** | 所有 buffer 的大小 / 生命周期 / 流式规则文档化 (Spec C) |
| **真正的事件驱动仿真** | 每个 module 在 pipeline 阶段边界发事件, 跳周期推进 (Spec D) |
| **Python 行为级先行, RTL/FPGA 后续** | 接口和模块边界按 RTL 友好的方式划分 |
| **参数化, 不硬编码** | 所有维度从 config 派生, hardware/workload 边界明确 |

---

## 2. 四个核心规格 (Spec A-D)

### Spec A: Confidence Provenance & DAG

**关键决策**: PKRN 置信度图由**外部 SGM 引擎**计算, 通过 DMA 在 frame 起始加载到 L2. **ADCU 不产生置信度** — 它只做稀疏 Harris 匹配 + 最小二乘求 scale-shift. 这是与早期论文描述的重要修正.

**算法依据**: `core/sgm_wrapper.py:_pkrn()` 显示 PKRN 来自 SGM cost volume 的单趟前向遍历, 与 ADCU 的稀疏关键点匹配是完全不同的运算.

**DAG 影响**: CRM 仅依赖 DMA 完成 (置信度可读), 不依赖 ADCU. CRM 可与 patch_embed 完全并行.

```
DMA: load image + SGM disp + confidence + DA2 weights
  |
  +--> SA: patch_embed
  |     |
  +--> CRM: pool conf -> sort -> select reps -> assign members
              |
              v
        GSU: gather block[0] reps
              |
              v
        SA: qkv[0] -> attn[0] -> out_proj[0]
              |
              v
        GSU: scatter_merge[0]
              |
              v
        SA: mlp[0]
              |
              ... (12 blocks total) ...
              v
        SA: decoder (Spec B 全部 11 阶段)
              |
              v
        ADCU: scale_shift + pixel_apply
              |
              v
        FU: fusion pipeline (输出)
```

### Spec B: Full Decoder Operation Taxonomy

DPT decoder 不是简单的卷积序列. 完整操作枚举 (来自 `core/decoder_adaptive_precision.py`):

**Stage 1 — Feature Extraction** (4 个 tap, 来自 encoder 层 {2,5,8,11}):

| Op | 模块 | HP/LP 双路径 | 分辨率 |
|----|------|--------------|--------|
| `readout_projects[i]` (linear, 若 use_clstoken) | SA (WS) | 是, 当 `should_apply(proj_{i+1})` | 37×37 |
| `projects[i]` (1×1 conv) | SA (WS) | 是 | 37×37 |
| `resize_layers[i]` (bilinear / conv transpose) | FU / SA | 是 | 37→{37,74,148,296} |
| `build_stage_high_precision_mask` | DPC | — | 输出分辨率 |
| `blend_spatial_outputs` | FU (ElemWise) | — | 输出分辨率 |

**Stage 2 — Layer Normalization** (4 个 1×1 conv):

| Op | 分辨率 |
|----|--------|
| `scratch.layer1_rn` | 37×37 |
| `scratch.layer2_rn` | 74×74 |
| `scratch.layer3_rn` | 148×148 |
| `scratch.layer4_rn` | 296×296 |

**Stage 3 — RefineNet Cascade** (4 个顺序阶段, 每个含 2 RCU × 2 3×3 conv):

| Op | 输入 | 分辨率 |
|----|------|--------|
| `refinenet4(layer_4_rn)` | layer_4_rn | 148×148 |
| `refinenet3(path_4, layer_3_rn)` | path_4 + layer_3_rn | 74×74 |
| `refinenet2(path_3, layer_2_rn)` | path_3 + layer_2_rn | 37×37 |
| `refinenet1(path_2, layer_1_rn)` | path_2 + layer_1_rn | 37×37 |

**Stage 4 — Output Head**:

| Op | 模块 | 分辨率 |
|----|------|--------|
| `output_conv1` (3×3 conv) | SA (OS) | 296×296 |
| bilinear upsample to 518×518 | FU (bilinear) | 518×518 |
| `output_conv2` (1×1 conv) | SA (WS) | 518×518 |

**stage_policy 影响**:

| 策略 | 受影响 tag | 双路径阶段数 |
|------|------------|--------------|
| `coarse_only` | proj_3, proj_4, rn_3, rn_4, path_3, path_4 | 6 |
| `all` | 全部 11 个 + output | 12 |
| `fine_only` | proj_1, proj_2, rn_1, rn_2, path_1, path_2, output | 7 |

### Spec C: Memory Residency & Streaming Contract

**Buffer Catalog** (518×518, ViT-S 配置):

| Buffer | 大小 | 驻留位置 | 生命周期 | 备注 |
|--------|------|----------|----------|------|
| 输入图像 | 805 KB | DRAM → L2 tiled | DMA → patch_embed 完成 | 用完释放 |
| SGM 视差 | 1.07 MB | DRAM → L2 streamed | DMA → fusion 完成 | FU 流式读 |
| **置信度图** | 1.07 MB | DRAM → L2 streamed | DMA → fusion 完成 | CRM 读 37×37 池化子集; FU 读全图流式 |
| DA2 weights (encoder) | ~12 MB INT8 | DRAM → L2 双缓冲 | 每 block, 预取 | 2 × 32KB 双缓冲 |
| DA2 weights (decoder) | ~13 MB INT8 | DRAM → L2 双缓冲 | 每 stage, 预取 | 同上 |
| **Token sequence** | Flash 分块: Br=64 → 24KB Q, 128×384=48KB KV | L2 (tiled) | 每 attention tile | **从不全部物化** (526KB > 512KB L2) |
| CRM 索引表 | member_to_rep_local: 1369×2=2.7KB | CRM L1 (4KB) | CRM 完成 → 最后 scatter | 12 个 block 全程驻留 |
| GSU compact buffer | K×384 ≈ 263KB (K=685) | L2 | 每 block | **关键约束**: 必须按 64 token chunk 流式 |
| Encoder 中间 features | 4 × 526KB = 2.1 MB | DRAM spill | 写于 {2,5,8,11} 层, decoder 读 | 单个 tap 已超 L2, 必须落 DRAM |
| Decoder feature maps | 最大 296×296×64 = 5.5 MB | L2 tiled / DRAM | 每 refinenet stage | 32×32 tile × 64 ch = 64KB/tile |
| **Decoder HP/LP 双路径** | 2× per affected stage | L2 双缓冲 | HP→LP→blend | HP 在 buf A, LP 在 buf B, blend 写回 A |
| DPC sensitivity map | 1.07 MB | DRAM → L2 streamed | 按 stage 读 | 流式 |
| DPC HP mask | H×W bits (≤296²/8 = 11KB) | DPC L1 (2KB) 或 L2 | 每 stage | 粗阶段可放 L1 |
| FU 中间图 | 多张 ~1MB (sgm_smooth, da2_low, detail_score 等) | DRAM 后备, L2 tiled | fusion pipeline 内 | 32 行 strip = 66KB/strip |
| ADCU LUT | 4096×2 = 8KB | ADCU L1 | 跨 frame 持久 | init 时加载 |

**Streaming Rules**:

1. **GSU compact buffer**: K=685 时 263KB, 占 L2 一半. 规则: GSU 按 64 token (24KB) 一个 chunk gather, 喂给 SA, 完成 attention 后 scatter 回去, 再 gather 下一个 chunk. 每 block 内 GSU→SA→GSU 串行多次.
2. **Encoder taps**: 单 tap 526KB > L2. 规则: 在每个 tap 层 SA 把 features DMA 写到 DRAM, decoder 阶段需要时再从 DRAM 加载.
3. **Decoder dual-path**: 每阶段 HP 写 L2 buf A, LP 写 L2 buf B (复用输入空间), DPC 触发 FU blend, 结果覆盖 A, 释放 B. 并发上限: 2 × tile_size.
4. **FU fusion pipeline**: 中间图 ~1MB 各, 不全放 L2. 按 32 行 strip (66KB/strip per map) 处理, L2 同时存 3 strip = 200KB.
5. **Double-buffering 契约**: 仅 SA weight 预取使用真正双缓冲 (2×32KB). SCU L1 单缓冲. L2 显式管理.

**L2 worst-case 并发预算验证**:

| 占用 | 大小 |
|------|------|
| SA weight 双缓冲 | 64 KB |
| SA activation tile (Q 或 KV) | 48 KB |
| GSU compact chunk (64 token) | 24 KB |
| Decoder 双路径 buffer (2 × tile) | 128 KB |
| FU strip buffer (3 strips) | 200 KB |
| DMA staging | 32 KB |
| **总计** | **496 KB / 512 KB (96.9% OK)** |

### Spec D: Module-Level Event Contracts

每个模块在 **pipeline 阶段边界**发事件, 不在每个周期发. 主循环跳周期推进, idle/stall gap 在每次跳跃时按 O(num_modules) 批量累计.

**SA 事件契约** (每个 op):

| 事件 | 触发时刻 | 数据 |
|------|----------|------|
| `sa:weight_load_start` | 接受新 op, 开始加载权重 tile | op_id, bytes |
| `sa:weight_load_done` | 权重 tile 已进 PE 寄存器 | — |
| `sa:tile_compute_start` | 第一个激活进入 skew buffer | tile_idx |
| `sa:tile_compute_done` | 最后一个结果出 de-skew buffer | tile_idx, macs |
| `sa:tile_writeback_done` | tile 结果写完 L1/L2 | tile_idx |
| `sa:op_complete` | op 全部 tile 完成 | op_id |

每 op 事件总数 ≈ `2 + 3 × num_tiles + 1`. 22 tile 即 69 事件. 60 op/frame ≈ 4000 事件, 性能预算内.

**SCU 事件契约**:

| 模块 | 事件 (per op) | 模式 |
|------|---------------|------|
| **CRM** | fetch_conf, sort_done, assign_done, op_complete | 多阶段 pipeline |
| **GSU** | per chunk: chunk_gather_start/done, chunk_scatter_done; op_complete | 64-token 流式 |
| **DPC** | per stage: mask_gen_start/done, blend_trigger; op_complete | 解码阶段控制器 |
| **ADCU** | sparse_match_start/done, pixel_apply_start, op_complete | 顺序 pipeline |
| **FU** | per strip: strip_fetch, strip_compute_done, strip_writeback_done; op_complete | strip 流式 |

**跨模块同步事件**:

| 事件 | 源 | 消费者 | 含义 |
|------|----|--------|------|
| `mem:grant` | L2Controller | 请求模块 | 内存访问授权 |
| `mem:deny` | L2Controller | 请求模块 | 带宽耗尽, 下一周期重试 |
| `sched:dispatch` | Scheduler | 目标模块 | 新 op 派发 |
| `sched:op_complete` | 任意模块 | Scheduler | op 完成, 检查 DAG 后继 |

**Idle/Stall 累计规则**: 两个连续事件 (周期 C1, C2) 之间, 所有模块按当前状态把 `C2 - C1` 累计到 `idle_cycles` / `compute_cycles` / `stall_mem_cycles` / `stall_dep_cycles` 之一. O(num_modules) 完成, 不逐周期.

---

## 3. 顶层架构

```
                          DRAM (LPDDR4x, 16 GB/s)
                                  |
                          DMA (4 channels)
                                  |
                          L2 SRAM (512 KB, 32 banks)
                                  |
            +---------+---------+--+--+---------+---------+
            |         |         |     |         |         |
        +---v---+ +---v---+ +---v-+ +-v---+ +---v---+ +---v---+
        |  SA   | | CRM   | | GSU | | DPC | | ADCU  | |  FU   |
        | 32x32 | |       | |     | |     | |       | |       |
        | + LN  | +-------+ +-----+ +-----+ +-------+ +-------+
        | + SM  |
        | + GeLU|
        +-------+

Sidecars (LN/Softmax/GELU): 共享 SA 控制总线, 独立数据通路, 与 SA 计算互斥
```

### 3.1 模块清单与职责

| 模块 | 类型 | 主要功能 | 优先级 |
|------|------|----------|--------|
| **SA** | 主算力 | 所有大矩阵乘法 (encoder QKV/proj/MLP, decoder conv) | 主体 |
| **CRM** | SCU-1 | 置信度池化 + 排序选 representative + 距离分配 + LUT 写出 | P1 |
| **GSU** | SCU-2 | 索引选址 + 数据搬运 (gather/scatter), 64 token 流式 | P1 |
| **DPC** | SCU-3 | 解码器双精度控制器: 敏感度图 → HP mask → SA 双路径 → FU blend | P2 |
| **ADCU** | SCU-4 | 稀疏 Harris 关键点 NCC 匹配 + 最小二乘 + LUT 倒数 | P3 |
| **FU** | SCU-5 | 像素级运算: 双边滤波 / 高斯 / Sobel / alpha 混合 / 双线性上采样 | P4 |

---

## 4. 模块详细设计

### 4.1 Foundation: Interfaces & Base Module

**`hardware/interfaces.py`**

| 类 | 用途 |
|----|------|
| `Direction(Enum)` | IN / OUT 方向标识 |
| `Signal` | 单根 wire (name, width_bits, direction) |
| `Port` | signal bundle + ready/valid 握手 (映射 AXI-Stream) |
| `MemoryPort(Port)` | addr_width=20, data_width_bits=256 (32 B/cycle), max_burst_len=16 |
| `StreamPort(Port)` | data_width_bits + has_last (tlast for end-of-tile) |
| `ConfigPort(Port)` | 低带宽 config 写入, 同步, 无握手 |

**`hardware/base_module.py`**

| 类 | 用途 |
|----|------|
| `ModuleState(Enum)` | IDLE, FETCH, COMPUTE, WRITEBACK, STALL_MEM, STALL_DEP, DRAIN, RECONFIGURE |
| `ModuleStats` | 各类周期累计 + MAC/事件/内存读写计数; `accumulate_gap()` 实现 Spec D 批量累计 |
| `Event` | (cycle, priority, module_id, action, data); `@dataclass(order=True)` |
| `PipelineStage` | 一个 pipeline stage 的 (name, remaining_cycles, is_active, metadata) |
| `HardwareModule` (ABC) | 抽象基类, 所有模块继承. 必须实现: `_declare_ports()`, `accept_op()`, `handle_event()`, `describe()`, `estimate_area_mm2()`, `estimate_power_mw()` |

### 4.2 Unified Systolic Array

**`hardware/pe_array/unified_sa.py`**

#### Configuration

```python
@dataclass
class SAConfig:
    rows: int = 32
    cols: int = 32
    bitwidth: int = 8           # INT8 operand
    acc_width: int = 32         # INT32 accumulator
    reconfigure_cycles: int = 4 # 模式切换代价
    default_tile_m: int = 64
    default_tile_n: int = 128
    default_tile_k: int = 64
    flash_tile_br: int = 64     # Flash attention Q tile
    flash_tile_bc: int = 128    # Flash attention KV tile
    softmax_segments: int = 16
    gelu_segments: int = 32
```

#### Modes

| 模式 | 用途 | 数据流 |
|------|------|--------|
| `WS` (Weight-Stationary) | 矩阵乘法 (QKV/proj/MLP/1x1 conv) | 权重驻留 PE, 激活流过 |
| `OS` (Output-Stationary) | 3×3 conv (decoder) | 输出驻留 PE, 权重 + 输入流过 |
| `IS` (Input-Stationary) | 预留 (depthwise conv) | — |

#### 内部 Pipeline (每个 tile)

| 阶段 | 周期 | 说明 |
|------|------|------|
| Weight Load | `ceil(tile_k * tile_n * bytes / port_bw)` | 流入 PE 寄存器 |
| Input Skew | `rows - 1 = 31` | 三角延迟, systolic 时序 |
| Compute | `tile_k` | systolic MAC 执行 |
| Output De-skew | `cols - 1 = 31` | 收集对角线波前 |
| Drain / Writeback | `ceil(tile_m * tile_n * acc_bytes / port_bw)` | 写回 L1/L2 |

**双缓冲 steady state** (Spec D 计算):
- 第一个 tile: full pipeline fill (上述 5 阶段串行)
- 后续 tile: `max(weight_load + compute, writeback)` (HP 写回与下一 tile 加载+计算并行)

#### Sidecar Units

| Sidecar | Pipeline 级数 | 吞吐 | 28nm 面积 | 28nm 功率 |
|---------|---------------|------|-----------|-----------|
| `PiecewiseLinearSoftmax` | 4 | 32 elem/cycle | 0.012 mm² | 2.0 mW |
| `GELUApprox` | 2 | 32 elem/cycle | 0.008 mm² | 1.2 mW |
| `LayerNormSidecar` | 3 | 32 elem/cycle | 0.010 mm² | 1.5 mW |

**关键设计**: Sidecar 是**独立数据通路**, 与 SA MAC 阵列**互斥** — 共享 SA 控制总线, SA 调度其执行. 在 attention 中, softmax 占用 SA 周期 (SA 暂停 MAC); 在 MLP 中, GELU 占用 SA 周期.

#### Cycle Estimation API

```python
sa.estimate_matmul_cycles(M, K, N, tile_m=None, tile_n=None, tile_k=None, utilization=0.85)
sa.estimate_conv_cycles(C_in, C_out, H, W, kernel_size=3, tile_h=8, tile_w=8, tile_co=16, utilization=0.75)
sa.estimate_attention_cycles(seq_len, embed_dim, num_heads, utilization=0.85)
sa.estimate_mlp_cycles(seq_len, embed_dim, mlp_ratio=4.0, utilization=0.90)
sa.estimate_sidecar_cycles(name, num_elements)
```

**ViT-S @ 518×518 验证结果**:

| Op | Cycles | 等效时间 @ 500MHz |
|----|--------|------------------|
| QKV 投影 (1370×384×1152) | 1,431,640 | 2.86 ms |
| Full attention (1370 tokens, 6 heads) | 6,341,127 | 12.68 ms |
| MLP block | 3,687,533 | 7.38 ms |
| 单 encoder block | 10,028,660 | 20.06 ms |
| 12 encoder blocks | 120,343,920 | 240.7 ms |
| Conv3x3 (64→64 @ 37×37) | 132,266 | 0.26 ms |
| Merge attention (685 tokens) | 2,167,484 | 4.33 ms (2.93x speedup) |

#### 面积 / 功耗 (28nm)

| 组件 | 面积 |
|------|------|
| MAC 阵列 (1024 PE × 400 µm²) | 0.410 mm² |
| Skew/De-skew buffer | 0.010 mm² |
| Sidecars (LN + Softmax + GELU) | 0.030 mm² |
| Control | 0.005 mm² |
| **SA 总计** | **0.455 mm²** |

功率 @ 500 MHz, 80% util ≈ 19.9 mW (MACs 主导)

### 4.3 SCU-1: CRM (Confidence Routing Module)

**`hardware/scu/crm.py`**

#### 职责

读取**外部 SGM** (DRAM → L2) 提供的全分辨率 PKRN 置信度图, 池化到 token grid, 生成 merge plan (representative 选择 + member-to-rep 分配).

**置信度来源 (Spec A)**: 外部 SGM 引擎. CRM 不依赖 ADCU.

#### Modes

| Mode | 算法 |
|------|------|
| `PRUNE` | 阈值 → 二值 keep/prune mask (兼容老硬件路径) |
| `MERGE` | 排序 → 选 reps → 距离分配 → LUT 写出 |
| `CAPS_MERGE` | MERGE + per-group 敏感度评分 + HP/LP mask |

#### Configuration

```python
@dataclass
class CRMConfig:
    max_tokens: int = 1369           # 37*37
    max_reps: int = 1024
    num_distance_units: int = 32     # 并行欧氏距离单元
    num_sort_comparators: int = 16   # bitonic 网络比较器
    l1_size_bytes: int = 4096        # 4 KB 内部 SRAM
```

#### Pipeline

| 阶段 | 算子 | 周期 (kr=0.5) | 硬件 |
|------|------|---------------|------|
| Pool | 自适应平均池化, 流式 | 16,771 | 累加器 |
| Sort | 1369 个 FP16 bitonic 排序 | 5,648 | 16 路并行比较器 |
| Select | 取 bottom-K 作为 representative | 1 | 索引寄存器 |
| Assign | 1369 token 各自计算到 K reps 距离取 argmin | 30,118 | 32 并行欧氏距离单元 (2 sub + 2 sqr + adder + cmp) |
| Score (CAPS only) | 每 group 流式 mean/var | K | 累加器 |
| HP-Select (CAPS only) | top-K on group scores | log²K | 复用 sort 网络 |
| LUT Writeback | member_to_rep_local 写 L2 | 87 | DMA |

**总周期** (kr=0.5, MERGE): 52,624 ≈ 105 µs @ 500MHz. 一帧一次, 与 patch_embed 并行.

**PRUNE 模式**: 1369 并行比较器 (1 周期) + prefix-sum (12 周期) = 13 周期 + 池化 + LUT.

#### 面积 / 功耗 (28nm)

| 组件 | 面积 |
|------|------|
| 32 距离单元 | 0.026 mm² |
| Sort 网络 | 0.003 mm² |
| L1 SRAM (4KB) | 0.004 mm² |
| Control | 0.002 mm² |
| **CRM 总计** | **0.035 mm²** |

### 4.4 SCU-2: GSU (Gather-Scatter Unit)

**`hardware/scu/gsu.py`**

#### 职责

L2 与 SA L1 input buffer 之间的**地址生成 + 数据导向**单元. 不做算术, 把 token 索引翻译成 SRAM 地址, 发起读写命令.

**Streaming 规则 (Spec C)**: 按 `chunk_size = 64` token (24 KB) 分块, 限制 L2 占用.

#### Modes

| Mode | 算法 |
|------|------|
| `GATHER` | 索引选 M 个 representative 从全序列到 compact buffer |
| `SCATTER_PRUNE` | 把 M 个 kept 输出写回原位置 |
| `SCATTER_MERGE` | 通过 LUT 把每个 representative 输出广播给所有 group 成员 (一对多读) |

#### Cycle Estimates (embed_dim=384, L2 BW=64 B/cycle)

| Op | 公式 | 示例 (kr=0.5, K=685) |
|----|------|---------------------|
| GATHER | `ceil(K * 384 / 64) + index_load` | 4,246 |
| SCATTER_MERGE | `ceil(1369 * 384 / 64) + index_load` | 8,491 |
| SCATTER_PRUNE | `ceil(M * 384 / 64) + index_load` | 4,132 |

支持 streaming: 索引未全部加载即开始发地址.

#### 面积 (28nm)

| 组件 | 面积 |
|------|------|
| 地址生成 (1 mul + 1 add) | 0.002 mm² |
| L1 SRAM (8KB, 索引 + burst buffer) | 0.008 mm² |
| Control | 0.001 mm² |
| **GSU 总计** | **0.011 mm²** |

### 4.5 SCU-3: DPC (Dual Precision Controller)

**`hardware/scu/dpc.py`**

#### 职责

控制器 (不是算力). 为 decoder 双路径执行做**编排**: 生成 HP mask, 编程 SA 跑 HP/LP conv, 触发 FU blend.

**覆盖 Spec B 全部 11 个解码器 tag**: proj_1-4, rn_1-4, path_1-4, output.

#### Pipeline (每个受影响阶段)

1. **mask_gen**: 将敏感度图 resize 到 stage 分辨率, histogram 阈值法选 top-K → 二值 HP mask
2. SA 跑 HP weights 卷积 → L2 buf A
3. SA 跑 LP weights 卷积 → L2 buf B
4. **blend_trigger**: 通知 FU 做 `mask * A + (1-mask) * B` → 覆盖 A, 释放 B

#### Mask Generation 周期分解 (示例 296×296)

| 阶段 | 周期 | 说明 |
|------|------|------|
| Resize | 2,738 | 双线性 sensitivity → stage 分辨率, 32 px/cycle |
| Histogram Build | 2,740 | 1 趟扫描 (256 bin), 32 px/cycle |
| Histogram Scan | 256 | 找阈值, 256 bin 串扫 |
| Mask Write | 86 | 二值 mask 写 L2 (1 bit/px) |
| **总计** | **5,820** | — |

#### 总开销 (coarse_only, 6 stages)

| 类目 | 周期 |
|------|------|
| Mask gen 总和 | 17,908 |
| Blend 总和 (FU) | 985,680 |
| **DPC overhead** | **1,003,588** |

#### 面积 (28nm)

| 组件 | 面积 |
|------|------|
| Histogram SRAM (1KB) | 0.001 mm² |
| Control | 0.002 mm² |
| **DPC 总计** | **0.003 mm²** |

### 4.6 SCU-4: ADCU (Absolute Disparity Calibration Unit)

**`hardware/scu/adcu.py`**

#### 职责

把 DA2 相对深度转为绝对视差. 三个子单元:
1. **稀疏 Harris 关键点 NCC 匹配** (32 keypoints, 11×11 patch, max_disp=192)
2. **Scale-Shift 最小二乘求解** (2×2 系统, 闭式逆)
3. **像素级 z = s·d + t → 1/z (LUT) → disp = B·f/z**

**重要**: ADCU **不产生置信度** (Spec A). 没有 Confidence Pooler 子单元.

#### 周期分解 (518×518)

| 子单元 | 周期 |
|--------|------|
| Sparse NCC matcher (32 KP × 192 disp × 11² ops / 32 par) | 116,160 |
| LS solver (累加 A^T A, A^T b + 2×2 逆) | 323 |
| Pixel applicator (s·d+t + LUT + multiply, 32 px/cycle) | 8,386 |
| **总计** | **124,869** ≈ 250 µs |

稀疏匹配占主导, pixel apply 仅 ~7% 周期.

#### 面积 (28nm)

| 组件 | 面积 |
|------|------|
| 32 NCC correlator | 0.020 mm² |
| LS solver | 0.005 mm² |
| Reciprocal LUT (4096 × 16 bit = 8KB) | 0.008 mm² |
| Pixel applicator (32 lane) | 0.010 mm² |
| Control | 0.002 mm² |
| **ADCU 总计** | **0.045 mm²** |

### 4.7 SCU-5: FU (Fusion Unit)

**`hardware/scu/fu.py`**

#### 职责

所有元素级 / 小核空间 / 双线性上采样, 这些不值得走 SA. 三个子单元:

| 子单元 | 描述 | 吞吐 |
|--------|------|------|
| **BilinearUpsampleEngine** | shift-add only (从 `upsample_pe.py` 移入) | 32 pixel/cycle |
| **ElementWiseEngine** | 32 lane, 支持 mul/add/cmp/clamp/alpha_blend | 32 pixel/cycle |
| **SpatialFilterEngine** | 3×3 kernel, Gaussian/Sobel/bilateral (含 range LUT) | 4 output pixel/cycle |

**Streaming 规则**: 32 行 strip 处理, L2 同时存 3 strip = 200 KB.

#### Edge-aware Residual Fusion 周期分解 (518×518)

| Step | 引擎 | 周期 |
|------|------|------|
| Bilateral filter SGM | SpatialFilter | 67,081 |
| Gaussian blur DA2 | SpatialFilter | 67,081 |
| Sobel gradient (x2) | SpatialFilter | 150,934 |
| Gradient magnitude + detail score | ElemWise | 41,935 |
| Alpha blend + residual | ElemWise | 16,774 |
| Strip overhead | — | 87,213 |
| **总计** | — | **431,018** ≈ 0.86 ms |

可与 SA decoder 计算并行 (FU 与 SA 独立).

#### DPC blend 调用

每个解码阶段的 `mask * hp + (1-mask) * lp` 委托给 FU ElementWiseEngine, 2 周期/像素 batch. 296×296 阶段: 2,738 周期.

#### 面积 (28nm)

| 组件 | 面积 |
|------|------|
| Bilinear (shift-add) | 0.005 mm² |
| ElementWise (32 lane) | 0.016 mm² |
| SpatialFilter (4 输出 × 9 MAC) | 0.014 mm² |
| Bilateral LUT (256 entry) | 0.001 mm² |
| L1 SRAM (16KB) | 0.016 mm² |
| Control | 0.002 mm² |
| **FU 总计** | **0.054 mm²** |

### 4.8 L2 Arbiter

**`hardware/architecture/interconnect.py`**

#### Configuration

```python
@dataclass
class ArbiterConfig:
    num_banks: int = 32
    bank_size_bytes: int = 16384       # 16 KB / bank → 512 KB total
    bank_width_bytes: int = 8          # 8 B/bank/cycle
    read_latency_cycles: int = 3
    write_latency_cycles: int = 3
```

峰值带宽: 32 banks × 8 B = 256 B/cycle.

#### Priority Arbitration

| 优先级 | 模块 |
|--------|------|
| 0 (最高) | systolic_array |
| 1 | crm |
| 2 | gsu |
| 3 | dpc |
| 4 | adcu |
| 5 | fu |
| 6 (最低) | dma |

同优先级内 round-robin. 32 bank 可同时服务无冲突的 32 个请求.

#### Buffer Allocation Tracking

`L2Arbiter.allocate_buffer(name, size)` / `free_buffer(name)` 跟踪逻辑 buffer 占用, 拒绝超容量分配. `validate_budget()` 返回当前占用 / 容量 / 利用率.

### 4.9 Memory Hierarchy (Revised)

**`hardware/architecture/memory_hierarchy.py`**

| Level | 容量 | BW (B/cycle) | Latency | E_read (pJ) | Banks |
|-------|------|--------------|---------|-------------|-------|
| L0 RegFile | 512 B/PE | 64 | 0 | 0.05 | 1 |
| L1 SA | 64 KB | 32 | 1 | 2.0 | 16 |
| L1 CRM | 4 KB | 16 | 1 | 1.5 | 4 |
| L1 GSU | 8 KB | 16 | 1 | 1.5 | 4 |
| L1 DPC | 2 KB | 8 | 1 | 1.2 | 2 |
| L1 ADCU | 8 KB | 16 | 1 | 1.5 | 4 |
| L1 FU | 16 KB | 32 | 1 | 2.0 | 8 |
| L2 Global | 512 KB | 64 | 3 | 10.0 | 32 |
| L3 DRAM | 2 GB | 4 | 50 | 200.0 | 4 |

SCU L1 总计: 38 KB.

`spec_c_worst_case_budget()` 返回 Spec C 的预算验证字典, 在 elaboration / static analysis 时调用.

---

## 5. 顶层 Accelerator

**`hardware/architecture/top_level.py`**

```python
class EdgeStereoDAv2Accelerator:
    sa: UnifiedSystolicArray         # WS/OS 可配置
    crm: ConfidenceRoutingModule     # SCU-1
    gsu: GatherScatterUnit           # SCU-2
    dpc: DualPrecisionController     # SCU-3
    adcu: AbsoluteDisparityCU        # SCU-4
    fu: FusionUnit                   # SCU-5
    arbiter: L2Arbiter
    mem: MemoryHierarchySpec
```

### 5.1 面积分解 (28nm)

| 组件 | 面积 (mm²) |
|------|-----------|
| SA (MAC + sidecars) | 0.455 |
| SA L1 SRAM (64 KB) | 0.064 |
| CRM | 0.035 |
| GSU | 0.011 |
| DPC | 0.003 |
| ADCU | 0.045 |
| FU | 0.054 |
| SCU L1 SRAMs (38 KB) | 0.038 |
| L2 SRAM (512 KB) + crossbar | 0.537 |
| Control Processor | 0.050 |
| IO + Interconnect | 0.400 |
| **总计** | **1.692 mm²** |

### 5.2 功率估计 (28nm/500MHz)

| 组件 | 功率 (mW) |
|------|----------|
| SA | 19.9 |
| 全部 SCU | ~5 |
| Arbiter (L2) | ~75 (50% util) |
| 动态功率 | ~99.9 |
| 漏电 (10%) | 10.0 |
| IO | 30.0 |
| **总计** | **~140 mW** |

### 5.3 帧延迟 (解析估计)

| 配置 | Cycles | Latency | FPS |
|------|--------|---------|-----|
| Dense (kr=1.0) | 194,880,702 | 389.76 ms | 2.57 |
| Merge kr=0.5 | 144,949,830 | 289.90 ms | 3.45 |

**Merge speedup: 1.34x**

注意: 此处为 `top_level.estimate_frame_cycles()` 的解析估计, 与事件驱动仿真结果稍有出入 (后者更精确).

---

## 6. 事件驱动仿真器

**`simulator/core/`** 下 5 个新文件.

### 6.1 EventQueue

**`event_queue.py`**

最小堆优先队列, 元素 `(cycle, priority, counter, Event)`. 主循环:

```python
while not eq.is_empty() and not scheduler.all_done:
    next_cycle = eq.peek_cycle()
    gap = next_cycle - current_cycle
    for mod in modules.values():
        mod.accumulate_gap(gap)        # O(num_modules)
    current_cycle = next_cycle
    for event in eq.drain_cycle(next_cycle):
        ...
```

### 6.2 WorkloadDAG

**`workload_dag.py`**

DAG 节点 `Operation`: id, name, engine, flops, weight/input/output_bytes, predecessors, successors, metadata.

API:
- `add_op()`, `add_edge()`, `add_chain()`
- `topological_order()` (Kahn 算法, 检测循环)
- `ready_ops(completed)` — 全部前驱完成的 op
- `critical_path(cycle_estimates)` — 最长路径
- `from_flat_ops(ops)` — 旧 batch 模型迁移辅助

### 6.3 Workload Builder (`build_workload()`)

按 Spec B 全部展开: ~85 op (dense) / ~110 op (merge).

**Encoder block 依赖链**:
- Sparse: `gather → qkv → attn → out_proj → scatter → mlp`
- Dense: `qkv → attn → out_proj → mlp`

**Decoder 双路径**: 每个受影响 stage 展开为 4 个 op (HP conv + DPC mask + LP conv + FU blend), 经依赖边正确同步.

**当前简化** (待完善):
- MLP 用单 matmul 占位, 未拆分 fc1/gelu/fc2/LN (见 §8 已知问题 C2)
- Attention 用单 matmul 占位, 未展开 flash tile (见 §8 已知问题 I5)

### 6.4 L2Controller

**`memory_controller.py`**

`L2Controller`: 32 bank 跟踪 (`bank_busy_until[]`), 优先级 + round-robin 仲裁. Fast path: 单请求时 `transfer_cycles = ceil(bytes / 256)`.

**当前状态**: 已实现, 但 `simulate_frame()` 中暂未调用 (内存延迟内嵌在各模块周期估计内). 需后续接入主循环 (见 §8 已知问题 I1).

### 6.5 OperationScheduler

**`scheduler.py`**

DAG 感知派发. 维护 `completed`, `in_flight`, `dispatched`. 每次 `op_complete` 事件触发 `dispatch_ready()`, 对 `ready_ops(completed)` 中尚未派发的 op, 把它发给空闲模块.

### 6.6 EventDrivenSimulator

**`event_simulator.py`**

主类. 实例化 6 模块 + L2Controller + scheduler, 跑事件循环, 输出与旧 batch simulator **格式向后兼容**的结果字典 (legacy `engine_breakdown` + 新 `detailed_breakdown` + `dag_summary`).

#### 当前 Benchmark (518×518, ViT-S)

| 配置 | DAG ops | 总周期 | Latency | FPS |
|------|---------|--------|---------|-----|
| Dense | 85 | 98,368,266 | 196.74 ms | 5.08 |
| Merge kr=0.5 | 110 | 84,527,250 | 169.05 ms | 5.92 |

**Merge speedup: 1.16x** (事件驱动模型, 比解析模型保守)

模块利用率 (merge):

| 模块 | busy_cycles | ops | utilization |
|------|-------------|-----|-------------|
| systolic_array | 83,347,037 | 69 | (主导) |
| crm | 52,624 | 1 | — |
| gsu | 152,844 | 24 | — |
| dpc | 17,908 | 6 | — |
| adcu | 124,869 | 1 | — |
| fu | 1,427,652 | 8 | — |

SA 占绝对主导 (~98% busy cycles).

---

## 7. 实现状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| Step 1 | `interfaces.py`, `base_module.py` | ✅ 完成 |
| Step 2 | `unified_sa.py` (WS/OS + sidecars) | ✅ 完成 |
| Step 3-7 | 5 个 SCU (CRM, GSU, DPC, ADCU, FU) | ✅ 完成 |
| Step 8 | `interconnect.py` (L2Arbiter) | ✅ 完成 |
| Step 9 | `memory_hierarchy.py` (revised), `top_level.py` (重写) | ✅ 完成 |
| Step 10-13 | `event_queue.py`, `workload_dag.py`, `memory_controller.py`, `scheduler.py`, `event_simulator.py` | ✅ 完成 |
| Step 14 | Code review 修复 (C1-C4, I4-I6, S1) + `run_simulator.py` 重写 + sparsity sweep | ✅ 完成 |

**总代码量**: ~4,700 行 / 16 文件.

**测试覆盖**: 各模块单元测试 + 顶层集成测试 + 事件驱动仿真器端到端测试 + Step 14 核心修复测试 + 完整 end-to-end 运行均通过.

---

## 8. Code Review 修复记录 (Step 14)

来自 `superpowers:code-reviewer` agent 的评审, 已按优先级修复.

### 已修复问题

| ID | 问题 | 修复方案 | 状态 |
|----|------|----------|------|
| **C1** | SA 仅发首 tile 事件 | `accept_op()` 支持 per-tile 事件发射, 封顶 `MAX_TILE_EVENTS=8`; 新增 `sa:tile_writeback_done` 事件 | ✅ 修复 |
| **C2** | MLP 单 matmul 低估 ~50% | 新增 `_add_mlp_block()` helper, 分解为 fc1 + gelu + fc2 + ln 4 个独立 op | ✅ 修复 |
| **C3** | `dispatch_ready` 非确定性 | `scheduler.dispatch_ready()` 排序 ready ops by op_id | ✅ 修复 |
| **C4** | DPC 无 blend_trigger | 在 mask_gen_done 之后发 `dpc:blend_trigger`; `handle_event` 添加对应 case | ✅ 修复 |
| **I4** | output tag 双路径缺失 | Stage 4 的 output_conv1 走 `_add_dual_path_stage("output", ...)` | ✅ 修复 |
| **I5** | Attention 单 matmul | 新增 `_add_attention_block()` helper, 展开为 QKV + per-flash-tile (QK^T + softmax + AV) + out_proj | ✅ 修复 |
| **I6** | Decoder resize 缺失 | Stage 1 每 tap 在 proj 后加 `decoder_resize_{idx}` op | ✅ 修复 |
| **S1** | `stats.__init__()` 脆弱 | 改用 `mod.stats = ModuleStats()` | ✅ 修复 |

### 已知遗留问题 (接受)

| ID | 问题 | 决定 |
|----|------|------|
| **I1** | `L2Controller` 未接入主循环 | 接受: 内存延迟内嵌在各模块周期估计, contention 不显著. 未来 RTL 验证时引入. |
| **I2** | L2 预算运行时验证 | 接受: 静态 `spec_c_worst_case_budget()` 在 elaboration 时调用已足够 |
| **I3** | GSU chunking 未在 DAG 展开 | 接受: GSU 模块内部按 64-token chunk 建模; DAG 保留 op 粒度为 gather/scatter, 避免 DAG 爆炸 |
| **I8** | `L2Arbiter` 不像 module | 小改, 待 RTL 实现时重构 |

### Spec 合规性 (Step 14 后)

| Spec | 状态 | 说明 |
|------|------|------|
| A (置信度来源) | ✅ PASS | CRM 读外部 conf, ADCU 无 conf |
| B (decoder taxonomy) | ✅ PASS | I4 + I6 修复后 11/12 tag 全覆盖 + output 可选双路径 |
| C (内存预算) | ✅ PASS | 静态验证 496/512 KB OK |
| D (事件契约) | ✅ PASS | per-tile SA 事件 + blend_trigger + 排序确定性 |

### Sparsity Sweep 结果 (28nm/500MHz)

| keep_ratio | policy | FPS | Latency (ms) | DAG ops |
|-----------|--------|-----|--------------|---------|
| 1.00 | coarse_only | 4.75 | 210.6 | 904 |
| 1.00 | all | 4.40 | 227.3 | 925 |
| 0.85 | coarse_only | 5.00 | 200.1 | 821 |
| 0.85 | all | 4.61 | 216.7 | 842 |
| 0.75 | coarse_only | 5.19 | 192.6 | 749 |
| 0.75 | all | 4.78 | 209.2 | 770 |
| 0.60 | coarse_only | 5.60 | 178.6 | 605 |
| 0.60 | all | 5.12 | 195.2 | 626 |
| **0.50** | **coarse_only** | **5.81** | **172.1** | **533** |
| 0.50 | all | 5.30 | 188.8 | 554 |

**论文关键数据**:
- Merge speedup (kr=0.5): **1.22x** 
- DPC 开销 (all vs coarse_only): ~7%
- 最优配置: kr=0.5 + coarse_only, 5.81 FPS @ 1.692 mm² / 140 mW @ 28nm

---

## 9. 后续路线

### 9.1 短期 (完成)

- ✅ Code review 4 Critical / 4 Important 问题修复
- ✅ `simulator/run_simulator.py` 重写用 `EventDrivenSimulator`
- ✅ 完整 sparsity sweep (10 config) + JSON 输出

### 9.2 中期 (RTL 实现)

按当前 Python 行为模型的接口和 pipeline 划分, 对应 Verilog 模块:

| Python 类 | Verilog 模块 |
|-----------|--------------|
| `UnifiedSystolicArray` | `unified_sa.v` (32×32 PE 阵列 + skew + control FSM) |
| `PiecewiseLinearSoftmax` | `softmax_pwl.v` (4 stage pipeline, 16 segment LUT) |
| `GELUApprox` | `gelu_pwl.v` (2 stage, 32 segment) |
| `ConfidenceRoutingModule` | `crm.v` (bitonic sort + 32 distance unit) |
| `GatherScatterUnit` | `gsu.v` (address gen + DMA-style streaming) |
| `DualPrecisionController` | `dpc.v` (resize + histogram + control) |
| `AbsoluteDisparityCU` | `adcu.v` (NCC + LS + LUT + applicator) |
| `FusionUnit` | `fu.v` (3 sub-engines) |
| `L2Arbiter` | `l2_arbiter.v` (32 bank crossbar + priority arb) |

### 9.3 长期 (FPGA 验证)

目标 FPGA: Xilinx ZCU102 / U250 (大容量 BRAM, DSP-rich).

测试流程:
1. 综合 + 时序验证 (500 MHz target)
2. SystemVerilog testbench, 对照 Python 行为模型黄金参考
3. 端到端 demo: 实际跑 KITTI 一帧, 输出与算法 GPU 参考视差对比

---

## 10. 关键源码参考

**算法参考** (硬件设计的依据):

| 文件 | 用途 |
|------|------|
| `core/sgm_wrapper.py` | `_pkrn()` — PKRN 置信度计算 (外部) |
| `core/sparse_attention.py` | `merge_block_forward`, GAS 函数 |
| `core/token_merge.py` | `build_token_merge_groups`, `build_caps_merge_plan` |
| `core/decoder_adaptive_precision.py` | `run_dpt_decoder_with_weight_adaptive_precision` (Spec B 完整结构) |
| `core/fusion.py` | `fuse_edge_aware_residual` |
| `core/pipeline.py` | `align_depth_to_sgm` |

**硬件代码** (本设计文档对应的实现):

```
hardware/
├── interfaces.py
├── base_module.py
├── pe_array/
│   ├── unified_sa.py          (新)
│   ├── mhsa_pe.py             (legacy 参考)
│   ├── conv_pe.py             (legacy 参考)
│   └── upsample_pe.py         (logic 已并入 fu.py)
├── scu/
│   ├── __init__.py
│   ├── crm.py
│   ├── gsu.py
│   ├── dpc.py
│   ├── adcu.py
│   └── fu.py
└── architecture/
    ├── interconnect.py        (新)
    ├── memory_hierarchy.py    (重写, 加 SCU L1)
    ├── top_level.py           (大重写)
    └── dataflow.py            (待 Step 14 更新)

simulator/core/
├── simulator.py               (legacy batch 保留)
├── event_queue.py             (新)
├── workload_dag.py            (新)
├── memory_controller.py       (新)
├── scheduler.py               (新)
└── event_simulator.py         (新, 主入口)
```

**项目计划文档**: `C:\Users\14527\.claude\plans\frolicking-weaving-forest.md` (本地)

---

## 11. 文档维护

本文档反映的硬件状态对应:
- `hardware/` 全部内容
- `simulator/core/` 中的 5 个新文件
- 不包含 `simulator/analysis/`, `simulator/config/`, `simulator/run_*.py` (这些是 Step 14 待办)

**更新规则**:
- 增加新模块时, 在 §4 加一节描述 (Configuration / Pipeline / 周期 / 面积)
- 改 SA / SCU 的接口或周期模型时, 同步更新对应章节并在 §7 标记
- 修复已知问题时, 从 §8 移除并在变更日志注明
- RTL 实现里程碑达成时, 更新 §9.2 表格

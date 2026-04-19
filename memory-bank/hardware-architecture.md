# 硬件加速器文件架构

**项目**: SGM-ViT EdgeStereoDAv2 硬件加速器
**路径**: `/home/pdongaa/workspace/SGM-ViT/hardware/` 和 `simulator/core/`
**最后更新**: 2026-04-14 (Step 14 完成)

本文档记录硬件相关每个文件的作用, 便于快速定位和理解代码.

---

## 顶层结构

```
SGM-ViT/
├── hardware/                       # 硬件架构建模 (Python 行为级)
│   ├── interfaces.py               # 信号 / 端口协议
│   ├── base_module.py              # HardwareModule 抽象基类
│   ├── pe_array/                   # PE 阵列 (主算力)
│   │   ├── unified_sa.py           # 统一脉动阵列 (主体)
│   │   ├── mhsa_pe.py              # legacy: MHSA PE (已并入 unified_sa)
│   │   ├── conv_pe.py              # legacy: Conv PE (已并入 unified_sa)
│   │   └── upsample_pe.py          # legacy: 上采样 PE (已并入 fu.py)
│   ├── scu/                        # Special Compute Units (5 个)
│   │   ├── __init__.py
│   │   ├── crm.py                  # SCU-1: Confidence Routing Module
│   │   ├── gsu.py                  # SCU-2: Gather-Scatter Unit
│   │   ├── dpc.py                  # SCU-3: Dual Precision Controller
│   │   ├── adcu.py                 # SCU-4: Absolute Disparity Calibration Unit
│   │   └── fu.py                   # SCU-5: Fusion Unit
│   ├── architecture/               # 顶层架构集成
│   │   ├── top_level.py            # EdgeStereoDAv2Accelerator (SA + 5 SCUs)
│   │   ├── memory_hierarchy.py     # L0-L3 存储层次 + Spec C 验证
│   │   ├── interconnect.py         # L2 32-bank arbiter + buffer 跟踪
│   │   └── dataflow.py             # 数据流分析 (待 Step 14 更新)
│   └── ViTCoD_ref/                 # ViTCoD 参考 (论文对比, 不参与实现)
│
└── simulator/core/                 # Cycle-accurate 事件驱动仿真器
    ├── simulator.py                # legacy batch-mode (保留作 baseline 对比)
    ├── event_queue.py              # 最小堆事件队列 + event-skip 设计
    ├── workload_dag.py             # Operation DAG + 拓扑排序 + 关键路径
    ├── memory_controller.py        # L2Controller (32 bank 跟踪)
    ├── scheduler.py                # OperationScheduler (DAG 感知派发)
    └── event_simulator.py          # EventDrivenSimulator (主入口)
```

---

## hardware/ — 硬件行为模型

### Foundation 层

| 文件 | 作用 |
|------|------|
| `interfaces.py` | **信号/端口协议**. 定义 `Signal` (单 wire, name+width+direction), `Port` (signal bundle + ready/valid 握手), `MemoryPort` (addr_width=20, data_width=256 即 32B/cycle), `StreamPort` (data + tlast), `ConfigPort` (低带宽配置写入). 全部映射到 RTL 的 AXI-Stream/AXI-Lite 接口. |
| `base_module.py` | **硬件模块基类**. 定义 `ModuleState` (IDLE/FETCH/COMPUTE/WRITEBACK/STALL_MEM/STALL_DEP/DRAIN/RECONFIGURE 共 8 状态), `ModuleStats` (各类周期累计 + `accumulate_gap()` 实现 Spec D 批量累计), `Event` (`@dataclass(order=True)` 用于 heap 排序), `PipelineStage`, `HardwareModule` 抽象类 (强制实现 `_declare_ports` / `accept_op` / `handle_event` / `describe` / `estimate_area_mm2` / `estimate_power_mw`). |

### pe_array/ — PE 阵列

| 文件 | 作用 |
|------|------|
| `unified_sa.py` | **统一脉动阵列 (主体)**. `UnifiedSystolicArray` 支持 WS/OS/IS 三种数据流, 4 周期模式切换. 内部 5 阶段 pipeline (weight_load → skew → compute → de-skew → drain), 双缓冲 steady state 模型. 包含 `SAConfig`, `DataflowMode` 枚举, `SidecarSpec`. **Sidecar 单元** (与 SA MAC 互斥): `SoftmaxSidecar` (4 stage PWL, 16 segment), `GELUSidecar` (2 stage PWL, 32 segment), `LayerNormSidecar` (3 stage). API: `estimate_matmul_cycles`, `estimate_conv_cycles`, `estimate_attention_cycles` (含 flash tiling), `estimate_mlp_cycles`, `estimate_sidecar_cycles`. `accept_op` 支持 5 种 op_type (`matmul`, `conv1x1`, `conv3x3`, **`sidecar`** — Step 14 新增用于 softmax/gelu/ln sidecar 路由). **C1 修复**: per-tile 事件发射 (`sa:weight_load_done` / `sa:tile_compute_done` / `sa:tile_writeback_done`), 封顶 `MAX_TILE_EVENTS=8`. 28nm 面积 0.455 mm². |
| `mhsa_pe.py` | **legacy 参考**. 旧 MHSA PE (32×32, WS only). 包含 `MHSAPEConfig`, `INT8MAC`, `PiecewiseLinearSoftmax` (源代码已迁入 unified_sa), `GELUApprox`, `SystolicPEArray`. **不再使用**, 仅作历史参考. |
| `conv_pe.py` | **legacy 参考**. 旧 ConvolutionEngine (16×16 OS, 1×1 + 3×3). 逻辑已合并到 `unified_sa.py:estimate_conv_cycles()`. **不再使用**. |
| `upsample_pe.py` | **legacy 参考**. 旧 BilinearUpsampleUnit (shift-add). 逻辑已合并到 `fu.py:BilinearUpsampleEngine`. **不再使用**. |

### scu/ — Special Compute Units

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包导出: `ConfidenceRoutingModule`, `GatherScatterUnit`, `DualPrecisionController`, `AbsoluteDisparityCU`, `FusionUnit` 及对应 Config 类. |
| `crm.py` | **SCU-1: Confidence Routing Module**. 读**外部 SGM** 提供的 PKRN 置信度图 (DRAM → L2), 池化到 token grid, 生成 merge plan. 3 模式: PRUNE/MERGE/CAPS_MERGE. 内部 pipeline: Pool → Sort (bitonic) → Select → Assign (32 并行欧氏距离单元) → (CAPS: Score → HP-Select) → LUT writeback. 周期 (kr=0.5): 52,624. 28nm 面积 0.035 mm². **Spec A 合规**: 不依赖 ADCU. |
| `gsu.py` | **SCU-2: Gather-Scatter Unit**. L2 ↔ SA L1 之间的索引选址 + 数据搬运 (无算术). 3 模式: GATHER (compact 收) / SCATTER_PRUNE (M 个 kept 写回) / SCATTER_MERGE (representative 一对多广播). **Streaming**: 64 token chunk = 24KB, 满足 Spec C 的 L2 预算. 周期: gather 685 reps = 4,246; scatter_merge 1369 = 8,491. 28nm 面积 0.011 mm². |
| `dpc.py` | **SCU-3: Dual Precision Controller**. 解码器双路径执行的**控制器** (不算力). 覆盖 Spec B 全部 11 个 decoder tag (proj_1-4, rn_1-4, path_1-4) + output. Pipeline: mask_gen (resize sensitivity → histogram top-K) → SA HP conv → SA LP conv → FU blend. `should_apply_decoder_precision(tag, stage_policy)` 实现 coarse_only / all / fine_only 策略. **C4 修复**: 在 mask_gen_done 之后发 `dpc:blend_trigger` 事件 (Spec D). 周期 (coarse_only): 1,003,588 overhead. 28nm 面积 0.003 mm². |
| `adcu.py` | **SCU-4: Absolute Disparity Calibration Unit**. 单目相对深度 → 绝对视差. 3 子单元: 稀疏 NCC matcher (32 keypoints, 11×11, max_disp=192, 32 并行 correlator), Scale-Shift LS solver (累加 A^T A + 2×2 闭式逆), 像素级 applicator (z = s·d + t → 1/z LUT → disp = B·f/z, 32 px/cycle). **Spec A**: 移除了 confidence pooler, 不产生置信度. 周期 (518×518): 124,869. 28nm 面积 0.045 mm². 8KB LUT 跨 frame 持久驻留. |
| `fu.py` | **SCU-5: Fusion Unit**. 像素级运算 + 双线性上采样. 3 子引擎: `BilinearUpsampleEngine` (shift-add, 32 px/cycle, 吸收旧 upsample_pe), `ElementWiseEngine` (32 lane: mul/add/cmp/clamp/alpha_blend), `SpatialFilterEngine` (3×3 kernel, Gaussian/Sobel/bilateral, 4 output px/cycle). **Streaming**: 32 行 strip, L2 同时 3 strip = 200KB. API: `estimate_bilateral_filter_cycles`, `estimate_gaussian_blur_cycles`, `estimate_sobel_cycles`, `estimate_upsample_2x_cycles`, `estimate_fusion_pipeline_cycles`. 周期 (518×518 完整 fusion): 431,018 ≈ 0.86 ms. 28nm 面积 0.054 mm². |

### architecture/ — 顶层架构

| 文件 | 作用 |
|------|------|
| `top_level.py` | **顶层 Accelerator**. `EdgeStereoDAv2Accelerator` 类实例化 SA + 5 SCUs + L2Arbiter + MemoryHierarchySpec + ControlProcessor. `AcceleratorConfig` 聚合所有子配置. API: `full_spec()` (完整规格), `area_breakdown()` (含 SCU 的面积分解), `power_estimate()` (功率), `estimate_frame_cycles(keep_ratio, stage_policy, ...)` (解析帧延迟估算). 28nm 总面积 1.692 mm², 总功率 ~140 mW. |
| `memory_hierarchy.py` | **存储层次规格**. `MemoryHierarchySpec` 定义 9 级 (L0 RegFile / L1 SA / L1 CRM / L1 GSU / L1 DPC / L1 ADCU / L1 FU / L2 Global / L3 DRAM), 每级有 capacity, bandwidth, latency, energy, banks 字段. SCU L1 总计 38 KB. 关键 API: `validate_l2_budget(allocations)`, `spec_c_worst_case_budget()` (返回 6 项 buffer 占用 + 验证, 当前 496/512 KB OK). `data_placement_strategy()` 给每级写明驻留内容和生命周期. |
| `interconnect.py` | **L2 仲裁器**. `L2Arbiter` 模拟 32-bank L2 crossbar. `ArbiterConfig` (num_banks=32, bank_size=16KB, bank_width=8B, latency=3). 优先级 `MODULE_PRIORITIES` (SA=0 最高, DMA=6 最低). Bank busy-until 跟踪 (event-driven 仿真用). Buffer 跟踪 API: `allocate_buffer` / `free_buffer` / `validate_budget`. 28nm 面积 (含 512KB L2): 0.537 mm². |
| `dataflow.py` | **数据流分析** (待 Step 14 更新). 包含 `DataflowAnalyzer` 的 WS/OS 数据流分析逻辑. 当前仍是旧版本, 引用旧 `pe_array` 类. **Step 14 待办**: 接新 `unified_sa` 接口. |

### 其他

| 路径 | 作用 |
|------|------|
| `__init__.py` (各级) | Python 包标识 |
| `ViTCoD_ref/` | ViTCoD 论文的参考实现 (Algorithm + Hardware Simulator). 仅作论文对比基准, **不参与本项目实现**. |
| `__pycache__/` | Python 字节码缓存, 不计入版本控制 |

---

## simulator/core/ — 事件驱动仿真器

| 文件 | 作用 |
|------|------|
| `simulator.py` | **legacy batch-mode 仿真器**. 旧 `CycleAccurateSimulator`, batch 累加 effective_cycles, 假设全部串行. **保留作 Step 14 验证 baseline** (新 EventDrivenSimulator 应不慢于 batch + 反映并行). 含 `SimConfig`, `SparsityConfig`, `SimStats`, `ModuleState`, `HardwareModule` (旧版本, 与新 `hardware/base_module.py` 并存). |
| `event_queue.py` | **事件队列**. `EventQueue` 最小堆 (cycle, priority, counter, Event) 元组排序. **Event-skip 设计**: 主循环跳到下一事件周期, 不逐周期 tick. API: `push`, `pop`, `peek_cycle`, `drain_cycle`, `is_empty`. 估计 80 op × 4 phase ≈ 320 事件/frame, 远少于 160M 周期. |
| `workload_dag.py` | **操作 DAG**. `Operation` 节点 (id, name, engine, flops, weight/input/output_bytes, predecessors, successors, metadata). `WorkloadDAG` 提供: `add_op`, `add_edge`, `add_chain`, `topological_order` (Kahn, 检测循环), `ready_ops(completed)`, `critical_path(cycle_estimates)`, `from_flat_ops(ops)` (旧 batch 模型迁移辅助). |
| `memory_controller.py` | **L2 控制器**. `L2Controller` 32 bank 跟踪 + 优先级仲裁 + fast-path. `MemoryTransfer` 数据类. `request_transfer(requester, type, bytes, cycle)` 返回 (completion_cycle, callback_event). 当前内存延迟内嵌在各模块周期估计 (I1 接受遗留). |
| `scheduler.py` | **DAG 感知调度器**. `OperationScheduler` 维护 `completed`, `in_flight`, `dispatched`. `dispatch_ready(cycle, modules, eq)` 找全部前驱完成的 ready ops, 派发到空闲模块. `mark_complete(op_id, cycle, modules, eq)` 在 `op_complete` 事件触发时更新. `all_done` 判完成. **C3 修复**: ready ops 按 op_id 排序保证仿真可重现. |
| `event_simulator.py` | **主仿真器**. `EventDrivenSimulator` 实例化 6 模块 (SA + 5 SCUs) + L2Controller + scheduler. `SimConfig` 配置 (clock, PE 大小, L2, sparsity, gsu_chunk_size). `build_workload(config, ...)` 完整展开 ~904 op (dense) / ~533 op (merge), 按 Spec B 全部 decoder 阶段. **Step 14 新增 helper 函数**: `_add_dual_path_stage` (Spec B 双路径 decoder stage), `_add_attention_block` (I5: QKV + flash tile QK^T/softmax/AV + out_proj), `_add_mlp_block` (C2: fc1+gelu+fc2+ln). `simulate_frame()` event-skip 主循环, 跳事件周期, 批量累计 idle/stall, 结果向后兼容旧 simulator JSON 格式. |

### simulator/ 根目录

| 文件 | 作用 |
|------|------|
| `run_simulator.py` | **主入口脚本** (Step 14 重写). 运行完整分析套件: dense 仿真 (28nm + 7nm) / merge 仿真 / 面积 + 功率 / sparsity sweep (5 kr × 2 policy). 输出 `results/simulation_results.json` + `results/sparsity_sweep.json`. API: `run_event_simulation(kr, node, freq, policy)`, `run_area_power_analysis(node, freq)`, `run_sparsity_sweep(keep_ratios, stage_policies, ...)`. |
| `run_sparsity_sweep.py` | **细粒度 sparsity 扫描** (legacy, 保留). 老脚本, 功能被 `run_simulator.py:run_sparsity_sweep()` 覆盖. |
| `results/` | JSON 输出目录. `simulation_results.json` (总结果), `sparsity_sweep.json` (10 config 表). |

---

## 测试与验证

| 测试 | 路径 | 状态 |
|------|------|------|
| Foundation 单元测试 | `/tmp/test_foundation.py` (本地临时) | ✅ 通过 |
| Unified SA 测试 | `/tmp/test_unified_sa.py` (本地临时) | ✅ 通过 |
| 5 SCU 测试 | `/tmp/test_all_scu.py` (本地临时) | ✅ 通过 |
| 顶层集成测试 | `/tmp/test_top_level.py` (本地临时) | ✅ 通过 |
| 事件驱动仿真器测试 | `/tmp/test_event_sim.py` (本地临时) | ✅ 通过 |
| Step 14 核心修复测试 | `/tmp/test_step14_core.py` (本地临时) | ✅ 通过 (8/8) |
| 完整 end-to-end 运行 | `simulator/run_simulator.py` | ✅ 通过, 输出 JSON |

---

## 关键 API 速查

### 创建一个加速器实例

```python
from hardware.architecture.top_level import EdgeStereoDAv2Accelerator, AcceleratorConfig

accel = EdgeStereoDAv2Accelerator()  # 默认 28nm/500MHz, 32x32
print(accel.area_breakdown())        # 面积分解
print(accel.power_estimate())        # 功率
print(accel.estimate_frame_cycles(keep_ratio=0.5))  # 帧延迟
```

### 跑事件驱动仿真

```python
from simulator.core.event_simulator import EventDrivenSimulator, SimConfig

sim = EventDrivenSimulator(SimConfig(keep_ratio=0.5, stage_policy="coarse_only"))
results = sim.simulate_frame(img_h=518, img_w=518)
print(f"Total cycles: {results['total_cycles']:,}")
print(f"FPS: {results['fps']}")
```

### 查 SA cycle 估算

```python
from hardware.pe_array.unified_sa import UnifiedSystolicArray

sa = UnifiedSystolicArray()
est = sa.estimate_attention_cycles(seq_len=1370, embed_dim=384, num_heads=6)
print(est['total_cycles'])
```

### 验证 L2 预算

```python
from hardware.architecture.memory_hierarchy import MemoryHierarchySpec

mem = MemoryHierarchySpec(28)
budget = mem.spec_c_worst_case_budget()
assert budget['ok'], f"L2 exceeded: {budget['total_bytes']} > 512KB"
```

### 检查 DAG 拓扑

```python
from simulator.core.event_simulator import build_workload, SimConfig

dag = build_workload(SimConfig(keep_ratio=0.5))
print(dag.summary())
print(dag.ops_by_engine())
print(dag.critical_path())
```

---

## 文件维护规则

- **新增模块**: 在对应目录 (pe_array / scu / architecture / simulator/core) 加文件, 同步更新本文档对应表
- **删除/合并**: legacy 文件 (mhsa_pe, conv_pe, upsample_pe) 保留作参考, 不删
- **接口变更**: 同步更新 §"关键 API 速查"
- **Step 14 完成后**: 把 `dataflow.py` 状态从"待更新"改为"已更新", 加正式测试套件路径

---

## 文档参考

- **设计文档**: `memory-bank/hardware-design-document.md` — 完整规格 (Spec A-D + 模块详细设计)
- **进度记录**: `memory-bank/hardware-progress.md` — 每个 Step 的执行 / 验证 / 关键决策
- **项目架构**: `memory-bank/architecture.md` — 全项目文件索引 (含算法侧)
- **本地计划文件**: `C:\Users\14527\.claude\plans\frolicking-weaving-forest.md`

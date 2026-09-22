# nano_perception 部署与验证手册

> 本文档面向"把 nano_perception 部署到另一台 Jetson 机器人"的场景：包含架构说明、环境构建、部署步骤、参数开关、**踩坑与解决方案**。
>
> 基准机器：ubuntu（Jetson AGX Orin，JetPack 5.1.3），以最近一次真机实测为准。

---

## 1. 系统架构

两个独立进程、一个启动脚本、HTTP 解耦：

```
相机驱动 (odin1, /odin1/image/undistorted, bgr8 1600x1296, ~10Hz)
        │ 订阅
        ▼
nano_perception_bridge.py（系统 python3.8 + ROS1 noetic）
  - 节流（max_rate，忙时丢帧）+ JPEG 编码
  - HTTP POST /infer（127.0.0.1:8891）
  - 收到 JSON（检测框 + RLE 掩码）后画 overlay、发 4 个 ROS topic
        │ HTTP
        ▼
nano_perception_engine.py（conda env nano_perception + TensorRT）
  - OWL 图像/文本编码器（open-vocabulary 检测，6 类 prompt）
  - SAM（resnet18 编码器 + mobile_sam 解码器）逐框分割
  - 掩码 RLE 游程编码（比 PNG 小 10-30 倍、编码亚毫秒）
```

**为什么拆两个进程**：torch/TensorRT 只能在 conda env 里跑（JetPack 5 的 NGC torch），而 cv2/rospy/cv_bridge 只能可靠地跑在系统 python3.8（conda 里装 cv2 会触发 libffi 冲突崩溃，见 §5.1）。用 HTTP + JSON 解耦，互不干扰。

**4 个 ROS topic**（命名空间 /nano_perception/）：

| topic | 类型 | 内容 |
|---|---|---|
| segmentation_image | sensor_msgs/Image (bgr8) | 彩色叠加图（掩码染色 + 框 + "label score"） |
| segmentation_result | std_msgs/String | JSON：stamp_ns / engine_ms / detections（label、score、box_xyxy） |
| masks | sensor_msgs/Image (mono16) | 位掩码图：像素值 = Σ 2^i，第 i 类命中 bit i |
| engine_latency | std_msgs/Float64 | 引擎单帧耗时 ms |

**引擎 `/infer` 协议**：请求 `{image_b64, prompts, confidence_threshold}`；响应 `{engine_ms, server_ms, results: [{prompt, scores, boxes, masks_png_b64}]}`。掩码字段名保留旧名但内容是 `rle:` 前缀的 base64 游程编码（uint32 小端 `[h, w, run0, run1, ...]`，段值 0 起交替）；无前缀 = 旧 PNG 格式（引擎/bridge 双向兼容，滚动部署零破坏）。

---

## 2. 环境要求与依赖

### 2.1 系统

| 项 | 基准机实测版本 | 说明 |
|---|---|---|
| JetPack | 5.1.3（CUDA 11.4） | JetPack 6 需重新导出 TRT engine + 换 NGC torch，未验证 |
| TensorRT | 8.5.2.2 | apt 装：`libnvinfer-dev` + `python3-libnvinfer`（python 绑定是系统 apt 的，conda 靠 PYTHONPATH 接入，见 §5.3） |
| ROS1 | noetic | 需 cv_bridge；ROS_MASTER_URI 指到 master（本机 http://localhost:11311） |
| 系统 python3 | 3.8 | bridge 进程用：实测 numpy 1.17.4、cv2 4.2.0（apt 版 python3-opencv） |
| GPU 锁频 | `sudo jetson_clocks` | 所有性能数字都基于锁频状态 |

### 2.2 conda env（引擎进程，名字 nano_perception）

| 包 | 实测版本 | 备注 |
|---|---|---|
| torch | 2.1.0a0+41361538.nv23.6 | JetPack 5 配套 NGC 容器版 torch，别用 PyPI 版 |
| torchvision | 0.16.1+fdea156 | 与上面 torch 配套 |
| torch2trt | 0.5.0 | nanoowl 导出 engine 用 |
| numpy | 1.24.4 | |
| Pillow | 10.4.0 | |
| cv2 | **不装** | 装了会崩，见 §5.1 |

### 2.3 模型与权重

| 文件 | 位置 | 说明 |
|---|---|---|
| owl_image_encoder_patch32.engine | repo `data/` | OWL-ViT B/32 patch32 图像编码器（TensorRT） |
| resnet18_image_encoder.engine | repo `data/` | SAM 图像编码器（TensorRT） |
| mobile_sam_mask_decoder.engine | repo `data/` | SAM 掩码解码器（TensorRT） |
| mobile_sam.pt | nanosam/assets/ | SAM 原始权重（导出 decoder 的输入） |
| nanoowl 仓库 | repo/nanoowl（commit fb553de） | OWL 文本编码器权重在 nanoowl/assets |
| nanosam 仓库 | repo/nanosam（commit 6536336） | |

---

## 3. 部署步骤

### 3.1 代码与模型

1. 克隆仓库，`nanoowl`/`nanosam` 是 git 子模块（各自独立 git），拉取时注意 commit 对齐（fb553de / 6536336，已实测配合）。
2. 下载权重：OWL-ViT open_clip B/32 patch32（放 nanoowl/assets）、mobile_sam.pt（放 nanosam/assets）。
3. 生成 3 个 TRT engine 放 `data/`（owl/nanosam 各自 README 的导出流程：torch → ONNX → trtexec；nanoowl 参考 examples/tree_demo.py，nanosam decoder 参考 export_sam_mask_decoder_onnx.py 起点）。
   - **engine 与 TensorRT/CUDA 版本强绑定**：换 JetPack 版本必须重新导出；跨设备跑 engine 会出 TRT 警告且可能崩（§5.11）。
4. 复制启动脚本（go2_ws/start_nano_perception.sh）与 prompts 文件（objectnav-realworld-prompts.txt）。

### 3.2 环境构建要点

```bash
# 系统侧：TensorRT python 绑定 + opencv（bridge 用）
sudo apt install python3-libnvinfer python3-opencv

# conda env：JetPack 5 配套 NGC torch（以实际 NGC 容器为准）
conda create -n nano_perception python=3.8
pip install torch==2.1.0a0+41361538.nv23.6 torchvision==0.16.1+fdea156 torch2trt==0.5.0
pip install numpy==1.24.4 Pillow==10.4.0
```

启动脚本内置两个关键 export（照抄即可，路径按机器改）：

```bash
# torch 缺 libopenblas：LD_PRELOAD 精确加载，别把整个 env/lib 塞 LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_ENV/lib/libopenblas.so.0
# TensorRT python 绑定是 apt 装的，conda 靠 PYTHONPATH 接入
export PYTHONPATH=/usr/lib/python3.8/dist-packages${PYTHONPATH:+:$PYTHONPATH}
```

### 3.3 启动与验证

```bash
bash start_nano_perception.sh   # Ctrl-C 一键收掉两个进程
```

验证链（每步都通才算部署成功）：

1. `curl -s http://127.0.0.1:8891/health` → 返回 `ready`（引擎加载+预热约 12-180 秒）
2. `rosnode info /nano_perception_bridge` → 节点注册正常、订阅了相机话题
3. `rostopic hz /nano_perception/segmentation_image` → 有稳定发布（基准机 4.6-5.4Hz）
4. `timeout 60 rostopic echo -n 1 /nano_perception/segmentation_result` → 拉到 JSON，`engine_ms` 与 `detections` 非空（收到 1 条即退出 0；超时被杀返回 124 = 没数据）
5. 告警检查：`grep -E "WARNING|ERROR" ~/.ros/log/nano_perception_bridge.log`（rospy 日志写这里不写终端，§5.6）

---

## 4. 参数与开关

### 4.1 start 脚本 env（默认值已按实测甜点落定）

| 变量 | 默认 | 作用 |
|---|---|---|
| ENGINE_PORT | 8891 | 引擎 HTTP 端口 |
| SAVE_DIR | 空 | 非空时 bridge 把 overlay 图 + JSON 摘要成对落盘（约 2.6 对/秒 × 0.35MB ≈ 3.3GB/小时，用完记得关） |
| SAVE_RAW_FRAMES | 0 | 1 = 额外存原始相机帧 raw_*.jpg（离线阈值扫描用） |
| CONFIDENCE_THRESHOLD | 0.08 | 检测分数阈值（0.08 是实测甜点：0 空帧、分数压线分布） |
| MIN_BOX_AREA | 100 | 小框面积过滤（768 高坐标系 px²，约合原图 285px²） |
| MAX_RATE | 10 | bridge 发布节流上限（Hz）；实际发布由引擎耗时决定 |
| MASK_CODEC | rle | 掩码编码：rle=游程（最快，默认）/ fast=torchvision PNG / pil=旧 PIL PNG（回退） |
| SAM_PREPROC | fast | SAM 预处理：fast=GPU 化（默认）/ legacy=旧 PIL CPU 路径（回退） |

### 4.2 bridge 私有 ROS 参数（脚本自动传）

`_engine_port` / `_prompts`（逗号分隔类别）/ `_confidence_threshold` / `_save_dir` / `_save_raw_frames` / `_max_rate`。

### 4.3 新机器必改项

- **相机话题名**：bridge 里硬编码 `/odin1/image/undistorted`（订阅处），新机器人按实际改。
- **相机分辨率**：odin1 是 1600x1296 bgr8；分辨率不同时性能数字需重测。
- **prompts**：objectnav-realworld-prompts.txt 一行逗号分隔；改后只重启管线（engine 与 bridge 各有一份 DEFAULT_PROMPTS 兜底，两处都要同步）。
- **ROS master**：ROS_MASTER_URI 指向实际 master。

---

## 5. 踩坑与解决方案

> 每条都是实机踩过、验证过解法的。部署新机器时逐个对照。

| # | 坑 | 现象 | 根因 | 解决方案 |
|---|---|---|---|---|
| 5.1 | conda 里装 cv2 | 引擎进程崩溃，报 libp11-kit / libffi 相关错误 | conda env 的 libffi 与系统 libp11-kit 冲突，cv2 加载时炸 | conda env **不装 cv2**；bridge 用系统 python3.8（apt python3-opencv），两进程架构就是为此设计的 |
| 5.2 | torch 缺 libopenblas | 引擎 import torch 报错找不到 libopenblas.so.0 | JetPack NGC torch 依赖系统没有的 openblas | `LD_PRELOAD=<conda env>/lib/libopenblas.so.0` 精确加载；**不要**把整个 env/lib 塞进 LD_LIBRARY_PATH（会触发 5.1 的 libffi 冲突） |
| 5.3 | conda 里 import tensorrt 失败 | 引擎启动报 No module named 'tensorrt' | TensorRT python 绑定由 apt 的 python3-libnvinfer 提供，装在系统 python 里 | `export PYTHONPATH=/usr/lib/python3.8/dist-packages`（追加模式，别覆盖） |
| 5.4 | 参数服务器空值不覆盖 | 明明 SAVE_DIR 为空，bridge 却往旧目录存盘 | rospy 私有参数传空值 `_save_dir:=` 不覆盖参数服务器上的残留旧值 | 启动脚本**非空才传** `_save_dir`；排查时 `rosparam get /nano_perception_bridge/save_dir` 看残留，`rosparam delete` 清理 |
| 5.5 | 引擎日志"不更新" | engine.log 停在旧时间戳，但引擎其实活着/或崩溃无痕迹 | conda python stdout 重定向到文件是全缓冲，日志驻留缓冲区；进程崩溃时缓冲丢失 | 引擎状态判定只信 `curl /health` + `kill -0 <pid>`，**不看日志文件新旧** |
| 5.6 | bridge"零输出" | 终端/管线日志看不到 bridge 任何日志，以为卡死 | rospy 日志写 `~/.ros/log/<节点名>.log`，不写 stdout | 排查 bridge 先读 `~/.ros/log/nano_perception_bridge.log`；节点死活以 `rosnode info` + `/proc/<pid>` 为准 |
| 5.7 | 高频请求大量丢帧 | 引擎忙时新请求被秒拒 503 | 引擎 HTTP 服务器带 busy 锁，忙时**直接回 503 不排队**（防止请求堆积撑爆 GPU 显存） | bridge 侧已有完整对策：引擎空闲预测（响应到达时刻 = 空闲，下一帧等到空闲+5ms 再发）+ 503 短等重试（20ms×4）；bridge 忙时丢帧是**设计行为**不是 bug |
| 5.8 | 残留引擎占端口 | 新引擎 bind 失败；或健康检查被残留服务欺骗"假就绪" | 上次运行没杀干净的引擎进程还占着 8891 | 启动脚本轮询时先 `kill -0` 检查**自己的引擎 PID** 再查 /health；清理残留用精确命令（§5.12），别 pkill 宽匹配 |
| 5.9 | 相机"没数据"但话题在 | 拉帧超时，`rostopic list` 却显示话题正常 | 相机驱动假死：话题注册还在但数据停发（驱动退出/卡住） | 数据探测用 `timeout N rostopic echo -n 1 <话题>/<字段>`（收到 1 条即退 0，超时 124 = 没数据）；**不要**用 rostopic list 判断、**不要**用 `rostopic hz \| grep`（grep 退出后 hz 不死，管道挂满 timeout） |
| 5.10 | 多话题同步卡死 | 用 ApproximateTimeSynchronizer 同步 image/cloud 永远等不到 | odin1 图像话题混用两种时间戳：多数是传感器相对时钟（几百秒量级），每隔几条混一条墙钟 Unix 秒，按头时间戳同步被墙钟消息永久卡死 | 多话题同步**按回调到达时间对齐**，不要按 header 时间戳（参考工具 quadruped_workbench/go2_software/tools/capture_odin1_frame.py） |
| 5.11 | TRT 警告"Using an engine plan file across different models of devices" | 引擎日志反复出现该警告 | engine 文件带设备指纹，跨 JetPack/设备跑不保证正确 | 换 JetPack 版本/换机型必须**重新导出 engine**；同机同版本该警告可忽略 |
| 5.12 | 清理进程误杀 | 清"孤儿进程"把健康引擎/其他服务也杀了 | pgrep/pkill 宽匹配抓到无关进程（含自己的 bash 包装） | 精确匹配：`ps -eo pid,cmd \| awk '$2=="/usr/bin/python3" && $0 ~ /nano_perception_bridge\.py/'`；注意 bash 后台任务"completed"通知 ≠ 进程真死（孤儿进程会继续跑） |
| 5.13 | 空帧率高达 68-72% | 室内场景几乎每帧无检测 | 默认阈值 0.1 时模型分数整体压线（median 仅 0.121），阈值恰好卡掉大量候选 | 阈值降到 0.08（实测 0 空帧、检测分布正常）；参数说明见 §4.1 |
| 5.14 | cv2.LUT 断言失败 | OpenCV 报 lut.cpp Assertion failed (lutcn==cn) | OpenCV 4.x 的 cv2.LUT 不支持"单通道输入 → 3 通道输出" | 用 numpy 索引查表替代（`palette[idx_map]`），更快且无版本限制 |
| 5.15 | 高发布频率调优 | bridge 串行时发布只有 ~3Hz，bridge CPU 才 15% | 每帧桥端固定开销（大图 JPEG 编码 + 逐掩码全分辨率放大/染色 ~180ms）与引擎推理（~120ms）**串行叠加** | 已内置 4 项优化：RLE 解码向量化（np.repeat）、掩码合并成索引图一次放大 + numpy 查表染色、2 帧在飞线程池流水线、引擎空闲预测。实测 3.3Hz → 4.6-5.4Hz；剩余瓶颈是引擎单帧占用（~140-200ms），硬上限约 7Hz |

---

## 7. 回退方案

| 场景 | 回退动作 |
|---|---|
| 新掩码编码异常 | `MASK_CODEC=pil` 重启管线（engine 侧自动回退 PIL PNG；bridge 兼容新旧两种格式） |
| SAM 预处理异常 | `SAM_PREPROC=legacy` 重启管线 |
| 频率/CPU 异常 | `MAX_RATE=5` 降节流 |
| 文件级回退 | results/baseline_backup/ 保存了加速改动前的 engine/bridge/predictor 三文件 |
| 检测跑偏 | 阈值 `CONFIDENCE_THRESHOLD` 与 `MIN_BOX_AREA` 回扫（参数说明见 §4.1） |
| prompts 回退 | 两仓库 prompts 文件 + 两 bridge 的 DEFAULT_PROMPTS 三处一起改 |

---

## 8. 新机器部署 checklist

1. ☐ JetPack 版本确认（≠5.1.3 需重导 engine、换 NGC torch）
2. ☐ ROS1 noetic + cv_bridge + python3-opencv + python3-libnvinfer（apt）
3. ☐ conda env 按 §2.2（不装 cv2）
4. ☐ nanoowl/nanosam 子模块 commit 对齐（fb553de / 6536336）
5. ☐ 权重下载 + 3 个 TRT engine 导出到 data/
6. ☐ start 脚本路径/端口按机器改；LD_PRELOAD/PYTHONPATH 两行照抄
7. ☐ 相机话题名改 bridge 订阅处；prompts 文件就位
8. ☐ ROS_MASTER_URI 配置
9. ☐ `sudo jetson_clocks` 锁频（性能数字的前提）
10. ☐ §3.3 五步验证链全通

# nano_perception

Go2 使用的独立感知引擎及 ROS 桥接。支持 NanoOWL + NanoSAM，以及显式选择的 YOLOE 后端。

- 模块环境和接口：`docs/deployment_jetson_agx_orin.md`。
- 引擎：`nano_perception_engine.py`；ROS 桥：`nano_perception_bridge.py`。
- NanoOWL、NanoSAM 作为本仓库的子模块，保留各自历史。
- 整机入口、后端选择和部署环境清单由 RoboHike 顶层的 robots/go2 管理。
- data 中的模型和 results 中的运行产物不入 Git；现有资产暂留原位。

单独使用时先获取两个子模块并按部署文档准备模型和各自环境。
加入 Go2 时从 RoboHike 执行 `tools/robotctl prepare`，不要复制另一份源码。

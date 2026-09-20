# 模块工作规则

本目录只维护感知实现和模块文档。整机启动、环境选择及机器人参数归 RoboHike 顶层。
修改前阅读 docs/deployment_jetson_agx_orin.md；引擎和 ROS 桥运行在不同解释器环境。
保持 NanoOWL/NanoSAM 嵌套子模块单一归属。更新它们时先提交并发布子模块，再更新父模块。
data、results、模型及日志不入 Git。不可把未进行的 GPU 或真机测试写成通过。

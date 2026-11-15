# client-zhuji

TRON2 + Wuji Hand 2 的 XFlow 在线部署与 replay 源码快照。

## 目录

- `client/`：XFlow 推理、重定向、IK、在线执行、审计记录和测试；
- `replay/`：基础回放、ServoP 传输、阈值与测试；
- `replay_hand/`：TRON2 + Wuji 双臂双手联合回放、监督与测试；
- `replay_robot/`：轨迹转换/验证工具，以及部署和碰撞检查使用的确认版机器人模型。

`main` 分支按 zhuji 上的同级目录结构组织，便于整体查看。`xflow-client` 分支保留
`/home/guest/pengyang/client` 的原始提交历史与部署目录结构，当前部署基线为 `78b2bce`。

## 发布范围

仓库包含当前源码、单元测试、说明文档和机器人模型许可证。以下主机产物不进入 Git：

- 真机运行日志、审计运行目录和临时停止文件；
- 历史备份、patch staging、Python 缓存与 IK 缓存；
- 轨迹 NPZ/NPY、视频、zip、ROS bag 和原始相机数据。

因此 `replay_hand/data/*.npz` 等任务轨迹需要从获授权的数据源单独提供。ROS 2 工作区、
机器人 SDK、`tools/`、`Pinocchio/`、现场标定和设备配置仍由 zhuji 主机环境提供，不在本仓库
内复制。

## 安全说明

这些程序包含真实机器人发送入口。离线测试或影子模式通过不等于真机验收；执行前必须检查
现场、标定、控制器所有权、速度/碰撞门和物理急停条件。不要通过修改代码绕过 `wuji_control`
或现有运动保护。

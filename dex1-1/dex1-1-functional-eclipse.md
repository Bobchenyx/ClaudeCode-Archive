# Dex1-1 遥操作控制流程说明

## Context

用户询问：遥操作时 dex1-1 夹爪的控制走的是哪部分代码？是直接传位置信息还是有其他处理？

本文件不是修改计划，而是对当前代码的调查结论。如需基于此进行改动（如调整映射区间、更换控制方式等），请在确认结论后提出具体需求。

## 结论概要

**Dex1-1 走的是"直接位置控制"路线**——XR 输入经过线性映射后，作为电机目标角度 `q` 通过 DDS 直接下发，无 IK / 无骨架重定向。

与 dex3 完全不同：dex3 传的是 25 个关节点 × 3D 坐标（75 维 Array），需要走 `dex-retargeting` 子模块做 IK；dex1 传的只是一个标量 `Value('d')`。

## 数据通路（XR → 电机）

### 1. 输入采集（`teleop/teleop_hand_and_arm.py`）

主循环根据 `--input-mode` 选择信号源，写入共享内存的两个 `multiprocessing.Value('d')`：

- **Hand 模式**（`teleop_hand_and_arm.py:301-305`）：
  写入 `tele_data.left_hand_pinchValue` / `right_hand_pinchValue`（拇指-食指捏合距离）
- **Controller 模式**（`teleop_hand_and_arm.py:296-300`）：
  写入 `tele_data.left_ctrl_triggerValue` / `right_ctrl_triggerValue`（VR 手柄扳机模拟值）

### 2. 控制器进程（`teleop/robot_control/robot_hand_unitree.py`）

类 `Dex1_1_Gripper_Controller`（约 232-390 行），独立 `multiprocessing.Process`，以 200Hz 运行。

**变换流水线**（每个循环周期）：

1. **读共享内存**（`robot_hand_unitree.py:355-358`）——拿到一个 float
2. **线性映射**（约 362-365 行）：
   ```python
   left_target_action = np.interp(
       left_gripper_value,
       [THUMB_INDEX_DISTANCE_MIN=5.0, THUMB_INDEX_DISTANCE_MAX=7.0],
       [LEFT_MAPPED_MIN=0.0,        LEFT_MAPPED_MAX=5.40]
   )
   ```
   把 pinch 距离 `[5.0, 7.0]` 映射到电机角度 `[0.0, 5.40] rad`
3. **速率限制**（约 367-372 行）：相对当前关节状态 `dual_gripper_state[0]` 钳位到 `±DELTA_GRIPPER_CMD=0.18 rad/step`，防止突变
4. **可选滤波**（约 375-377 行）：物理机模式下走 `WeightedMovingFilter`；`--sim` 时跳过
5. **DDS 下发**（`ctrl_dual_gripper`, 约 309-316 行）：
   ```python
   self.left_gripper_msg.cmds[0].q  = dual_gripper_action[0]
   self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]
   self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
   self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
   ```
   - DDS 主题：`rt/dex1/left/cmd` 和 `rt/dex1/right/cmd`
   - 消息类型：`MotorCmds_`（单电机）
   - 控制增益固定：`kp=5.00, kd=0.05`（约 334-335 行）

### 3. 与 dex3 对比

| 维度 | Dex1 | Dex3 |
|---|---|---|
| 共享内存输入 | `Value('d')` 单标量 | `Array('d', 75)` 25 关节 × 3D |
| 转换方式 | 线性插值 + 速率限制 | dex-retargeting（IK 求解） |
| DDS 消息 | `MotorCmds_`（单电机） | `HandCmd_`（每手 7 电机） |
| 输出维度 | 每手 1 个电机角度 | 每手 7 个电机角度 |
| 频率 | 200 Hz | 100 Hz |

## 关键文件

- `teleop/teleop_hand_and_arm.py:296-305` —— 输入选择与共享内存写入
- `teleop/robot_control/robot_hand_unitree.py` `Dex1_1_Gripper_Controller` —— 控制进程主体
  - 常量：`THUMB_INDEX_DISTANCE_MIN/MAX`, `LEFT_MAPPED_MIN/MAX`, `RIGHT_MAPPED_MIN/MAX`, `DELTA_GRIPPER_CMD`
  - 方法：`ctrl_dual_gripper`（DDS 发布）

## 一句话回答

**直接传位置（电机目标角度 q）**——XR 端的 pinch 距离或扳机模拟量经过一次线性插值映射成电机弧度值，再加上速率限幅与可选 WMA 滤波，通过 DDS 主题 `rt/dex1/{left,right}/cmd` 以 `MotorCmds_` 消息下发；不经过 IK，也不做骨架重定向。

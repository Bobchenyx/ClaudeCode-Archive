# Dex1-1 遥操作控制完整调研

> 仅聚焦 dex1-1。与 dex3 的对比已剔除。

## 一、结论速览

- **控制方式**：直接位置控制——XR 输入经线性映射后，作为电机目标角度 `q` 通过 DDS 直接下发；**不走 IK，不做骨架重定向**。
- **输入源**：手柄"扳机键"（Trigger，前端食指那个模拟键）或手部"拇食指捏合距离"。两路输入都被归一到同一量纲（`~10.0 → 0.0`），共用同一套映射代码。
- **是否模拟量**：**是**。扳机有 `bool` 和 `float` 两种暴露，dex1 用的是 `float`。
- **是否有快慢/力控**：**没有**。纯位置映射，扳机扣多深就指令夹爪张多大；速度由服务端速率限幅约束（≈ 36 rad/s）。
- **重要细节**：扳机有效行程只占原始量程的 30%~50% 这 20% 的窗口，前后都是饱和死区。

---

## 二、数据通路（XR → 电机）

```
VR 手柄扳机 / 手部捏合
        │
        │ WebXR
        ▼
televuer (TeleVuerWrapper)        ← 反相 + 缩放，统一到 [10.0 → 0.0]
        │
        │ tele_data.left_ctrl_triggerValue 或 .left_hand_pinchValue
        ▼
teleop_hand_and_arm.py            ← 写入 multiprocessing.Value('d')
        │
        │ left_gripper_value / right_gripper_value（共享内存）
        ▼
Dex1_1_Gripper_Controller (子进程, 200 Hz)
        │
        │ np.interp([5.0, 7.0] → [0.0, 5.40]) + 速率限幅 + 可选 WMA 滤波
        ▼
DDS:  rt/dex1/left/cmd  /  rt/dex1/right/cmd     (MotorCmds_, 单电机)
        │
        ▼
夹爪电机（kp=5.00, kd=0.05）
```

---

## 三、各阶段细节

### 1. 输入采集（televuer 内部）

**底层 WebXR 原始信号**（`teleop/televuer/src/televuer/televuer.py:241-245`）：
```python
# trigger（底层取自 WebXR controllerState）
self.left_ctrl_trigger_shared.value      = bool(controllerState.get("trigger", False))
self.left_ctrl_triggerValue_shared.value = float(controllerState.get("triggerValue", 0.0))  # 原始 [0.0, 1.0]
```

**Wrapper 层重映射**（`teleop/televuer/src/televuer/tv_wrapper.py:417,425`）：
```python
left_ctrl_triggerValue  = 10.0 - self.tvuer.left_ctrl_triggerValue  * 10
right_ctrl_triggerValue = 10.0 - self.tvuer.right_ctrl_triggerValue * 10
```
注释（`tv_wrapper.py:169,186`）：
> `float (10.0 → 0.0) trigger pull depth, 0.0 means fully pressed (for align with hand pinch value's logic)`

设计意图：把扳机量纲做成和"拇食指距离 cm"同方向同区间，使下游一套映射代码可以同时处理 hand 和 controller 两种 `--input-mode`。

**字段一览**（`tv_wrapper.py:155-192`，仅与 dex1 相关）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `left_hand_pinch` / `right_hand_pinch` | bool | 是否在捏合 |
| `left_hand_pinchValue` / `right_hand_pinchValue` | float | 捏合距离 cm，`~15.0 → 0.0` |
| `left_ctrl_trigger` / `right_ctrl_trigger` | bool | 扳机是否被按 |
| `left_ctrl_triggerValue` / `right_ctrl_triggerValue` | float | 扳机模拟深度，`10.0 → 0.0` |

### 2. 主循环写共享内存

`teleop/teleop_hand_and_arm.py:296-305`：
```python
elif args.ee == "dex1" and args.input_mode == "controller":
    with left_gripper_value.get_lock():
        left_gripper_value.value  = tele_data.left_ctrl_triggerValue
    with right_gripper_value.get_lock():
        right_gripper_value.value = tele_data.right_ctrl_triggerValue
elif args.ee == "dex1" and args.input_mode == "hand":
    with left_gripper_value.get_lock():
        left_gripper_value.value  = tele_data.left_hand_pinchValue
    with right_gripper_value.get_lock():
        right_gripper_value.value = tele_data.right_hand_pinchValue
```

共享内存类型：`multiprocessing.Value('d')`，每路一个标量浮点数（不是 Array）。

### 3. 控制器进程（核心逻辑）

**入口**：`teleop/robot_control/robot_hand_unitree.py` 类 `Dex1_1_Gripper_Controller`（约 232-390 行），独立 `multiprocessing.Process`，**200 Hz** 运行。

**关键常量**（`robot_hand_unitree.py:321-328`）：
```python
DELTA_GRIPPER_CMD        = 0.18    # 单步速率限幅（rad/step）
THUMB_INDEX_DISTANCE_MIN = 5.0     # 输入下限（夹爪闭合点）
THUMB_INDEX_DISTANCE_MAX = 7.0     # 输入上限（夹爪张开点）
LEFT_MAPPED_MIN          = 0.0     # 电机角度下限
LEFT_MAPPED_MAX          = 5.40    # 电机角度上限（=MIN+5.40）
# 物理含义：电机转 5.4 rad → 夹爪滑开 9 cm（即 0.6 rad/cm）
```

**单周期变换**（`robot_hand_unitree.py:352-388`）：

1. 读共享内存（line 355-358） → 单 float
2. 输入零检查（line 362）：`if left_gripper_value != 0.0 or right_gripper_value != 0.0` → 仅当 wrapper 传过非零数据时才更新 target，避免初始化噪声
3. **线性插值映射**（line 364-365）：
   ```python
   left_target_action  = np.interp(left_gripper_value,
                                   [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX],
                                   [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
   ```
   `np.interp` 区间外自动钳位（不外推）：
   - 输入 ≤ 5.0 → 输出 0.0（夹爪全闭）
   - 输入 ≥ 7.0 → 输出 5.40（夹爪全开）
4. **速率限幅**（line 367-369，仅物理机模式）：
   ```python
   left_actual_action = np.clip(left_target_action,
                                dual_gripper_state[0] - 0.18,
                                dual_gripper_state[0] + 0.18)
   ```
   相对当前关节状态最多改变 ±0.18 rad/step → 36 rad/s 极速 → 全程 5.40 rad ≈ **150 ms**
5. **WMA 滤波**（line 375-377，仅物理机模式）：系数 `[0.5, 0.3, 0.2]`，再做一次低通
6. **DDS 下发**（`ctrl_dual_gripper`, line 309-316）：
   ```python
   self.left_gripper_msg.cmds[0].q = dual_gripper_action[0]
   self.left_gripper_msg.cmds[0].dq  = 0.0
   self.left_gripper_msg.cmds[0].tau = 0.0
   self.left_gripper_msg.cmds[0].kp  = 5.00
   self.left_gripper_msg.cmds[0].kd  = 0.05
   self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
   ```
   - 主题：`rt/dex1/left/cmd`、`rt/dex1/right/cmd`
   - 类型：`MotorCmds_`（单电机数组）
   - **dq=0, tau=0**：不指定速度/力矩，纯位置控制
   - 增益固定：`kp=5.00, kd=0.05`

---

## 四、扳机扣深 ↔ 夹爪对应表

> Wrapper 公式：`mapped = 10 - raw × 10`，原始扣深 `raw ∈ [0, 1]`

| 扳机扣深 raw | wrapper 输出 mapped | np.interp 后电机角度 | 夹爪结果 |
|:---:|:---:|:---:|:---|
| 0% （完全松开） | 10.0 | 5.40 | 全开 |
| 20% | 8.0 | 5.40 | 全开（饱和）|
| **30%** | 7.0 | 5.40 | 全开端点 ← 上死区终点 |
| 35% | 6.5 | 4.05 | 75% 张开 |
| 40% | 6.0 | 2.70 | 半开 |
| 45% | 5.5 | 1.35 | 25% 张开 |
| **50%** | 5.0 | 0.0 | 全闭端点 ← 下死区起点 |
| 80% | 2.0 | 0.0 | 全闭（饱和）|
| 100% （扣到底）| 0.0 | 0.0 | 全闭（饱和）|

**结论**：扳机有效控制窗口仅为原始扣深 30%~50% 这 20% 的区间，前 30% 和后 50% 都是饱和死区。

> Hand 模式同理：`pinchValue` 在 5.0~7.0 cm 之间为有效区，<5cm（捏紧）夹爪全闭，>7cm（张开）夹爪全开。

---

## 五、速度/力 控制说明

| 维度 | 是否支持 |
|---|---|
| 位置控制 | ✅ 唯一控制方式 |
| 速度控制（dq） | ❌ 恒为 0 |
| 力矩控制（tau） | ❌ 恒为 0 |
| 力反馈 | ❌ 不读取夹爪力矩 |
| 用户扳机扣得快/慢 | ❌ 无差异（只看当前位置）|

**唯一的速度上限**来自服务端速率限幅：
- `DELTA_GRIPPER_CMD = 0.18 rad/step` @ 200 Hz → **极速 36 rad/s**
- 全开↔全闭（5.40 rad）≈ **150 ms**
- 用户扳机切换若 < 150 ms，夹爪能跟上；若 > 150 ms，夹爪按用户节奏；中间不存在"力控/速度控/加速度控"语义

**仿真模式（`--sim`）的差异**：
- 跳过速率限幅（`robot_hand_unitree.py:367-372`）
- 跳过 WMA 滤波（`robot_hand_unitree.py:262-265`）
- → 仿真里夹爪可瞬间到位，物理机里有限速

---

## 六、手柄其它按键功能（仅供参考）

| 按键 | 调用位置 | 作用 |
|---|---|---|
| **左/右扳机** | `teleop_hand_and_arm.py:298,300` | **dex1 左/右夹爪开合（本文主题）** |
| 右手柄 A 键 | `teleop_hand_and_arm.py:312` | `--motion` 模式下退出遥操作 |
| 双手柄 thumbstick 同时按下 | `teleop_hand_and_arm.py:316-317` | 软急停（进 Damp 模式）|
| 左手柄 thumbstick (x,y) | `teleop_hand_and_arm.py:319-321` | 行走前后/左右（×0.3 限速）|
| 右手柄 thumbstick x | `teleop_hand_and_arm.py:321` | 自旋 yaw（×0.3 限速）|
| B/X/Y 键、Squeeze（握把） | — | wrapper 暴露但主程序未使用 |

---

## 七、数据采集（仅 dex1 视角）

### 7.1 触发方式

- **键盘 `s`**：主循环切换录制（`teleop_hand_and_arm.py:278-285`）
  - 第一次按 → `recorder.create_episode()` 创建新 episode 目录
  - 第二次按 → `recorder.save_episode()` 收尾
- **IPC**：发送 `CMD_RECORD_TOGGLE`（参考 `utils/ipc.py`）
- **采样率**：跟随主循环，由 `--frequency` 决定（默认 30 Hz）

### 7.2 目录结构（每段 episode）

```
<task-dir>/<task-name>/
└── episode_0000/
    ├── colors/                    # JPEG 彩色图，按 帧×相机 命名
    │   ├── 000000_color_0.jpg     # 头部相机左目（双目时）
    │   ├── 000000_color_1.jpg     # 头部相机右目
    │   ├── 000000_color_2.jpg     # 左腕相机（如启用 ZMQ）
    │   └── 000000_color_3.jpg     # 右腕相机（如启用 ZMQ）
    ├── depths/                    # 目录建好但 dex1 主流程不写入
    ├── audios/                    # 同上，未用
    └── data.json                  # 元信息 + 每帧状态/动作
```

`episode_writer.py:106-114` 负责创建上述目录；图像命名 `{6 位帧号}_{key}.jpg`。

### 7.3 `data.json` 顶层

```json
{
  "info": { ... },     // 元信息
  "text": { ... },     // 任务描述（可由 --task-goal/--task-desc/--task-steps 覆盖）
  "data": [ ... ]      // 每帧一条记录
}
```

**`info`**（`episode_writer.py:67-87`）：包含 `version` / `date` / `author` / `image` / `depth` / `audio` / `joint_names` / `tactile_names` / `sim_state`。
> ⚠️ `joint_names` 字段建好但**始终是空列表**，下游训练若需关节名要自行补。

**`text`**（`episode_writer.py:20-30`）：默认占位 `goal/desc/steps`，需通过 CLI 覆盖。

### 7.4 每帧字段（dex1 视角）

主循环每帧调用 `recorder.add_item(...)`（`teleop_hand_and_arm.py:471/473`），写入：

```json
{
  "idx":      <帧号>,
  "colors":   {"color_0": "colors/000000_color_0.jpg", ...},
  "depths":   {},                  // dex1 主路径未填
  "states":   { ... },             // 见下
  "actions":  { ... },             // 见下
  "tactiles": null,
  "audios":   null,
  "sim_state": <仅 --sim 模式有>   // 来自 sim_state_subscriber
}
```

### 7.5 `states` / `actions` 结构

`teleop_hand_and_arm.py:419-468`，state（当前实测）和 action（指令目标）字段对称：

```json
{
  "left_arm":  { "qpos": [7 个 float], "qvel": [], "torque": [] },
  "right_arm": { "qpos": [7 个 float], "qvel": [], "torque": [] },
  "left_ee":   { "qpos": [1 个 float], "qvel": [], "torque": [] },
  "right_ee":  { "qpos": [1 个 float], "qvel": [], "torque": [] },
  "body":      { "qpos": [...] }
}
```

| 字段 | dex1 内容 | 来源 |
|---|---|---|
| `left_arm.qpos` (state) | 左臂 7 关节当前角度 | `current_lr_arm_q[:7]` ← `arm_ctrl` 读 `rt/lowstate` |
| `left_arm.qpos` (action) | 左臂 7 关节 IK 目标 | `sol_q[:7]` ← Pinocchio + CasADi |
| `right_arm.qpos` | 同上，右臂 7 关节 | `current_lr_arm_q[-7:]` / `sol_q[-7:]` |
| `left_ee.qpos` (state) | **左夹爪当前位置 [1 元素]** | `dual_gripper_state_array[0]` |
| `left_ee.qpos` (action) | **左夹爪指令位置 [1 元素]** | `dual_gripper_action_array[0]` |
| `right_ee.qpos` | 同上，右夹爪 | `dual_gripper_state_array[1]` / `dual_gripper_action_array[1]` |
| `body.qpos` | 仅 controller 模式：state=全身 35 关节当前位置；action=三维行走指令 `[vx, vy, vyaw]` ×0.3 | `arm_ctrl.get_current_motor_q()` / 左右 thumbstick |

> 💡 **dex1 的 `left_ee.qpos` / `right_ee.qpos` 都是单元素数组**（区别于 dex3 的 7 元素、inspire 的 6 元素）。

> 💡 **存的不是扳机原始值**，而是经过 wrapper + `np.interp` + 速率限幅 + WMA 滤波之后**下发给电机的 `q`**（弧度，`0.0 → 5.40`，0=闭/5.40=开）。如果想训练扳机映射本身，需要自行采集原始 `triggerValue`。

> 💡 录制坐标系做了**减偏移归零**（`robot_hand_unitree.py:381-382`）：
> ```python
> dual_gripper_state_out[:] = dual_gripper_state - [LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN]
> ```
> 当前 `LEFT_MAPPED_MIN = RIGHT_MAPPED_MIN = 0.0`，等价于直通；若以后改偏移，记录的 qpos 自动是相对值。

### 7.6 缺省 / 不录制的字段（dex1 主路径）

| 字段 | 状态 | 说明 |
|---|---|---|
| `qvel` / `torque`（所有部位）| 全部空数组 `[]` | 主循环未赋值 |
| `tactiles` | `null` | `add_item` 调用未传 |
| `audios` | `null` | 同上 |
| `depths` | `{}` 空字典 | dex1 主路径未填 |
| `joint_names` | `{}` 空 | 元信息里就是空 |
| `sim_state` | 仅 `--sim` 模式 | 物理机不存 |

### 7.7 写盘异步性

- `add_item` 把数据塞进 `Queue(-1)`，**主循环立刻返回**（`episode_writer.py:144`）
- 后台 `worker_thread` 调用 `_process_item_data` 异步：
  - JPEG 编码并 `cv2.imwrite` 写图
  - 把 `colors[key]` 字符串路径回填，再以 JSON 增量追加到 `data.json`（`episode_writer.py:193-197`）
- `save_episode()` 只是设标志位，等队列清空后追加 `]\n}` 收尾（`episode_writer.py:212-221`）
- 同步：`rerun-sdk` 实时可视化（除非 `--headless`）

---

## 八、关键文件索引

| 文件 | 行号 | 内容 |
|---|---|---|
| `teleop/televuer/src/televuer/televuer.py` | 241-245 | WebXR 原始 trigger 取数 |
| `teleop/televuer/src/televuer/tv_wrapper.py` | 168-169 / 417 | trigger 反相+缩放到 `10.0→0.0` |
| `teleop/televuer/src/televuer/tv_wrapper.py` | 156 / 375 | pinch 同上（×100，单位 cm）|
| `teleop/teleop_hand_and_arm.py` | 296-305 | 输入源选择，写共享内存 |
| `teleop/teleop_hand_and_arm.py` | 346-363 | dex1 ee/body state/action 组装 |
| `teleop/teleop_hand_and_arm.py` | 419-473 | states/actions 字典构造 + `add_item` |
| `teleop/robot_control/robot_hand_unitree.py` | 232-390 | `Dex1_1_Gripper_Controller` 主体 |
| `teleop/robot_control/robot_hand_unitree.py` | 321-328 | 关键常量（DELTA、DISTANCE_MIN/MAX、MAPPED）|
| `teleop/robot_control/robot_hand_unitree.py` | 364-365 | 线性映射 `np.interp` |
| `teleop/robot_control/robot_hand_unitree.py` | 367-372 | 速率限幅 |
| `teleop/robot_control/robot_hand_unitree.py` | 309-316 | DDS 下发（`MotorCmds_`）|
| `teleop/robot_control/robot_hand_unitree.py` | 334-335 | `kp=5.00, kd=0.05` |
| `teleop/robot_control/robot_hand_unitree.py` | 381-382 | 共享内存输出减偏移归零 |
| `teleop/utils/episode_writer.py` | 13-233 | `EpisodeWriter` 完整实现 |
| `teleop/utils/episode_writer.py` | 67-87 | `info` 元信息字段 |
| `teleop/utils/episode_writer.py` | 106-114 | episode 目录创建 |
| `teleop/utils/episode_writer.py` | 129-144 | `add_item` 入队 |
| `teleop/utils/episode_writer.py` | 163-203 | 异步落盘流程 |

---

## 九、一句话总结

- **控制**：dex1-1 是模拟量位置控制——手柄前端"扳机键"扣多深 → wrapper 反相缩放成 `[10, 0]` → `np.interp` 映射到电机角度 `[0.0, 5.40] rad` → 经速率限幅与 WMA 滤波 → 通过 DDS `rt/dex1/{left,right}/cmd` 以 `MotorCmds_` 下发；**不走 IK，无力控/速度控**，扳机有效行程仅原始量程的 30%~50% 这 20%，夹爪极速约 36 rad/s（全程 150 ms）。
- **采集**：每帧把头/腕相机 JPEG + 双臂 7+7 关节 `qpos`（state/action 各一份）+ **双夹爪每边 1 个电机角度的 `qpos`** 写进 `data.json`；`qvel/torque/tactile/audio/depth/joint_names` 全部为空；夹爪存的是**经过完整映射链路后下发到电机的角度（0=闭，5.40=开）**，不是扳机原始值。

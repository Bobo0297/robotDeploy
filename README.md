<h1 align="center"> ASAP项目的相关deploy
</h1>

# 准备工作
## Conda环境安装
创建 conda 环境

```bash
conda create -n unitree python=3.8
conda activate unitree
```

安装 Pytorch

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

安装 unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e . && cd ..
```

安装 isaac_utils

```bash
pip install -e isaac_utils/
```

安装其余依赖

```bash
pip install -r requirements.txt
```

测试

```bash
python deploy_mujoco/sim2sim.py
```

使用上述命令, 可以在 mujoco 中完成 siuu! 的相关动作

## 训练得到策略网络

在 ASAP 项目中, 训练好相关的模型后, 使用`eval_agent.py`文件, 会在`logs/MotionTracking/XXX/exported/`目录下生成 onnx 策略文件, 该文件是使用 onnx 网络进行的策略实现, 可以实现较为高效的推理过程

# 策略部署
## 策略部署 mujoco(sim2sim)
通过以下命令将训练好的策略网络部署至 mujoco 中
```bash
python deploy_mujoco/sim2sim.py --motion_file [MOTION_FILE_PATH] --policy [POLICY_PATH]
```

参数说明:
- `--motion_file`: 加载动作文件(`*.pkl`), 可以获取到动作长度;
- `--policy`: 加载onnx策略文件, 根据当前时刻输入的`obs`矩阵, 推理得到`action`输出, 从而让机器人做出相应动作.

示例:
```bash
python deploy_mujoco/sim2sim.py --motion_file policy/bend_pick_up/bend_pick_box_up.pkl --policy_path policy/bend_pick_up/16000/model_16000_init_noise_1.0.onnx
```
运行上述示例可以得到机器人弯腰拿东西的仿真.

## 策略部署实机(sim2real)
由于时机(实机)未到, 目前可以使用 unitree_mujoco 项目, 将 deploy_real 的代码进行仿真验证, 时机成熟时可以部署到G1上.

测试
```bash
python deploy_real/simulate_python/unitree_mujoco.py
# 重新打开一个终端, 执行以下命令
python deploy_real/deploy_real.py
```

验证策略部署实机效果:

```bash
python deploy_real/simulate_python/unitree_mujoco.py
# 重新打开一个终端, 执行以下命令
python deploy_real/deploy_real.py --net [NET_INTERFACE] --motion_file [MOTION_FILE_PATH] --policy [POLICY_PATH]
```

参数说明:
- `--net`: 设置连接网口, 机器人的网口可以使用`ifconfig`命令进行查看, `lo`是指本机地址, 即`localhost`. 有关连接实体机器人的相关设置, 请见官方文档: [G1快速开发](https://support.unitree.com/home/zh/G1_developer/quick_development)
- `--motion_file`: 加载动作文件(`*.pkl`), 可以获取到动作长度;
- `--policy`: 加载onnx策略文件, 根据当前时刻输入的`obs`矩阵, 推理得到`action`输出, 从而让机器人做出相应动作.

示例:
```bash
python deploy_real/deploy_real.py --net lo --motion_file policy/bend_pick_up/bend_pick_box_up.pkl --policy_path policy/bend_pick_up/16000/model_16000_init_noise_1.0.onnx
```
运行上述示例可以得到机器人弯腰拿东西的 mujoco 验证.

**备注:**

1. 在启动 unitree_mujoco.py 文件后, 由于电机没有输入信号, 故设置了 ElasticBand 使用键盘上的7、8、9按键即可对绑带进行控制: 7-上升, 8-下降, 9-启停绑带.
2. 为在 mujoco 中部署 deploy_real.py 的代码, deploy_real.py 文件中有关手柄操作的代码已被注释, 为保证操作的安全性和便捷性, 建议对比官方 deploy_real 的代码, 增加手柄操作一项.

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from motion_lib.motion_lib_robot import MotionLibRobot

import numpy as np
import time
import torch
import onnxruntime

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from deploy_config import Sim2SimConfig, DEPLOY_ROOT

class Controller:
    def __init__(self, config: Sim2SimConfig) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        self.dt = config.sim_config.dt
        self.dof_idx = config.robot_config.dof_idx
        self.kps = config.robot_config.kps
        self.kds = config.robot_config.kds

        # Initialize the policy network
        self.ort_session = onnxruntime.InferenceSession(config.policy.policy_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        print("Load policy from: ", config.policy.policy_path)

        # Initializing process variables
        self.qj = np.zeros(config.robot_config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.robot_config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.robot_config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.robot_config.default_angles.copy()

        # history buffer
        self.history_buf = {
            "ang_vel_buf": np.zeros(3 * config.robot_config.history_length, dtype=np.float32),
            "proj_g_buf": np.zeros(3 * config.robot_config.history_length, dtype=np.float32),
            "dof_pos_buf": np.zeros(config.robot_config.num_actions * config.robot_config.history_length, dtype=np.float32),
            "dof_vel_buf": np.zeros(config.robot_config.num_actions * config.robot_config.history_length, dtype=np.float32),
            "action_buf": np.zeros(config.robot_config.num_actions * config.robot_config.history_length, dtype=np.float32),
            "ref_motion_phase_buf": np.zeros(1 * config.robot_config.history_length, dtype=np.float32)
        }

        self.obs = np.zeros(config.robot_config.num_obs, dtype=np.float32)
        self.counter = 0

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher(config.sim_config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()
        print("Publisher initialized.")

        self.lowstate_subscriber = ChannelSubscriber(config.sim_config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        print("Subscriber initialized.")

        # wait for the subscriber to receive data
        # self.wait_for_low_state()

        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.dt)
            print("Waiting for the low state...")
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")

        # print("Waiting for the start signal...")
        # while self.remote_controller.button[KeyMap.start] != 1:
        #     create_zero_cmd(self.low_cmd)
        #     self.send_cmd(self.low_cmd)
        #     time.sleep(self.config.control_dt)

        input("等待按下回车键...")

        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        time.sleep(self.dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.dt)
        
        # record the current pos
        init_dof_pos = np.zeros(29, dtype=np.float32)
        for i in range(len(self.dof_idx)):
            motor_idx = self.dof_idx[i]
            init_dof_pos[motor_idx] = self.low_state.motor_state[motor_idx].q

        for i in range(len(self.config.robot_config.other_dof_idx)):
            motor_idx = self.config.robot_config.other_dof_idx[i]
            init_dof_pos[motor_idx] = self.low_state.motor_state[motor_idx].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(len(self.dof_idx)):
                motor_idx = self.dof_idx[j]
                target_pos = self.config.robot_config.default_angles[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[motor_idx] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self.kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            for j in range(len(self.config.robot_config.other_dof_idx)):
                motor_idx = self.config.robot_config.other_dof_idx[j]
                self.low_cmd.motor_cmd[motor_idx].q = 0
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self.kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.dt)

    def launch_motion_lib(self):
        motion_lib_cfg = Sim2SimConfig.motion_config
        motion_lib = MotionLibRobot(motion_lib_cfg, num_envs=1, device="cpu")
        motion_lib.load_motions()
        self.motion_length = motion_lib.get_motion_length()  # 动作长度, 单位为秒

    def default_pos_state(self):
        print("Enter default pos state.")
        
        # print("Waiting for the Button A signal...")
        # while self.remote_controller.button[KeyMap.A] != 1:

        for i in range(len(self.config.robot_config.dof_idx)):
            motor_idx = self.config.robot_config.dof_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.robot_config.default_angles[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.kps[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].kd = self.kds[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.robot_config.other_dof_idx)):
            motor_idx = self.config.robot_config.other_dof_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = 0
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.kps[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].kd = self.kds[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(self.dt)

    def run(self):
        self.counter += 1

        if self.counter % self.config.sim_config.control_decimation == 0:
        # if 1:
            # Get the current joint position and velocity
            for i in range(len(self.dof_idx)):
                self.qj[i] = self.low_state.motor_state[self.dof_idx[i]].q
                self.dqj[i] = self.low_state.motor_state[self.dof_idx[i]].dq

            # imu_state quaternion: w, x, y, z
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

            # create observation
            action = self.action.copy()
            gravity_orientation = get_gravity_orientation(quat)
            qj_obs = self.qj.copy()
            dqj_obs = self.dqj.copy()
            qj_obs = (qj_obs - self.config.robot_config.default_angles) * self.config.normalization.dof_pos_scale
            dqj_obs = dqj_obs * self.config.normalization.dof_vel_scale
            ang_vel = ang_vel * self.config.normalization.base_ang_vel_scale
            count = self.counter * self.dt
            ref_motion_phase = count % self.motion_length / self.motion_length

            ang_vel_buf = self.history_buf["ang_vel_buf"]
            proj_g_buf = self.history_buf["proj_g_buf"]
            dof_pos_buf = self.history_buf["dof_pos_buf"]
            dof_vel_buf = self.history_buf["dof_vel_buf"]
            action_buf = self.history_buf["action_buf"]
            ref_motion_phase_buf = self.history_buf["ref_motion_phase_buf"]

            history_obs_buf = np.concatenate((action_buf, ang_vel_buf, dof_pos_buf, dof_vel_buf, proj_g_buf, ref_motion_phase_buf), axis=-1, dtype=np.float32)
            obs_buf = np.concatenate((action, ang_vel, qj_obs, dqj_obs, history_obs_buf, gravity_orientation, np.array([ref_motion_phase])), axis=-1, dtype=np.float32)

            # Update the history buffers
            num_actions = self.config.robot_config.num_actions
            ang_vel_buf = np.concatenate((ang_vel, ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
            proj_g_buf = np.concatenate((gravity_orientation, proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
            dof_pos_buf = np.concatenate((qj_obs, dof_pos_buf[:-num_actions] ), axis=-1, dtype=np.float32)
            dof_vel_buf = np.concatenate((dqj_obs, dof_vel_buf[:-num_actions] ), axis=-1, dtype=np.float32)
            action_buf = np.concatenate((action, action_buf[:-num_actions] ), axis=-1, dtype=np.float32)
            ref_motion_phase_buf = np.concatenate((np.array([ref_motion_phase]), ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)

            self.history_buf = {
                "ang_vel_buf": ang_vel_buf,
                "proj_g_buf": proj_g_buf,
                "dof_pos_buf": dof_pos_buf,
                "dof_vel_buf": dof_vel_buf,
                "action_buf": action_buf,
                "ref_motion_phase_buf": ref_motion_phase_buf
            }

            obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
            action = np.squeeze(self.ort_session.run(None, {self.input_name: obs_tensor})[0])
            self.action = np.clip(action, -self.config.normalization.clip_actions, self.config.normalization.clip_actions)
            
            # transform action to target_dof_pos
            target_dof_pos = self.config.robot_config.default_angles + self.action * self.config.normalization.action_scale

            # Build low cmd
            for i in range(len(self.config.robot_config.dof_idx)):
                motor_idx = self.config.robot_config.dof_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self.kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.robot_config.other_dof_idx)):
                motor_idx = self.config.robot_config.other_dof_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = 0
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self.kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="部署策略至G1机器人")
    parser.add_argument("--net", type=str, help="网口名称, 默认为lo", default="lo")
    parser.add_argument('--motion_file', type=str, help="动作文件", default=os.path.join(DEPLOY_ROOT, "policy/siuu/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl"))
    parser.add_argument('--policy', type=str, help="策略文件", default=os.path.join(DEPLOY_ROOT, "policy/siuu/model_26000.onnx"))
    args = parser.parse_args()
    config = Sim2SimConfig()
    config.motion_config.motion_file = args.motion_file
    config.policy.policy_path = args.policy

    # Initialize DDS communication
    if args.net == "lo" or args.net == "":
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    while True:
        try:
            controller.default_pos_state()
        except KeyboardInterrupt:
            break
    
    print("Enter the run state.")

    controller.launch_motion_lib()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            # if controller.remote_controller.button[KeyMap.select] == 1:
            #     break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")

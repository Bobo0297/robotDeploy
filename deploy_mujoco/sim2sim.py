import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import mujoco.viewer
import mujoco
import numpy as np
import argparse

import torch
import onnxruntime
from motion_lib.motion_lib_robot import MotionLibRobot
from sim2sim_config import Sim2SimConfig, DEPLOY_ROOT, ElasticBand
from loguru import logger

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    q, v = torch.from_numpy(q), torch.from_numpy(v)
    shape = q.shape
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.mm(q_vec.view(1, 3), v.view(3, 1)).squeeze(-1) * 2.0
    return (a - b + c).cpu().numpy().squeeze()
    
SINGLE = False

def run_mujoco(config: Sim2SimConfig):
    # 初始化参数
    control_decimation = config.sim_config.control_decimation
    # control_start_time = config.sim_config.control_start_time
    control_start_time = 0.0
    simulation_dt = config.sim_config.dt
    simulation_duration = config.sim_config.sim_duration
    num_actions = config.env.num_actions
    action = np.zeros(num_actions, dtype=np.float32)
    default_angles = config.robot_config.default_angles
    init_angles = config.robot_config.bend_pick_init_angles
    kps = config.robot_config.kps
    kds = config.robot_config.kds
    tau_limit = config.robot_config.tau_limit
    model_path = config.sim_config.model_path
    policy_path = config.policy.policy_path
    history_length = config.env.history_length
    clip_actions = config.normalization.clip_actions
    clip_observations = config.normalization.clip_observations
    dof_pos_limit = config.robot_config.dof_pos_limit

    dof_pos_scale = config.normalization.dof_pos_scale
    dof_vel_scale = config.normalization.dof_vel_scale
    base_ang_vel_scale = config.normalization.base_ang_vel_scale
    base_lin_vel_scale = config.normalization.base_lin_vel_scale
    action_scale = config.normalization.action_scale

    target_dof_pos = default_angles.copy()
    # target_dof_pos = init_angles.copy() + default_angles.copy()
    ref_motion_phase = 0

    # obs_buf = np.zeros(76, dtype=np.float32)

    # history buffer
    lin_vel_buf = np.zeros(3 * history_length, dtype=np.float32)
    ang_vel_buf = np.zeros(3 * history_length, dtype=np.float32)
    proj_g_buf = np.zeros(3 * history_length, dtype=np.float32)
    dof_pos_buf = np.zeros(num_actions * history_length, dtype=np.float32)
    dof_vel_buf = np.zeros(num_actions * history_length, dtype=np.float32)
    action_buf = np.zeros(num_actions * history_length, dtype=np.float32)
    ref_motion_phase_buf = np.zeros(1 * history_length, dtype=np.float32)

    # 加载动作
    motion_lib_cfg = Sim2SimConfig.motion_config
    motion_lib = MotionLibRobot(motion_lib_cfg, num_envs=1, device="cpu")
    motion_lib.load_motions()
    motion_length = motion_lib.get_motion_length()  # 动作长度, 单位为秒
    # motion_length = 3.933

    # 加载mujoco模型
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 加载onnx模型
    ort_session = onnxruntime.InferenceSession(policy_path)
    input_name = ort_session.get_inputs()[0].name

    elastic_band = ElasticBand()
    elastic_band.enable = False
    band_attached_link = m.body("torso_link").id

    # d.qpos[7:7 + len(target_dof_pos)] = default_angles + init_angles
    d.qpos[7:7 + len(target_dof_pos)] = default_angles # 设置关节位置
    d.qvel[6:6 + len(target_dof_pos)] = [0] # 设置关节速度为 0


    counter = 0
    with mujoco.viewer.launch_passive(m, d, key_callback=elastic_band.MujuocoKeyCallback) as viewer:
        input("Press Enter to start the simulation...")
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            tau = np.clip(tau, -tau_limit, tau_limit)
            d.ctrl[:] = tau
            if elastic_band.enable:
                d.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(d.qpos[:3], d.qvel[:3])
            mujoco.mj_step(m, d)

            counter += 1

            # 5s后开始控制
            # if counter * simulation_dt > control_start_time:
            if counter % control_decimation == 0:
            # if 0:
                # Apply control signal here.

                # 获取观测值
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]       # w,x,y,z
                lin_vel = d.qvel[:3]
                # lin_vel = quat_rotate_inverse(quat, lin_vel)
                ang_vel = d.qvel[3:6]
                # ang_vel = d.sensor('imu-angular-velocity').data.astype(np.float32)
                # ang_vel = quat_rotate_inverse(quat, ang_vel)
                # print("ang_vel", ang_vel)
                projected_gravity = get_gravity_orientation(quat)
                # projected_gravity = quat_rotate_inverse(quat, np.array([0.,0.,-1.]))
                # print("projected_gravity", projected_gravity)
                ref_motion_phase = (counter * simulation_dt - control_start_time) % motion_length / motion_length
                print("time:", counter * simulation_dt - control_start_time)
                print("ref_motion_phase", ref_motion_phase)

                # 归一化观测值
                dof_pos = (qj - default_angles) * dof_pos_scale
                # dof_pos = qj * dof_pos_scale
                dof_vel = dqj * dof_vel_scale
                base_ang_vel = ang_vel * base_ang_vel_scale
                base_lin_vel = lin_vel * base_lin_vel_scale
                # action = action * action_scale

                """_summary_
                    curr_obs:
                            base_lin_vel         3
                            base_ang_vel         3
                            projected_gravity    3
                            dof_pos              23
                            dof_vel              23
                            actions              23
                            ref_motion_phase     1
                """

                '''
                存在问题, history_actor的顺序究竟为何, 存储的时候, 难道就是按照顺序的吗?
                已解决, history_actor的顺序就是每个buf按照时间顺序进行存储的, 整体的顺序为：
                action[t-2:t-5]
                base_ang_vel[t-1:t-4]
                dof_pos[t-1:t-4]
                dof_vel[t-1:t-4]
                projected_gravity[t-1:t-4]
                ref_motion_phase[t-1:t-4]
                '''

                # 构造观测值
                if SINGLE:
                    obs_buf = np.concatenate((action_buf, base_ang_vel, base_lin_vel, dof_pos, dof_vel, projected_gravity, np.array([ref_motion_phase])), axis=-1, dtype=np.float32)
                    obs_buf = np.clip(obs_buf, -clip_observations, clip_observations)
                else:
                    # 3 history frames without liner velocity
                    history_obs_buf = np.concatenate((action_buf, ang_vel_buf, dof_pos_buf, dof_vel_buf, proj_g_buf, ref_motion_phase_buf), axis=-1, dtype=np.float32)
                    # history_obs_buf = np.zeros_like(history_obs_buf)
                    obs_buf = np.concatenate((action, base_ang_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, np.array([ref_motion_phase])), axis=-1, dtype=np.float32)
                    # obs_buf = np.clip(obs_buf, -clip_observations, clip_observations)

                # 更新history buffer
                ang_vel_buf = np.concatenate((base_ang_vel, ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
                lin_vel_buf = np.concatenate((base_lin_vel, lin_vel_buf[:-3]), axis=-1, dtype=np.float32)
                proj_g_buf = np.concatenate((projected_gravity, proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
                dof_pos_buf = np.concatenate((dof_pos, dof_pos_buf[:-num_actions] ), axis=-1, dtype=np.float32)
                dof_vel_buf = np.concatenate((dof_vel, dof_vel_buf[:-num_actions] ), axis=-1, dtype=np.float32)
                action_buf = np.concatenate((action, action_buf[:-num_actions] ), axis=-1, dtype=np.float32)
                ref_motion_phase_buf = np.concatenate((np.array([ref_motion_phase]), ref_motion_phase_buf[:-1] ), axis=-1, dtype=np.float32)

                # 获取动作
                obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
                action = np.squeeze(ort_session.run(None, {input_name: obs_tensor})[0])
                action = np.clip(action, -clip_actions, clip_actions)

                # 更新目标关节位置
                target_dof_pos = action * action_scale + default_angles
                # target_dof_pos = np.clip(target_dof_pos, -dof_pos_limit, dof_pos_limit)
                
            # 同步viewer
            viewer.sync()

            # 时间同步
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run Mujoco simulation with Sim2Sim configuration.")
    argparser.add_argument('--motion_file', type=str, default=os.path.join(DEPLOY_ROOT, "policy/siuu/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl"))
    argparser.add_argument('--policy', type=str, default=os.path.join(DEPLOY_ROOT, "policy/siuu/model_26000.onnx"))
    logger.info(f"Deploy root: {DEPLOY_ROOT}")

    config = Sim2SimConfig()
    config.motion_config.motion_file = argparser.parse_args().motion_file
    config.policy.policy = argparser.parse_args().policy
    logger.info(f"Loaded policy from {os.path.join(DEPLOY_ROOT, config.policy.policy_path)}")

    run_mujoco(config)

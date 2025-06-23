import os
import numpy as np
import mujoco
from easydict import EasyDict

DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Sim2SimConfig():
    class sim_config:
        model_path = os.path.join(DEPLOY_ROOT, "resources/robots/g1/g1_29dof_anneal_23dof.xml")
        sim_duration = 60
        dt = 0.002
        control_decimation = 10
        control_start_time = 5.0

    class robot_config:
        kps = np.array([# 双腿kps
            100, 100, 100, 200, 20, 20,
            100, 100, 100, 200, 20, 20,
            # waist及双臂kps
            400, 400, 400,
            90, 60, 20, 60,
            90, 60, 20, 60], dtype=np.float32)
        
        kds = np.array([# 双腿kds
            2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
            2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
            # waist及双臂kds
            5.0, 5.0, 5.0,
            2.0, 1.0, 0.4, 1.0,
            2.0, 1.0, 0.4, 1.0], dtype=np.float32)
        
        tau_limit = np.array([# 双腿
            88, 88, 88, 139, 50, 50, 
            88, 88, 88, 139, 50, 50,
            # waist及双臂
            88, 50, 50, 
            25, 25, 25, 25, 
            25, 25, 25, 25], dtype=np.float32)
        
        default_angles = np.array([# 双腿
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
            # waist及双臂
            0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        init_angles = np.array([# 双腿
             0.2544, -0.0083,  0.1539, -0.0031, -0.1407, -0.0344,
            -0.2696, -0.0288, -0.0889,  0.0851,  0.2500, -0.0796,
            -0.2375, -0.0128,  0.2247, 
             0.0883,  0.0647,  0.1817,  0.0846,
             0.3386,  0.0190, -0.2373,  0.4618], dtype=np.float32)
        
        bend_pick_init_angles = np.array([# 双腿
            -0.1237, 0.2061, 0.1797, 0.2564, 0.0000, 0.0000, -0.2738, -0.3335, -0.3657, 0.5707, 0.0000, 0.0000, 0.0132, 0.0910, 0.4282, -0.2061, 0.1226, 0.2970, 1.1025, -0.2313, -0.2205, -0.3539, 0.9527], dtype=np.float32)
        
        jump_init_angles = np.array([ 0.0472,  0.0777,  0.2031,  0.1678,  0.0000,  0.0000, -0.0475,  0.0427,
          0.2061,  0.3153,  0.0000,  0.0000,  0.0302,  0.0129,  0.2066, -0.0366,
          0.0732,  0.2884,  1.2232,  0.0000,  0.0000,  0.0000,  0.0333], dtype=np.float32)
        
        dof_pos_limit = np.array([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
                              2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
                              2.618,  0.52,   0.52, 
                              2.6704, 2.2515, 2.618, 2.0944, 
                              2.6704, 2.2515, 2.618, 2.0944], dtype=np.float32)
        
    class env:
        num_actions = 23
        history_length = 4
        num_obs_without_history = 76

    class normalization:
        clip_actions = 100.
        clip_observations = 100.

        action_scale = 0.25
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05
        base_ang_vel_scale = 0.25
        base_lin_vel_scale = 2.0

    class policy:
        policy_path = os.path.join(DEPLOY_ROOT, "policy/siuu/model_26000.onnx")

    class motion_config:
        motion_file = os.path.join(DEPLOY_ROOT, "policy/siuu/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl")

        asset_file = os.path.join(DEPLOY_ROOT, "resources/robots/g1/g1_29dof_anneal_23dof_fitmotionONLY.xml")
        extend_config = [
            EasyDict(
                joint_name="left_hand_link",
                parent_name="left_elbow_link",
                pos=[0.25, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0]
            ),
            EasyDict(
                joint_name="right_hand_link",
                parent_name="right_elbow_link",
                pos=[0.25, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0]
            ),
            EasyDict(
                joint_name="head_link",
                parent_name="torso_link",
                pos=[0.0, 0.0, 0.42],
                rot=[1.0, 0.0, 0.0, 0.0]
            ),
        ]

class ElasticBand: 
    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
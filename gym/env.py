from random import random, seed
import time
import pybullet as p
import pybullet_data
from datetime import datetime
import numpy as np
from numpy import float32, inf
from gym.spaces import Box


seed(datetime.now())
TARGET_LOC = np.array([0, 0, 0.28])
PI = 3.14159
HIP_LIMIT = 2*0.785398
HIP_CENTER = 0.785398
HIP_ROT_LIMIT = PI
KNEE_LIMIT = PI
ACTION_MULT = np.array([
    HIP_LIMIT, HIP_ROT_LIMIT, KNEE_LIMIT,
    HIP_LIMIT, HIP_ROT_LIMIT, KNEE_LIMIT])
ACTION_MOVE = np.array([
    HIP_CENTER, 0, 0, HIP_CENTER, 0, 0])

def map_actions(actions):
    return np.array(actions) * ACTION_MULT - ACTION_MOVE


class Env:
    def __init__(self, name, var=0.1, vis=False):
        self.observation_space = Box(
            (9, ),
            np.array(
                [inf, inf, inf, *[2*PI for _ in range(6)]],
                dtype=float32),
            np.array(
                [-inf, -inf, -inf, *[0 for _ in range(6)]],
                dtype=float32)
        )
        self.action_space = Box(
            (6, ),
            np.array(
                [1, 1, 1, 1, 1, 1],
                dtype=float32),
            np.array(
                [0, 0, 0, 0, 0, 0],
                dtype=float32)
            )

        self.var = var
        self.vis = vis
        self.name = name
        self.client = p.connect(p.GUI) if vis else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

    def reset(self):
        self.plane_id = None
        self.robot_id = None
        for body_id in range(p.getNumBodies()):
            p.removeBody(body_id)

        slope = p.getQuaternionFromEuler([
            self.var*random() - self.var/2,
            self.var*random() - self.var/2,
            0])

        self.plane_id = p.loadURDF(
            "plane.urdf",
            [0, 0, -13],
            slope)

        robot_start_pos = [0, 0, 1]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            "gym/urdf/robot.urdf",
            robot_start_pos,
            robot_start_orientation)

        p.setGravity(0, 0, -10)
        return self._get_state()


    def take_action(self, actions):
        actions = map_actions(actions)
        for joint_i, action in enumerate(actions):
            maxForce = 500
            p.setJointMotorControl2(self.robot_id, joint_i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=action,
                                    force=maxForce)

    def step(self, actions):
        self.take_action(actions)
        p.stepSimulation()
        if self.vis:
            pass
            # time.sleep(1./240.)
        return self._get_state(), self._get_reward(), False, None

    def _get_state(self):
        base_loc = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        return np.array([*base_loc, *[p.getJointState(self.robot_id, i)[0]
                for i in range(p.getNumJoints(self.robot_id))]])

    def _get_reward(self):
        base_loc = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        return np.linalg.norm(base_loc - TARGET_LOC)

    def close(self):
        p.disconnect()

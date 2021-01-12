from random import random, seed
import time
import pybullet as p
import pybullet_data
from datetime import datetime
import numpy as np

seed(datetime.now())
TARGET_LOC = np.array([0, 0, 0.28])


HIP_LIMIT = 2*0.785398
HIP_CENTER = 0.785398

HIP_ROT_LIMIT = 3.14159
KNEE_LIMIT = 3.14159


ACTION_MULT = np.array([
    HIP_LIMIT, HIP_ROT_LIMIT, KNEE_LIMIT,
    HIP_LIMIT, HIP_ROT_LIMIT, KNEE_LIMIT])

ACTION_MOVE = np.array([
    HIP_CENTER, 0, 0, HIP_CENTER, 0, 0])


def map_actions(actions):
    return np.array(actions) * ACTION_MULT - ACTION_MOVE


class Gym:
    def __init__(self, name, var=0.1, vis=False):
        self.var = var
        self.vis = vis
        self.name = name
        self.client = p.connect(p.GUI) if vis else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

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
            time.sleep(1./240.)
        return self._get_state(), self._get_reward()

    # def action_space(self):
    #     return p.getNumJoints(self.robot_id)

    def _get_state(self):
        base_loc = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        return [*base_loc, *[p.getJointState(self.robot_id, i)[0]
                for i in range(p.getNumJoints(self.robot_id))]]

    def _get_reward(self):
        base_loc = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        return np.linalg.norm(base_loc - TARGET_LOC)

    def close(self):
        p.disconnect()

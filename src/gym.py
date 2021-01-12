import time
import pybullet as p
import pybullet_data


class gym:
    def __init__(self, name, vis=False):
        self.vis = vis
        self.name = name
        self.client = p.connect(p.GUI) if vis else p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        cubeStartPos = [0, 0, 1]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -15])
        p.setGravity(0, 0, -10)
        self.robot_id = p.loadURDF(
            "urdf/robot.urdf",
            cubeStartPos,
            cubeStartOrientation,
            # useFixedBase=True
        )

    def step(self, actions):
        p.stepSimulation()
        if self.vis:
            time.sleep(1./240.)
        return self._get_state()

    def _get_state(self):
        return []

    def close(self):
        p.disconnect()

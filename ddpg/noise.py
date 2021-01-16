
import numpy as np
from scipy.interpolate import interp1d


class NormalNoise:
    def __init__(
            self,
            dim,
            sigma=0.2,
            dt=1e-2):
        self.dim = dim
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def __call__(self):
        return self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.dim)

    def reset(self):
        return


class OUNoise:
    """Ornstein-Uhlenbeck process.

    Taken from https://keras.io/examples/rl/ddpg_pendulum/
    Formula from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
    """
    def __init__(
            self,
            dim=1,
            sigma=0.15,
            theta=0.2,
            dt=1e-2,
            x_initial=None):
        self.theta = theta
        self.dim = dim
        self.sigma = sigma
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (- self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt)
            * np.random.normal(size=self.dim)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros(self.dim)


class LinearSegmentNoise:
    """LinearSegmentNoise

    Generates noise values by generating random events in the form of point
    disconitinuitys and interpolating linearly between them. If event_prob is
    set to one this is just normal noise. The lower it is and the smoother the
    noise becomes.
    """
    def __init__(
            self,
            dim=1,
            sigma=0.2,
            event_prob=0.1,
            dt=1e-2):
        self.dim = dim
        self.sigma = sigma
        self.event_prob = event_prob
        self.dt = dt
        self.prev_event = np.zeros(self.dim)
        self.current = np.zeros(self.dim)
        self.next_event = self.get_next_event()

    def get_next_event(self):
        return self.sigma * np.sqrt(self.dt) * \
            np.random.normal(np.zeros(self.dim))

    def __call__(self):
        if np.random.uniform(0, 1, 1) < self.event_prob:
            self.prev_event = self.next_event
            self.next_event = self.get_next_event()
        self.current = self.current + self.dt * \
            (self.next_event - self.prev_event)
        return self.current

    def reset(self):
        self.prev_event = np.zeros(self.dim)
        self.next_event = self.get_next_event()
        self.current = np.zeros(self.dim)


class SmoothNoise1D:
    """1 Dimensional SmoothSegmentNoise

    Generates noise values by interpolating between randomly sampled points
    along the orbit.
    """
    def __init__(
            self,
            steps=200,
            sigma=0.2,
            num_interp_points=10,
            dt=1e-2):
        self.sigma = sigma
        self.steps = steps
        self.dt = dt
        self.num_interp_points = num_interp_points
        self.orb = np.linspace(0, steps, num=steps+1, endpoint=True)
        self.points_x = None
        self.points_y = None
        self.step_ind = None
        self.f = None
        self.setup()

    def setup(self):
        self.points_x = np.array(
            [0, *np.random.choice(
                    self.orb[1:-1],
                    size=self.num_interp_points,
                    replace=False), self.steps+1])
        self.points_y = self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=(len(self.points_x)))
        self.step_ind = 0
        self.f = interp1d(
            self.points_x,
            self.points_y,
            kind='cubic',
            fill_value="extrapolate")

    def __call__(self):
        self.step_ind += 1
        return self.f(self.orb[self.step_ind])

    def reset(self):
        self.setup()


class SmoothNoiseND:
    """N Dimensional SmoothSegmentNoise

    Generates multi Dimensional noise values by interpolating between randomly
    sampled points along the orbit.
    """
    def __init__(
            self,
            dim=2,
            steps=200,
            sigma=0.2,
            num_interp_points=10,
            dt=1e-2):
        self.dim = dim
        self.generator = [
            SmoothNoise1D(steps, sigma, num_interp_points, dt)
            for _ in range(self.dim)]

    def __call__(self):
        return [g() for g in self.generator]

    def reset(self):
        return [g.reset() for g in self.generator]

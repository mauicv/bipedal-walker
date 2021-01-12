from gym import Gym
from gym import Connection
import click
import numpy as np


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))


@cli.command()
def run_control():
    env = Gym('bipedal', var=0, vis=True)
    with Connection() as conn:
        for i in range(10000):
            pos_state, _ = env.step([])
            conn.send(pos_state)


@cli.command()
def run_test():
    env = Gym('bipedal', var=0.1, vis=True)
    print(env.action_space)
    print(env.action_space.high)
    print(env.action_space.low)
    for i in range(10000):
        random_action = np.random.normal(0, 0.1, size=(6))
        state, reward = env.step(random_action)


if __name__ == "__main__":
    cli()

from src.gym import gym
from src.control import Connection
import click


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))


@cli.command()
def run_control():
    env = gym('arm', vis=True)
    with Connection() as conn:
        for i in range(10000):
            state = env.step([0, 0, 0])
            conn.send(state)


@cli.command()
def run_test():
    env = gym('arm', vis=True)
    for i in range(10000):
        env.step([0, 0, 0])


if __name__ == "__main__":
    cli()

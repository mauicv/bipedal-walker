"""Agent class

Responsible for correctly loading and interfaceing with the models.
"""

import tensorflow as tf
from ddpg.model import build_models


class Agent:
    def __init__(
            self,
            state_space_dim,
            action_space_dim,
            low_action,
            high_action,
            noise_process,
            layer_dims=[32, 19],
            load=False,
            tau=0.05):
        self.noise_process = noise_process
        self.low_action = low_action
        self.high_action = high_action
        self.tau = tau
        self.actor, self.critic = (None, None)
        if not load or not self.load_models():
            print("Creating new models")
            self.actor, self.critic = \
                build_models(
                    state_space_dim,
                    action_space_dim,
                    layer_dims=layer_dims,
                    upper_bound=high_action)
        assert((self.actor, self.critic) != (None, None))

        self.target_actor, self.target_critic = \
            build_models(
                state_space_dim,
                action_space_dim,
                layer_dims=layer_dims,
                upper_bound=high_action)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def load_models(self):
        try:
            self.critic = tf.keras.models.load_model('./save/critic')
            self.actor = tf.keras.models.load_model('./save/actor')
            return True
        except Exception as err:
            print(err)

    def save_models(self):
        self.actor.save('./save/actor')
        self.critic.save('./save/critic')

    def track_weights(self):
        self._track_model_weights(
            self.target_actor.variables,
            self.actor.variables)
        self._track_model_weights(
            self.target_critic.variables,
            self.critic.variables)

    @tf.function
    def _track_model_weights(self, target_weights, weights):
        for target_weight, weight in zip(target_weights, weights):
            target_weight.assign(
                weight * self.tau + target_weight * (1 - self.tau))

    @tf.function
    def get_action(self, state, with_exploration=False):
        action = self.actor(state)
        # *self.high_action
        if with_exploration:
            action = action + self.noise_process()
            action = tf.clip_by_value(action,
                                      clip_value_min=self.low_action,
                                      clip_value_max=self.high_action)
        return action

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.distributions import Beta
from battery import Battery

import numpy as np

# ==============================================================================
#    THIS CLASS WILL BE IMPLEMENTED BY COMPETITORS
# ==============================================================================
class ActorCriticNetwork(object):
  def __init__(self, trainer=None, scope='global', augment_size=41,
               forecast_size=96, forecast_dim=4, rnn_a_size=24, fc_size=16):
    self.inputs = tf.placeholder(shape=[None, forecast_size, forecast_dim], dtype=tf.float32)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_a_size, state_is_tuple=True)
    c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
    h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
    self.state_init = [c_init, h_init]
    c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
    h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
    self.state_in = (c_in, h_in)
    state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
    lstm_output, _ = tf.nn.dynamic_rnn(lstm_cell, self.inputs, initial_state=state_in)
    rnn_out = tf.reshape(lstm_output[:,-1,:], [-1, rnn_a_size])
    self.curr_state = tf.placeholder(shape=[None, augment_size], dtype=tf.float32)
    augmented_input = tf.concat([rnn_out, self.curr_state], axis=1)

    policy_dense = slim.fully_connected(augmented_input, fc_size,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.glorot_normal_initializer)
    self.policy_alpha = tf.exp(slim.fully_connected(policy_dense, 1,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.glorot_normal_initializer))
    self.policy_beta = tf.exp(slim.fully_connected(policy_dense, 1,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.glorot_normal_initializer))

    self.policy = (self.policy_alpha - 1) / (self.policy_alpha + self.policy_beta - 2)
    value_dense = slim.fully_connected(augmented_input, fc_size,
                                       activation_fn=tf.nn.relu,
                                       weights_initializer=tf.glorot_normal_initializer)

    self.value = slim.fully_connected(value_dense, 1,
                                      activation_fn=None,
                                      weights_initializer=tf.glorot_normal_initializer)

    self.actions = tf.placeholder(shape=[None], dtype=tf.float32)
    self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
    self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
    
    self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - self.value))

    self.policy_loss = - tf.reduce_sum(tf.log(self.policy) * self.advantages) # TODO what is the policy loss supposed to be?
    self.entropy = Beta(self.policy_alpha, self.policy_beta).entropy()
    self.loss = 0.5 * self.value_loss + self.policy_loss + 0.001 * self.entropy

    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    self.gradients = tf.gradients(self.loss, local_vars)
    self.var_norms = tf.global_norm(local_vars)
    grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    if trainer is not None:
      self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
    else:
      self.apply_grads = None


class BatteryContoller(object):
  """ The BatteryContoller class handles providing a new "target state of charge"
      at each time step.

      This class is instantiated by the simulation script, and it can
      be used to store any state that is needed for the call to
      propose_state_of_charge that happens in the simulation.

      The propose_state_of_charge method returns the state of
      charge between 0.0 and 1.0 to be attained at the end of the coming
      quarter, i.e., at time t+15 minutes.

      The arguments to propose_state_of_charge are as follows:
      :param site_id: The current site (building) id in case the model does different work per site
      :param timestamp: The current timestamp inlcuding time of day and date
      :param battery: The battery (see battery.py for useful properties, including current_charge and capacity)
      :param actual_previous_load: The actual load of the previous quarter.
      :param actual_previous_pv_production: The actual PV production of the previous quarter.
      :param price_buy: The price at which electricity can be bought from the grid for the
        next 96 quarters (i.e., an array of 96 values).
      :param price_sell: The price at which electricity can be sold to the grid for the
        next 96 quarters (i.e., an array of 96 values).
      :param load_forecast: The forecast of the load (consumption) established at time t for the next 96
        quarters (i.e., an array of 96 values).
      :param pv_forecast: The forecast of the PV production established at time t for the next
        96 quarters (i.e., an array of 96 values).

      :returns: proposed state of charge, a float between 0 (empty) and 1 (full).
  """
  def __init__(self):
    self.network = self.initialize_network()
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def initialize_network(self):
    return ActorCriticNetwork()

  def propose_state_of_charge(self,
                              site_id,
                              timestamp,
                              battery,
                              actual_previous_load,
                              actual_previous_pv_production,
                              price_buy,
                              price_sell,
                              load_forecast,
                              pv_forecast):

    # return the proposed state of charge ...
    forecast = self.make_forecast_array(price_buy, price_sell, load_forecast, pv_forecast, battery)
    out_inputs = self.make_other_inputs(battery, timestamp, actual_previous_load, actual_previous_pv_production)

    return self.get_action(forecast, out_inputs)

  def get_action(self, forecast, state):
    policy_output = self.session.run(self.network.policy,
                                     feed_dict={self.network.inputs:forecast,
                                                self.network.curr_state:state})
    return policy_output

  def make_forecast_array(self, price_buy, price_sell, load_forecast, pv_forecast,
                          battery: Battery):
    forecast_size = len(price_buy)
    forecast_array = np.zeros((1, forecast_size, 4))
    forecast_array[0,:,0] = price_buy - price_sell # premium
    forecast_array[0,:,1] = price_buy + price_sell # spread
    forecast_array[0,:,2] = load_forecast * (15/60) / (battery.capacity * battery.discharging_efficiency)
    forecast_array[0,:,3] = pv_forecast * (15/60)  / battery.capacity * battery.discharging_efficiency # not a typo, efficiency is treated differently for charge and discharge
    return forecast_array

  def make_other_inputs(self, battery: Battery, timestamp, actual_previous_load, actual_previous_pv_production):
    input_array = np.zeros((1, 5 + 12 + 24))
    input_array[0, 0] = battery.current_charge
    input_array[0, 1] = battery.charging_power_limit / battery.capacity
    input_array[0, 2] = battery.discharging_power_limit / battery.capacity
    input_array[0, 3] = actual_previous_load
    input_array[0, 4] = actual_previous_pv_production
    input_array[0, 5 + timestamp.month] = 1.0
    input_array[0, 17 + timestamp.hour] = 1.0
    return input_array


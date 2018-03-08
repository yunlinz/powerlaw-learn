from battery_controller import BatteryContoller
from battery_controller import ActorCriticNetwork

from battery import Battery

from time import sleep

import os
import tensorflow as tf

import multiprocessing
import threading
# ==============================================================================
#    THIS CLASS WILL BE IMPLEMENTED BY COMPETITORS
# ==============================================================================
class BatteryContollerTrainable(BatteryContoller):
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
    pass

  def initialize_network(self):
    self.network = ActorCriticNetwork(tf.train.AdamOptimizer())
    self.session = tf.Session()


  def get_feedback(self, reward):
    pass


"""For A3C implementation

max_episode_length = 300
gamma = 0.99

load_model = False
model_path = './assets'

if not os.path.exists(model_path):
  os.makedirs(model_path)

with tf.device('/cpu:0'):
  global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
  trainer = tf.train.AdamOptimizer()
  master_network = ActorCriticNetwork('global', trainer)
  num_workers = multiprocessing.cpu_count()
  workers = [BatteryContollerTrainable() for _ in range(num_workers)]
  saver = tf.train.Saver(max_to_keep=5)

with tf.Session as sess:
  coord = tf.train.Coordinator()
  if load_model:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    sess.run(tf.global_variables_initializer())

  worker_threads = []
  for worker in workers:
    work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
    t = threading.Thread(target=(work))
    t.start()
    sleep(0.5)
    worker_threads.append(t)
  coord.join(worker_threads)
"""
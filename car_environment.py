'''
standard for reinforcement learning environment is to have some sort of .step() method in which you pass an action

# either at the very beginning of the environment or after we returned another done flag if want to run another episode
def reset(self):

def step(self, action):
    # when done you return the next observation, the reward, whether or not you are done (flag) (thus success or
    # running out of time or whether you died, and then extra info
    return obs, reward, done, extra_info
'''

import glob
import math
import os
import sys

try:
    # finding carla egg file
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ======================================c========================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import random
import time
import numpy as np
import cv2
import keras.backend.tensorflow_backend as backend
import tensorflow as tf
from threading import Thread

from DQN_agent import DQNAgent
from tqdm import tqdm

# whether or not we want to display the camera as it will take resources
SHOW_PREVIEW = False
IMG_WIDTH = 640
IMG_HEIGHT = 480

SECONDS_PER_EPISODE = 15

MEMORY_FRACTION = 0.8
EPISODES = 100

# epsilon needed to make the agent learn, the higher it is, the more likely the agent will on random moves,
# and in time this epsilon will decrease to enable the agent to learn and make more appropriate moves
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

MODEL_NAME = "Xception"

MIN_REWARD = -200

AGGREGATE_STATS_EVERY = 10


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    # full turns -> need to change this if we want sth more precise
    STEER_AMT = 1.0

    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    actor_list = []

    front_camera = None
    collision_hist = []

    def __init__(self):
        # connect to server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        self.blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        # print(blueprint_library.filter('vehicle'))
        self.model_3 = self.blueprint_library.filter('model3')[0]

    # method either at the very beginning of the environment or after we have returned a done flag, if we want to run
    # another episode so to speak
    def reset(self):
        # if any collision is detected, it's considered as fail (collision can sometimes only be sth like going uphill
        # really fast which kind of makes the car bump
        self.collision_hist = []
        # collect actors so we can clean them up at the end
        self.actor_list = []

        # spawn point
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        # vehicle in world defined by model and spawn location
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        # specify spawn point of camera
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        # sensor in world defined by type of sensor, location (relative to vehicle)
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        # do something with this sensor, in this case listen for its imagery
        self.sensor.listen(lambda data: self.process_img(data))

        # this 2 lines need improvement as it takes a lot of time
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # sleep to get things started (sensors can take time to initialize and return values) and
        # to not detect a collision when the car spawns/falls from sky. which happens quite often
        time.sleep(4)

        colsensor = self.blueprint_library.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # make sure to wait until all is setup
        while self.front_camera is None:
            time.sleep(0.01)

        # episode length
        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera

    def collision_data(self, event):
        actor_we_collide_against = event.other_actor
        print(actor_we_collide_against)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        print(intensity)
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)  # convert to an array
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4)) # was flattened, so we're going to shape it.
        i3 = i2[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
        if self.SHOW_CAM:
            cv2.imshow("", i3)  # show it.
            cv2.waitKey(1)
        self.front_camera = i3  # normalize

    # methods takes an action, does something with that action and then returns the next observation, reward, done
    # (flag, true or false -> telling the environment if we successfully finished or that we died or ran out of time,
    # any_extra_info as for the usual learning paradigm
    def step(self, action):
        """
        For now let's just pass steer left, center, right?
        0, 1, 2
        """
        # make turns smoother
        steer_amount = [0.2*i for i in range(1, 6)]
        if action == 0:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT * random.choice(steer_amount)))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        if action == 2:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT * random.choice(steer_amount)))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # episode is done when collision is detected or time is completed

        # for now, if any collision is detected, we stop
        # TODO develop collision event detection to filter down collision actors
        # TODO add invasion sensor to detect when vehicle trespasses lane lines
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


if __name__ == '__main__':
    FPS = 60
    # For stats
    # starting with high epsilon -> high probability that we will be choosing an action randomly,
    # rather than predicting it with our neural network. A random choice is going to be much faster than a predict
    # operation, so we can arbitrarily delay this by setting some sort of general FPS
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when training multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Iterate over episodes
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # try:

        env.collision_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, 3)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1 / FPS)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon -> makes agent act on less random moves
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
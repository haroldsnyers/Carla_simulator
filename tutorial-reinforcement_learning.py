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

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import random
import time
import numpy as np
import cv2

# whether or not we want to display the camera as it will take resources
SHOW_PREVIEW = False
IMG_WIDTH = 640
IMG_HEIGHT = 480

SECONDS_PER_EPISODE = 10

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    # full turns
    STEER_AMT = 1.0

    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    actor_list = []

    front_camera = None
    collision_hist = []

    def __init__(self):
        # connect to server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        self.blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        #print(blueprint_library.filter('vehicle'))
        self.model_3 = self.blueprint_library.filter('model3')[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        # spawn point
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library().find('sensor.camera.rgb')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        # specify spawn point of camera
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        # do something with this sensor, in this case listen for its imagery
        self.sensor.listen(lambda data: self.process_img(data))

        # this 2 lines need improvement as it takes a lot of time
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # sleep to get things started (sensors can take time to initialize and return values) and
        # to not detect a collision when the car spawns/falls from sky. which happens quite often
        time.sleep(4)

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
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

    # methods takes an action and then returns an observation, reward, done, any_extra_info as for the usual
    # learning paradigm
    def step(self, action):
        '''
        For now let's just pass steer left, center, right?
        0, 1, 2
        '''
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # for now, if any collision is detected, we stop
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
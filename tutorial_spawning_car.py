import glob
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

IM_WIDTH = 640
IM_HEIGHT = 480

def process_img(image):
    i = np.array(image.raw_data)  # convert to an array
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # was flattened, so we're going to shape it.
    i3 = i2[:, :, :3]  # remove the alpha (basically, remove the 4th index  of every pixel. Converting RGBA to RGB)
    cv2.imshow("", i3)  # show it.
    cv2.waitKey(1)
    return i3/255.0  # normalize


actor_list = []

try:
    # Carla server must be running when this script is ran
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    # environment
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # blueprints are the attributes of actors
    bp = blueprint_library.filter("model3")[0]
    print(bp)

    # 200 spawn positions possible, randomly choosing between them
    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)

    # vehicle.set_autopilot(True)

    # Control car : make vehicle just move forward
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    actor_list.append(vehicle)

    # docs sensors : https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    # fov : field of view
    blueprint.set_attribute('fov', '110')

    # adjust sensor to a relative position and then attach this to our car
    # Adjust sensor relative to vehicle (thus moving the sensor relative to the vehicle 2.5 forward and 0.7 up
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # do something with this sensor, in this case listen for its imagery
    sensor.listen(lambda data: process_img(data))

    time.sleep(15.0)


finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")


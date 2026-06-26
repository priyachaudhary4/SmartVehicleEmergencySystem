import carla
import random
import time

print("Connecting to CARLA...")

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()

blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.filter('model3')[0]

spawn_points = world.get_map().get_spawn_points()

spawn_point = random.choice(spawn_points)

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle:

    print("Tesla Spawned!")

    # Drive forward
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.5,
            steer=0.0
        )
    )

    print("Driving Forward...")

    time.sleep(5)

    # Brake
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            brake=1.0
        )
    )

    print("Vehicle Stopped!")

else:

    print("Spawn Failed")
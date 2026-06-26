import carla
import random
import cv2
import numpy as np


# Function to process camera images
def process_image(image):
    img = np.array(image.raw_data)

    # Convert raw data into image
    img = img.reshape((image.height, image.width, 4))

    # Remove alpha channel
    img = img[:, :, :3]

    # Display image
    cv2.imshow("Tesla Camera", img)
    cv2.waitKey(1)


# -----------------------------
# Connect to CARLA
# -----------------------------
print("Connecting to CARLA...")

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()

print("Connected to CARLA!")


# -----------------------------
# Get Blueprint Library
# -----------------------------
blueprint_library = world.get_blueprint_library()


# -----------------------------
# Spawn Tesla Model 3
# -----------------------------
vehicle_bp = blueprint_library.filter('model3')[0]

spawn_points = world.get_map().get_spawn_points()

spawn_point = random.choice(spawn_points)

vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is None:
    print("❌ Failed to spawn Tesla.")
    exit()

print("✅ Tesla spawned successfully!")


# -----------------------------
# Create RGB Camera
# -----------------------------
camera_bp = blueprint_library.find('sensor.camera.rgb')

camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

print("Camera Blueprint Ready!")


# -----------------------------
# Camera Position
# -----------------------------
camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle
)

print("✅ Camera Attached!")


# -----------------------------
# Start Camera Streaming
# -----------------------------
camera.listen(lambda image: process_image(image))

print("Streaming Camera Feed...")
print("Press CTRL + C to stop.")


# -----------------------------
# Main Loop
# -----------------------------
try:
    while True:
        world.wait_for_tick()

except KeyboardInterrupt:
    print("\nStopping Simulation...")

finally:
    print("Cleaning Up...")

    camera.stop()

    camera.destroy()
    vehicle.destroy()

    cv2.destroyAllWindows()

    print("Simulation Closed Successfully!")
import carla
import time
import math

# ==========================================
# SMART VEHICLE CONTROLLER
# ==========================================

class VehicleController:
    def __init__(self, vehicle, world):
        """
        Initializes the Vehicle Controller.
        :param vehicle: The CARLA vehicle actor object.
        :param world: The CARLA world object.
        """
        self.vehicle = vehicle
        self.world = world
        self.emergency_mode_active = False

    def get_speed(self):
        """
        Calculates the current speed of the vehicle in km/h.
        """
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed

    def normal_drive(self, tm_port=8000):
        """
        Simulates normal autonomous driving (maintaining a steady speed).
        """
        if not self.emergency_mode_active:
            # Enable CARLA's built-in autopilot so it navigates the city naturally
            self.vehicle.set_autopilot(True, tm_port)

    def trigger_hazard_lights(self):
        """
        Turns on the vehicle's hazard lights.
        """
        self.vehicle.set_light_state(carla.VehicleLightState.All)

    def trigger_emergency_stop(self):
        """
        Executes a safe emergency stop maneuver.
        Reduces speed smoothly, steers slightly to the right for a roadside stop, and applies hazards.
        """
        if self.emergency_mode_active:
            return  # Already in emergency mode
            
        print("[VEHICLE CONTROL] Emergency Stop Initiated!")
        self.emergency_mode_active = True
        self.vehicle.set_autopilot(False)
        self.trigger_hazard_lights()

        current_speed = self.get_speed()
        
        # 1. Gradual speed reduction
        while current_speed > 2.0:
            current_speed = self.get_speed()
            
            # Progressive braking logic matching storyboard (80 -> 60 -> 40 -> 20 -> 0)
            if current_speed > 60.0:
                brake_amt = 0.2
            elif current_speed > 40.0:
                brake_amt = 0.4
            elif current_speed > 20.0:
                brake_amt = 0.6
            elif current_speed > 5.0:
                brake_amt = 0.8
            else:
                brake_amt = 1.0
                
            print(f"[VEHICLE CONTROL] Speed: {int(current_speed)} km/h | Brake: {int(brake_amt*100)}%")
                
            # Apply progressive braking and keep steering straight (lane keep)
            control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,  
                brake=brake_amt
            )
            self.vehicle.apply_control(control)
            time.sleep(0.5)

        # 2. Hard stop once speed is low
        print("[VEHICLE CONTROL] Finalizing Stop...")
        control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True
        )
        self.vehicle.apply_control(control)
        print("[VEHICLE CONTROL] Vehicle safely stopped on the roadside.")

    def reset_control(self):
        """
        Resets the vehicle back to normal operation (e.g. driver wakes up).
        """
        self.emergency_mode_active = False
        self.vehicle.set_light_state(carla.VehicleLightState.NONE)
        print("[VEHICLE CONTROL] Resumed normal control.")

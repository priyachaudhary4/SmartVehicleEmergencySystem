import time
import threading
import pyttsx3
import winsound

# ==========================================
# EMERGENCY MANAGER (Decision Engine)
# ==========================================

class EmergencyManager:
    def __init__(self, vehicle_controller, notification_system):
        """
        Initializes the Emergency Decision Engine.
        :param vehicle_controller: Instance of VehicleController.
        :param notification_system: Instance of NotificationSystem.
        """
        self.vehicle_controller = vehicle_controller
        self.notification_system = notification_system
        self.emergency_handled = False
        
        # Initialize text to speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        except:
            self.tts_engine = None

    def evaluate_driver_state(self, driver_state):
        """
        Evaluates the current driver state and decides the emergency response.
        :param driver_state: The state string (e.g., NORMAL, DROWSY, UNRESPONSIVE)
        """
        if driver_state == "UNRESPONSIVE" and not self.emergency_handled:
            print("[DECISION ENGINE] CRITICAL: Driver Unresponsive Detected!")
            self.execute_emergency_protocol()
            
        elif driver_state == "NORMAL" and self.emergency_handled:
            # Optionally reset if driver wakes up before final stop
            pass

    def execute_emergency_protocol(self):
        """
        Executes the full emergency protocol in a separate thread to avoid blocking the main loop.
        """
        self.emergency_handled = True
        
        def play_siren():
            for _ in range(5):
                winsound.Beep(1000, 500)
                winsound.Beep(700, 500)
        
        def protocol_sequence():
            from future.dashboard import vehicle_state
            vehicle_state["emergency_active"] = True
            
            # 0. Voice Alert & Siren
            if self.tts_engine:
                self.tts_engine.say("Driver not responding. Emergency mode activated.")
                self.tts_engine.runAndWait()
            
            threading.Thread(target=play_siren, daemon=True).start()

            # 1. Take over vehicle control
            print("[DECISION ENGINE] Taking over vehicle control...")
            self.vehicle_controller.trigger_emergency_stop()
            
            # 2. Trigger notifications (Call ambulance, send GPS, broadcast profile)
            print("[DECISION ENGINE] Vehicle stopped. Dispatching emergency alerts...")
            self.notification_system.dispatch_all_alerts()
            
            vehicle_state["hospital_alert"] = "Sent"
            vehicle_state["police_alert"] = "Sent"
            vehicle_state["guardian_alert"] = "Sent"
            vehicle_state["v2v_alert"] = "Sent"
            
            # Spawn Emergency Responder (Ambulance)
            world = self.vehicle_controller.world
            ego_vehicle = self.vehicle_controller.vehicle
            
            if ego_vehicle and world:
                print("[SCENARIO] Spawning Ambulance Persona...")
                import carla
                transform = ego_vehicle.get_transform()
                forward_vector = transform.get_forward_vector()
                # Spawn 12 meters behind the ego vehicle
                spawn_location = transform.location - forward_vector * 12
                # Ensure it's slightly above ground to prevent collision with map
                spawn_location.z += 1.0 
                spawn_transform = carla.Transform(spawn_location, transform.rotation)
                
                blueprint_library = world.get_blueprint_library()
                ambulance_bps = blueprint_library.filter('vehicle.ford.ambulance')
                if ambulance_bps:
                    ambulance_bp = ambulance_bps[0]
                    ambulance = world.try_spawn_actor(ambulance_bp, spawn_transform)
                    if ambulance:
                        print("[SCENARIO] Ambulance arrived on scene!")
                        # Turn on flashing emergency lights
                        ambulance.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2 | carla.VehicleLightState.Position))
                    else:
                        print("[SCENARIO] Failed to spawn Ambulance (collision at spawn point).")
            
            print("[DECISION ENGINE] Emergency Protocol Complete. Waiting for responders.")

        # Run the protocol in a daemon thread so it doesn't freeze the camera feed
        thread = threading.Thread(target=protocol_sequence)
        thread.daemon = True
        thread.start()

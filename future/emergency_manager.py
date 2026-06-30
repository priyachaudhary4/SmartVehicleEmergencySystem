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
        
        # Initialize text to speech engine flag (initialized per thread for Windows COM compatibility)
        self.tts_enabled = True

    def evaluate_driver_state(self, driver_state):
        """
        Evaluates the current driver state and decides the emergency response.
        :param driver_state: The state string (e.g., NORMAL, DROWSY, UNRESPONSIVE)
        """
        # Only take over control for truly critical conditions (eyes closed > 4 sec -> UNRESPONSIVE)
        critical_states = ["UNRESPONSIVE", "FAINTED", "DROWSY"]
        
        if driver_state in critical_states and not self.emergency_handled:
            print(f"[DECISION ENGINE] CRITICAL: Driver {driver_state} Detected! AI taking over control.")
            
            if self.tts_enabled:
                def speak_critical():
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.say(f"Warning. Driver {driver_state} detected. AI taking over vehicle control to ensure safety.")
                        engine.runAndWait()
                    except Exception as e:
                        print("TTS Error:", e)
                threading.Thread(target=speak_critical, daemon=True).start()
                
            self.execute_emergency_protocol()
            
        elif "WARNING" in driver_state or driver_state == "DISTRACTED":
            if not getattr(self, "warning_played", False):
                self.warning_played = True
                print(f"[DECISION ENGINE] {driver_state} Detected! Playing audio alert.")
                if self.tts_enabled:
                    def speak_warning():
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 150)
                            engine.say("Emergency. Emergency. Alert. Alert.")
                            engine.runAndWait()
                        except Exception as e:
                            print("TTS Error:", e)
                    threading.Thread(target=speak_warning, daemon=True).start()

        elif driver_state == "NORMAL" or driver_state == "ATTENTIVE":
            self.drowsy_warned = False
            self.warning_played = False
            
            if self.emergency_handled:
                print("[DECISION ENGINE] Driver is ALERT again. Reverting emergency mode...")
                self.emergency_handled = False
                self.vehicle_controller.reset_control()
                self.vehicle_controller.normal_drive()
                from future.dashboard import vehicle_state
                vehicle_state["emergency_active"] = False
                
                if self.tts_enabled:
                    def speak_resume():
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 150)
                            engine.say("Driver alert. Resuming normal operations.")
                            engine.runAndWait()
                        except Exception as e:
                            print("TTS Error:", e)
                    threading.Thread(target=speak_resume, daemon=True).start()

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
            if self.tts_enabled:
                def speak_protocol():
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.say("Driver not responding. Emergency mode activated.")
                        engine.runAndWait()
                    except Exception as e:
                        print("TTS Error:", e)
                threading.Thread(target=speak_protocol, daemon=True).start()
            
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
            
            # V2V Response System Simulation
            self.simulate_v2v_response()
            
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

    def simulate_v2v_response(self):
        """
        Simulates nearby vehicles receiving V2V communication, stopping, and people coming out to help.
        """
        import carla
        import math
        world = self.vehicle_controller.world
        ego_vehicle = self.vehicle_controller.vehicle
        
        if not ego_vehicle or not world:
            return
            
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        
        # 1. Identify nearby vehicles
        all_actors = world.get_actors()
        vehicles = all_actors.filter('vehicle.*')
        
        nearby_vehicles = []
        for v in vehicles:
            if v.id == ego_vehicle.id:
                continue
            v_loc = v.get_transform().location
            dist = math.sqrt((v_loc.x - ego_location.x)**2 + (v_loc.y - ego_location.y)**2 + (v_loc.z - ego_location.z)**2)
            if dist < 50.0:  # Within 50 meters
                nearby_vehicles.append((dist, v))
                
        # Sort by distance
        nearby_vehicles.sort(key=lambda x: x[0])
        
        # Pick the closest 1-2 vehicles
        assisting_vehicles = nearby_vehicles[:2]
        
        blueprint_library = world.get_blueprint_library()
        pedestrian_bps = blueprint_library.filter('walker.pedestrian.*')
        
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        
        if not assisting_vehicles:
            print("[V2V COMM] No nearby vehicles available to assist. Spawning virtual crowd...")
            # Spawn 2-3 pedestrians randomly around the ego vehicle to act as a virtual crowd
            if pedestrian_bps:
                import random
                num_pedestrians = random.randint(2, 3)
                for i in range(num_pedestrians):
                    ped_bp = random.choice(pedestrian_bps)
                    if ped_bp.has_attribute('is_invincible'):
                        ped_bp.set_attribute('is_invincible', 'true')
                        
                    # Calculate spawn location (randomly around ego vehicle, 10-15 meters away)
                    angle = random.uniform(0, 2 * math.pi)
                    distance = random.uniform(10.0, 15.0)
                    spawn_loc = carla.Location(
                        x=ego_location.x + distance * math.cos(angle),
                        y=ego_location.y + distance * math.sin(angle),
                        z=ego_location.z + 1.0
                    )
                    spawn_transform = carla.Transform(spawn_loc, carla.Rotation())
                    pedestrian = world.try_spawn_actor(ped_bp, spawn_transform)
                    
                    if pedestrian:
                        print(f"[VIRTUAL CROWD] Bystander {pedestrian.id} stepping in to help!")
                        if walker_controller_bp:
                            controller = world.spawn_actor(walker_controller_bp, carla.Transform(), pedestrian)
                            if controller:
                                controller.start()
                                controller.go_to_location(ego_location)
                                controller.set_max_speed(1.5)
            return
            
        for dist, v in assisting_vehicles:
            print(f"[V2V COMM] Vehicle {v.type_id} (ID: {v.id}) received distress signal. Coming to help...")
            
            def guide_and_assist(assist_veh):
                try:
                    import time
                    assist_veh.set_autopilot(False)
                    
                    # Navigate towards ego vehicle
                    for _ in range(150): # Max 15 seconds
                        if not ego_vehicle.is_alive or not assist_veh.is_alive:
                            break
                            
                        e_loc = ego_vehicle.get_location()
                        v_t = assist_veh.get_transform()
                        v_loc = v_t.location
                        
                        d = math.sqrt((v_loc.x - e_loc.x)**2 + (v_loc.y - e_loc.y)**2)
                        if d < 12.0:
                            break # Reached close enough
                            
                        # Simple proportional steering
                        dx = e_loc.x - v_loc.x
                        dy = e_loc.y - v_loc.y
                        target_yaw = math.degrees(math.atan2(dy, dx))
                        current_yaw = v_t.rotation.yaw
                        
                        diff = (target_yaw - current_yaw + 180) % 360 - 180
                        steer = max(-1.0, min(1.0, diff / 40.0))
                        
                        throttle = 0.6 if d > 20.0 else 0.35
                        assist_veh.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0))
                        time.sleep(0.1)
                        
                    # Stop vehicle safely near ego vehicle
                    assist_veh.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
                    time.sleep(1.0) # Wait for complete stop
                    
                    # Spawn pedestrian
                    v_transform = assist_veh.get_transform()
                    if pedestrian_bps:
                        import random
                        ped_bp = random.choice(pedestrian_bps)
                        if ped_bp.has_attribute('is_invincible'):
                            ped_bp.set_attribute('is_invincible', 'true')
                            
                        right_vector = v_transform.get_right_vector()
                        spawn_loc = v_transform.location + right_vector * 2.5
                        spawn_loc.z += 1.0 # Lift slightly to avoid collision
                        
                        spawn_transform = carla.Transform(spawn_loc, v_transform.rotation)
                        pedestrian = world.try_spawn_actor(ped_bp, spawn_transform)
                        
                        if pedestrian:
                            print(f"[V2V COMM] Pedestrian from vehicle {assist_veh.id} stepping out to assist!")
                            if walker_controller_bp:
                                controller = world.spawn_actor(walker_controller_bp, carla.Transform(), pedestrian)
                                if controller:
                                    controller.start()
                                    controller.go_to_location(ego_vehicle.get_location())
                                    controller.set_max_speed(2.5) # Run towards the emergency!
                except Exception as e:
                    print(f"Error guiding vehicle: {e}")

            import threading
            t = threading.Thread(target=guide_and_assist, args=(v,))
            t.daemon = True
            t.start()

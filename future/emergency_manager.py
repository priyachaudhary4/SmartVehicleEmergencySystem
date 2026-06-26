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
            
            print("[DECISION ENGINE] Emergency Protocol Complete. Waiting for responders.")

        # Run the protocol in a daemon thread so it doesn't freeze the camera feed
        thread = threading.Thread(target=protocol_sequence)
        thread.daemon = True
        thread.start()

import time

# ==========================================
# NOTIFICATION & COMMUNICATION SYSTEM
# ==========================================

class NotificationSystem:
    def __init__(self, driver_profile):
        """
        Initializes the Emergency Notification System.
        :param driver_profile: Dictionary containing medical and contact info.
        """
        self.driver_profile = driver_profile
        
    def get_simulated_gps(self):
        """
        Returns simulated GPS coordinates of the vehicle.
        In a real scenario, this would come from a GPS hardware module.
        """
        return {"latitude": 37.7749, "longitude": -122.4194, "location": "Highway 101, Marker 42"}

    def notify_emergency_contacts(self):
        print("\n" + "="*50)
        print("[NOTIFICATION] 📡 SENDING SMS TO EMERGENCY CONTACTS...")
        time.sleep(1)
        for contact in self.driver_profile.get("emergency_contacts", []):
            print(f" -> Sent Alert to {contact['name']} at {contact['phone']}")
        print("="*50 + "\n")

    def notify_hospital_and_police(self, gps_coords):
        print("\n" + "="*50)
        print("[NOTIFICATION] 🚑 DISPATCHING EMS & POLICE...")
        time.sleep(1.5)
        print(f" -> Location Sent: {gps_coords['latitude']}, {gps_coords['longitude']} ({gps_coords['location']})")
        print(" -> Transmitting Medical Profile:")
        print(f"    - Blood Type: {self.driver_profile.get('blood_type', 'Unknown')}")
        print(f"    - Allergies: {', '.join(self.driver_profile.get('allergies', []))}")
        print(f"    - Medical Conditions: {', '.join(self.driver_profile.get('medical_conditions', []))}")
        print("="*50 + "\n")

    def alert_nearby_vehicles(self):
        print("\n" + "="*50)
        print("[V2V COMM] ⚠️ BROADCASTING V2X WARNING TO NEARBY CONNECTED VEHICLES...")
        time.sleep(1)
        print(" -> Message: 'HAZARD AHEAD. VEHICLE STOPPING DUE TO MEDICAL EMERGENCY.'")
        print("="*50 + "\n")

    def display_onboard_medical_screen(self):
        print("\n" + "="*50)
        print("[ONBOARD DISPLAY] 🖥️ SWITCHING INFOTAINMENT TO MEDICAL EMERGENCY MODE")
        print(" -> Displaying 'DRIVER UNCONSCIOUS - DO NOT MOVE WITHOUT EMS'")
        print(f" -> Displaying Blood Type: {self.driver_profile.get('blood_type', 'Unknown')}")
        print("="*50 + "\n")

    def dispatch_all_alerts(self):
        """
        Executes the full chain of emergency communications.
        """
        gps = self.get_simulated_gps()
        self.alert_nearby_vehicles()
        self.notify_emergency_contacts()
        self.notify_hospital_and_police(gps)
        self.display_onboard_medical_screen()
        print("[NOTIFICATION] ALL EMERGENCY PROTOCOLS DISPATCHED SUCCESSFULLY.")

# Simple test block
if __name__ == "__main__":
    test_profile = {
        "blood_type": "O+",
        "allergies": ["Penicillin", "Peanuts"],
        "medical_conditions": ["Type 1 Diabetes"],
        "emergency_contacts": [
            {"name": "Wife", "phone": "+1-555-0192"},
            {"name": "Brother", "phone": "+1-555-0293"}
        ]
    }
    notifier = NotificationSystem(test_profile)
    notifier.dispatch_all_alerts()

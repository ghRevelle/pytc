from plane import Plane
import numpy as np
import shapely
from flightsim import FlightSimulator
from slot_manager import FixedSlotPlaneManager
from airport import Runway, Airport
from commands import *

test_runways = {
	'Runway1': Runway((0, 0), (0.05, 0.05)),
	'Runway2': Runway((0.05, 0), (0, 0.05)),
}

test_airport = Airport(test_runways)

def create_plane(init_state, slot_manager):
    return Plane(init_state, slot_manager)

fs = FlightSimulator(display_size=(900, 900), airport = test_airport, slot_manager = FixedSlotPlaneManager())

for i in range(5):
	# Create 5 planes with random positions and speeds
	fs.add_plane(create_plane(
		{
			'callsign': f"UA{i+1}",
			'lat': np.random.uniform(-0.01, 0.01),  # random latitude
			'lon': np.random.uniform(-0.01, 0.01),  # random longitude
			'alt': np.random.uniform(0, 12000),  # random altitude in meters
			'v_z': np.random.uniform(-10, 10),  # random vertical speed in meters per second
			'gspd': np.random.uniform(60, 100),  # random ground speed in meters per second
			'hdg': np.random.uniform(0, 360)  # random heading in degrees
		},
        fs.slot_manager
	))
fs.add_plane(create_plane(
	{
		'callsign': 'UA6',
		'lat': 0.0,
		'lon': -0.1,
		'alt': 1000,
		'v_z': 0,
		'gspd': 100,
		'hdg': 45
	},
    fs.slot_manager
))


# sample command to turn a plane
"""
fs.add_command(Command(
    command_type=CommandType.TURN,
    target_id=0,                # ID slot assigned to 'UA1'
	last_update=100             # when to issue command
    argument=90,                # heading to turn to    
))
"""

# Alternatively, use add_command_by_callsign, easier for testing
fs.add_command_by_callsign('UA6', CommandType.TURN, last_update=100, argument=90)

print(fs.slot_manager.show_ids())
print(fs.slot_manager.get_id('UA6'))

fs.run(ticks=500)  # Run the simulation for 500 ticks
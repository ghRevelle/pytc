from plane import Plane
import numpy as np
import shapely
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from airport import Runway, Airport
from commands import *

test_runways = {
	'Runway1': Runway((0, 0), (0.05, 0.05)),
	'Runway2': Runway((0.05, 0), (0, 0.05)),
}

test_airport = Airport(test_runways)

fs = FlightSimulator(display_size=(900, 900), airport = test_airport, plane_manager = PlaneManager())

for i in range(5):
	# Create 5 planes with random positions and speeds
	fs.add_plane_to_manager(
		{
			'callsign': f"UA{i+1}",
			'lat': np.random.uniform(-0.01, 0.01),  # random latitude
			'lon': np.random.uniform(-0.01, 0.01),  # random longitude
			'alt': np.random.uniform(0, 12000),  # random altitude in meters
			'v_z': np.random.uniform(-10, 10),  # random vertical speed in meters per second
			'gspd': np.random.uniform(60, 100),  # random ground speed in meters per second
			'hdg': np.random.uniform(0, 360)  # random heading in degrees
		},
	)
fs.add_plane_to_manager(
	{
		'callsign': 'UA6',
		'lat': 0.0,
		'lon': -0.1,
		'alt': 1000,
		'v_z': 0,
		'gspd': 100,
		'hdg': 45
	},
)


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

for tick in range(500):
	if tick == 150:
		fs.delete_plane_from_manager(callsign='UA1')
	fs.next_tick()

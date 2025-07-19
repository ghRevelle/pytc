import numpy as np
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
			'v_z': np.random.uniform(-5, 5),  # random vertical speed in meters per second
			'gspd': np.random.uniform(32.1014075233, 83.8546382418),  # random ground speed in meters per second
			'hdg': np.random.uniform(0, 360), # random heading in degrees
			'acc_xy': np.random.uniform(0, 3.0)  # random horizontal acceleration in m/s^2
		},
	)
fs.add_plane_to_manager(
	{
		'callsign': 'UA6',
		'lat': 0.0,
		'lon': -0.1,
		'alt': 1000,
		'v_z': 0,
		'gspd': 83.8546382418,
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

ua6 = fs.plane_manager.planes[-1]
runway = test_runways['Runway2']
calculated_time = ua6.find_turn_initiation_time(runway.get_line_xy(),0)
print(f"Plane {ua6.callsign} with ID {ua6.id} is set to turn to heading {runway.hdg} degrees at {calculated_time} seconds.")

fs.add_command_by_callsign('UA6', CommandType.TURN, last_update=calculated_time, argument=runway.hdg)
fs.run(ticks=500)  # Run the simulation for 500 ticks
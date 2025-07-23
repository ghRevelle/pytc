import numpy as np
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from airport import Runway, Airport
from commands import *
from planestates import PlaneState

test_runways = {
	'Runway1': Runway((0, 0), (0.05, 0.05)),
	'Runway2': Runway((0.05, 0.05), (0, 0)), # Runway 1 reversed
	'Runway3': Runway((0.05, 0), (0, 0.05)),
	'Runway4': Runway((0, 0.05), (0.05, 0))  # Runway 2 reversed
}

test_airport = Airport(test_runways)

fs_airport = test_airport

fs = FlightSimulator(display_size=(900, 900), airport = fs_airport, plane_manager = PlaneManager())

for i in range(5):
	# Create 5 planes with random positions and speeds
	fs.add_plane_to_manager(
		{
			'callsign': f"UA{i+1}",
			'lat': np.random.uniform(-0.01, 0.01),  # random latitude
			'lon': np.random.uniform(-0.01, 0.01),  # random longitude
			'alt': np.random.uniform(304.8, 607.6),  # random altitude in meters corresponding to between 1000 and 2000 feet
			'v_z': np.random.uniform(-5, 5),  # random vertical speed in meters per second
			'gspd': np.random.uniform(24.1789448, 83.8546382418),  # random ground speed in meters per second corresponding to between 62.4 and 163 kts
			'hdg': np.random.uniform(0, 360), # random heading in degrees
			'acc_xy': np.random.uniform(0, 3.0),  # random horizontal acceleration in m/s^2
			'state' : PlaneState.AIR # plane's initial state
		},
	)
fs.add_plane_to_manager(
	{
		'callsign': 'UA6',
		'lat': 0.0,
		'lon': -0.1,
		'alt': 400,
		'v_z': 0,
		'gspd': 83.8546382418,
		'hdg': 45,
		'state' : PlaneState.AIR
	},
)

fs.add_plane_to_manager(
	{
		'callsign' : 'RG1',
		'lat' : 0.0,
		'lon' : 0.0,
		'alt' : 0,
		'v_z' : 0,
		'gspd' : 0,
		'hdg' : 0,
		'state' : PlaneState.GROUND
	}
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

ua6 = fs.plane_manager.planes[-2]
runway = test_runways['Runway2']

fs_airport.add_to_queue(fs.plane_manager.planes[-1].id, runway=1)
print(fs_airport.get_top_of_queue())

#fs.add_command_by_callsign('UA6', CommandType.LINE_UP_AND_WAIT, last_update=0, argument=runway)
fs.add_command_by_callsign('UA6', CommandType.GO_AROUND, last_update=300, argument=None)
fs.add_command_by_callsign('UA6', CommandType.REALIGN, last_update=1100, argument=runway)
fs.add_command(Command(command_type=CommandType.LINE_UP_AND_WAIT, target_id=fs_airport.pop_top_of_queue(), last_update=100, argument=test_runways['Runway3']))
fs.add_command_by_callsign('RG1', CommandType.CLEARED_FOR_TAKEOFF, last_update=150, argument=test_runways['Runway3'])
fs.run(ticks=2500)  # Run the simulation for 500 ticks

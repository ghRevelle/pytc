import numpy as np
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from airport import Runway, Airport
from commands import *
from planestates import PlaneState

# Use coordinates relative to 44.04882° N / 103.06126° W (Rapid City, SD area)
# Create runways with realistic coordinates for this location
test_runways = {
	'Runway14': Runway((44.05383, -103.06500), (44.03283, -103.04933), "Runway14"),  # NE-SW runway
	'Runway32': Runway((44.03283, -103.04933), (44.05383, -103.06500), "Runway32"),  # SW-NE runway
	'Runway05': Runway((44.04783, -103.06383), (44.05250, -103.05167), "Runway05"),  # NW-SE runway
	'Runway23': Runway((44.05250, -103.05167), (44.04783, -103.06383), "Runway23"),  # SE-NW runway
}

test_airport = Airport(test_runways)

fs = FlightSimulator(display_size=(900, 900), airport = test_airport, plane_manager = PlaneManager())

# Base coordinates around Rapid City area
base_lat = 44.04882
base_lon = -103.06126

for i in range(5):
	# Create 5 planes with random positions and speeds
	fs.add_plane_to_manager(
		{
			'callsign': f"UA{i+1}",
			'lat': base_lat + np.random.uniform(-0.01, 0.01),  # random latitude around base
			'lon': base_lon + np.random.uniform(-0.01, 0.01),  # random longitude around base
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
		'lat': base_lat + 0.2,  # North of the airport
		'lon': base_lon + 0.05,  # West of the airport
		'alt': 400,
		'v_z': 0,
		'gspd': 83.8546382418,
		'hdg': test_runways['Runway14'].hdg,  # Heading towards Runway14
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

ua6 = fs.plane_manager.planes[-1]
runway = test_runways['Runway14']

fs.add_command_by_callsign('UA6', CommandType.REALIGN, last_update=0, argument=runway)
fs.run(ticks=2500)  # Run the simulation for 500 ticks

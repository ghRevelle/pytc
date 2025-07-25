import numpy as np
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from airport import Runway, Airport
from commands import *
from planestates import PlaneState

# Base coordinates around San Diego area
base_lat = 32.7329
base_lon = -117.1897

# Use coordinates relative to 32.7329° N, -117.1897° W (San Diego)
# Create runways with realistic coordinates for this location
test_runways = {
	'Runway9': Runway((32.73713, -117.20433), (32.73, -117.175), "Runway9"),  # NW-SE runway
	'Runway27': Runway((32.73, -117.175), (32.73713, -117.20433), "Runway27"),  # SE-NW runway
}

test_airport = Airport(test_runways)

rolling_initial_state = [{
		'callsign': 'UA6',
		'lat': base_lat + 0.15,  # North of the airport
		'lon': base_lon - 0.5,  # West of the airport
		'alt': 400,
		'v_z': 0,
		'gspd': 83.8546382418,
		'hdg': test_runways['Runway9'].hdg,  # Heading towards Runway9
		'state': PlaneState.AIR,
		'time_added': 0  # Time added to the simulation
	}]

fs = FlightSimulator(display_size=(900, 900), airport = test_airport, plane_manager = PlaneManager(), rolling_initial_state=rolling_initial_state)
fs.pass_airport_to_pm(test_airport)

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

runway = test_runways['Runway9']

fs.add_command_by_callsign('UA6', CommandType.REALIGN, last_update=0, argument=runway)
fs.add_command_by_callsign('UA6', CommandType.CLEARED_TO_LAND, last_update=200, argument=runway)
fs.run(ticks=2500)  # Run the simulation for 2e500 ticks

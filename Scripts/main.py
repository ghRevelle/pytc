import numpy as np
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from airport import Runway, Airport
from commands import *
from planestates import PlaneState
import time

# Base coordinates around San Diego area
base_lat = 32.7329
base_lon = -117.1897

# Use coordinates relative to 32.7329° N, -117.1897° W (San Diego)
# Create runways with realistic coordinates for this location
test_runways = {
	9: Runway((32.73713, -117.20433), (32.73, -117.175), 9),  # NW-SE runway
	27: Runway((32.73, -117.175), (32.73713, -117.20433), 27),  # SE-NW runway
}

test_airport = Airport(test_runways)

rolling_initial_state = [
	{
		'callsign': 'UA6',
		'lat': base_lat + 0.075,  # North of the airport
		'lon': base_lon - 0.25,  # West of the airport
		'alt': 400,
		'v_z': 0,
		'gspd': 83.8546382418,
		'hdg': test_runways[9].hdg,  # Heading towards Runway9
		'state': PlaneState.AIR,
		'time_added': 0  # Time added to the simulation
	},
	{
		'callsign': 'UA93',
		'lat': base_lat + 0.075,  # North of the airport
		'lon': base_lon - 0.25,  # West of the airport
		'alt': 400,
		'v_z': 0,
		'gspd': 83.8546382418,
		'hdg': test_runways[9].hdg,  # Heading towards Runway9
		'state': PlaneState.AIR,
		'time_added': 500  # Time added to the simulation
	}
]

fs = FlightSimulator(display_size=(900, 900), airport = test_airport, plane_manager = PlaneManager(), rolling_initial_state=rolling_initial_state)
fs.pass_airport_to_pm(test_airport)

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

runway = test_runways[9]

fs.add_command_by_callsign('UA6', CommandType.REALIGN, last_update=0, argument=runway)
fs.add_command_by_callsign('UA6', CommandType.CLEARED_TO_LAND, last_update=200, argument=runway)

fs.add_command_by_callsign('UA93', CommandType.REALIGN, last_update=500, argument=runway)
fs.add_command_by_callsign('UA93', CommandType.CLEARED_TO_LAND, last_update=700, argument=runway)

print(fs.plane_manager.show_ids())

for i in range(2500):
	# Run the simulation for 2500 ticks
	if i % 5 == 0:
		fs.plane_manager.print_planes(i)
	fs.tick()

#fs.run(ticks=2500)  # Run the simulation for 2e500 ticks

import numpy as np
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from airport import Runway, Airport
from commands import *
from planestates import PlaneState
from rolling_initial_state_20250301 import *

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

rolling_initial_state = []

# Convert the rolling initial state states to use PlaneState enums
for state in rolling_initial_state_01:
	rolling_initial_state.append(state.copy())
	if rolling_initial_state[-1]['state'] == 'takeoff':
		rolling_initial_state[-1]['state'] = PlaneState.QUEUED
	elif rolling_initial_state[-1]['state'] == 'landing':
		rolling_initial_state[-1]['state'] = PlaneState.AIR

# Check for duplicate callsigns in the rolling initial state
seen = set()
for state in rolling_initial_state:
	if state['callsign'] not in seen:
		seen.add(state['callsign'])
	else:
		raise ValueError(f"Duplicate callsign found: {state['callsign']}")

fs = FlightSimulator(display_size=(900, 900), airport = test_airport, plane_manager = PlaneManager(), rolling_initial_state=rolling_initial_state)

runway = test_runways[27]

# fs.add_command_by_callsign('BAW82P', CommandType.CLEARED_TO_LAND, last_update=80, argument=runway)

for i in range(2500):
	# # Run the simulation for 2500 ticks
	for state in rolling_initial_state:
		if state['time_added'] == i and state['state'] == PlaneState.AIR:
			fs.add_command_by_callsign(state['callsign'], CommandType.CLEARED_TO_LAND, last_update = i+90, argument=runway)
		elif state['time_added'] == i and state['state'] == PlaneState.QUEUED:
			fs.add_command_by_callsign(state['callsign'], CommandType.LINE_UP_AND_WAIT, last_update = i+1, argument=runway)
			fs.add_command_by_callsign(state['callsign'], CommandType.CLEARED_FOR_TAKEOFF, last_update = i+2, argument=runway)
	# if i >= 1400 and i <= 1800 and i % 100 == 0:
	# 	target_id = copy.deepcopy(fs.plane_manager.airport.get_top_of_queue())
	# 	fs.add_command(Command(command_type=CommandType.LINE_UP_AND_WAIT, target_id=target_id, last_update=i+1, argument=runway))
	# 	fs.add_command(Command(command_type=CommandType.CLEARED_FOR_TAKEOFF, target_id=target_id, last_update=i+2, argument=runway))
	fs.tick()

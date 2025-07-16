from plane import *
import numpy as np
from flightsim import FlightSimulator

# test planes
planes = []
for i in range(5):
	# Create 5 planes with random positions and speeds
	planes.append(SimPlane(
		{
			'id': f"UA{i+1}",
			'lat': np.random.uniform(-1, 1),  # random latitude
			'lon': np.random.uniform(-1, 1),  # random longitude
			'alt': np.random.uniform(0, 12000),  # random altitude in meters
			'v_z': np.random.uniform(-10, 10),  # random vertical speed in meters per second
			'gspd': np.random.uniform(200, 900),  # random ground speed in meters per second
			'hdg': np.random.uniform(0, 360)  # random heading in degrees
		}
	))

fs = FlightSimulator(display_size=(640, 480), planes=planes)

fs.add_command({ # sample command to turn a plane
	'id': 'UA1',
	'cmd': 'turn',
	'args': {'hdg': 90},
	'last_updated': 100
})

fs.run(ticks=500)  # Run the simulation for 500 ticks
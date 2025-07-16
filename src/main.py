from plane import *
import numpy as np
from flightsim import FlightSimulator
from airport import Runway, Airport

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

test_runways = {
	'Runway1': Runway((0, 0), (1, 1), 90, 1000)
}

test_airport = Airport(test_runways)

fs = FlightSimulator(display_size=(640, 480), planes=planes, airport=test_airport)

fs.add_command({ # sample command to turn a plane
	'id': 'UA1',
	'cmd': 'turn',
	'args': {'hdg': 90},
	'last_updated': 100
})

fs.run(ticks=500)  # Run the simulation for 500 ticks
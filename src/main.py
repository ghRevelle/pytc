from plane import Plane
import numpy as np
from flightsim import FlightSimulator

# test plane
myPlane = Plane(
	{
        'id': "UA93",
		'lat': 0.0,  # latitude
		'lon': 0.0,  # longitude
		'alt': 0.0,  # altitude in meters
		'v_z': 0.0,  # vertical speed in meters per second
		'gspd': 420,  # ground speed in meters per second
		'hdg': 30,  # heading in degrees
	}
)
# test multiple planes
planes = []
for i in range(5):
	# Create 5 planes with random positions and speeds
	planes.append(Plane(
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

fs.run(ticks=250)  # Run the simulation for 250 ticks
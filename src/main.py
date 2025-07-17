from plane import *
import numpy as np
import shapely
from flightsim import FlightSimulator
from airport import Runway, Airport

# test planes
planes = []
for i in range(5):
	# Create 5 planes with random positions and speeds
	planes.append(SimPlane(
		{
			'id': f"UA{i+1}",
			'lat': np.random.uniform(-0.01, 0.01),  # random latitude
			'lon': np.random.uniform(-0.01, 0.01),  # random longitude
			'alt': np.random.uniform(0, 12000),  # random altitude in meters
			'v_z': np.random.uniform(-10, 10),  # random vertical speed in meters per second
			'gspd': np.random.uniform(60, 100),  # random ground speed in meters per second
			'hdg': np.random.uniform(0, 360)  # random heading in degrees
		}
	))
planes.append(SimPlane(
	{
		'id': 'UA6',
		'lat': 0.0,
		'lon': -0.1,
		'alt': 1000,
		'v_z': 0,
		'gspd': 100,
		'hdg': 45
	}
))
test_runways = {
	'Runway1': Runway((0, 0), (0.05, 0.05)),
	'Runway2': Runway((0.05, 0), (0, 0.05)),
}

test_airport = Airport(test_runways)

fs = FlightSimulator(display_size=(900, 900), planes=planes, airport=test_airport)

fs.add_command({ # sample command to turn a plane
	'id': 'UA1',
	'cmd': 'turn',
	'args': {'hdg': 90},
	'last_updated': 100
})

# fs.add_command({ # sample command to turn a plane onto a runway
# 	'id': 'UA6',
# 	'cmd': 'turn',
# 	'args': {'hdg': test_runways['Runway2'].hdg},
# 	'last_updated': planes[-1].find_turn_initiation_time(test_runways['Runway2'].get_line(), 0)
# })

fs.add_command({ # sample command to turn a plane
	'id': 'UA6',
	'cmd': 'turn',
	'args': {'hdg': test_runways['Runway2'].hdg},
	'last_updated': 60
})

print(f"Plane UA0 will turn to heading {test_runways['Runway2'].hdg} at tick {planes[-1].find_turn_initiation_time(test_runways['Runway2'].get_line(), 0)}")

fs.run(ticks=500)  # Run the simulation for 500 ticks
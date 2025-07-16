from plane import Plane
from pygame_display import *
import time

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
planes = []
for i in range(5):
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

pg_display = Pygame_Display(640, 480)
for i in range(500):
	for plane in planes:
		plane.tick()
		pg_display.update_display(plane.get_state())
	# print(myPlane.get_state())
	time.sleep(0.01)
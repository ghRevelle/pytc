from plane import Plane

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

myPlane.tick()

print(myPlane.get_state())

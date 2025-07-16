from plane import Plane
from display import *
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

fig, ax = plt.subplots()
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim((-0.03, 0.03))
ax.set_ylim((-0.03, 0.03))
plt.grid()
"""
for i in range(10):
	myPlane.tick()
	add_position(myPlane.get_state())
	print(myPlane.get_state())
	time.sleep(0.2)
plot_positions("UA93")
display()
"""
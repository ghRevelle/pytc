# Plane class for flight simulator
import math
import geopy.distance

class Plane:
	def __init__(self, init_state : dict):
		self.state = init_state
		"""state = {'id' : callsign,
					'lat' : lat position,
		            'lon' : lon position,
					'alt' : altitude,
					'hdg' : heading,
					'gspd' : ground speed,
					'v_z' : climb/sink rate}"""

	def get_state(self) -> object:
		return self.state

	def tick(self): # advance time by 1 second
		next_pt = geopy.distance.distance(meters=self.state['gspd']).destination((self.state['lon'],self.state['lat']), bearing=self.state['hdg'])
		self.state['lat'] = next_pt.latitude
		self.state['lon'] = next_pt.longitude
		self.state['alt'] += self.state['v_z']
		if self.state['alt'] < 0:
			self.state['alt'] = 0
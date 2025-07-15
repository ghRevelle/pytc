class Plane:
	def __init__(self, init_state: dict):
		self.state = init_state
		"""state = {'id' : callsign,
		            'x' : x position,
					'y' : y position,
					'alt' : altitude,
					'hdg' : heading,
					'gspd' : ground speed,
					'v_z' : climb/sink rate}"""

	def tick(self): # advance time by 10 seconds
		self.state['x'] += self.state['gspd']*math.sin(90 - self.state['hdg'])
# Plane class for flight simulator
import math
import geopy.distance

class Plane:
	"""Plane class to represent a single aircraft in a flight simulation environment."""
	def __init__(self, init_state : dict):
		"""Initialize the Plane with its state.
		Args:
			init_state (dict): Initial state of the plane containing the following keys:
				'id' (str): Unique identifier for the plane (callsign).
				'lat' (float): Latitude position of the plane in degrees.
				'lon' (float): Longitude position of the plane in degrees.
				'alt' (float): Altitude of the plane in meters.
				'hdg' (float): Heading of the plane in degrees.
				'gspd' (float): Ground speed of the plane in meters per second.
				'v_z' (float): Vertical speed (climb/sink rate) in meters per second.
		"""
		self.state = init_state

	def get_state(self) -> object:
		"""Get the current state of the plane.
		Returns:
			object: The current state of the plane.
			Contains keys: 'id', 'lat', 'lon', 'alt', 'v_z', 'gspd', 'hdg'.
		"""
		return self.state

	def tick(self):
		"""Update the plane's position based on its ground speed and heading."""
		# Calculate the next position based on ground speed and heading
		next_pt = geopy.distance.distance(meters=self.state['gspd']).destination((self.state['lat'],self.state['lon']), bearing=self.state['hdg'])
		# Update the plane's state with the new position and altitude
		self.state['lat'] = next_pt.latitude
		self.state['lon'] = next_pt.longitude
		self.state['alt'] += self.state['v_z']
		if self.state['alt'] < 0:
			self.state['alt'] = 0

class SimPlane(Plane):
	"""Simulated plane class to represent a plane in a flight simulation environment, able to be commanded to change its state."""
	def __init__(self, init_state: dict):
		"""Create a simulated plane with no initial command."""
		super().__init__(init_state)
		self.state['command'] = {
			'cmd': None,  # Command to execute
			'args': {},   # Arguments for the command
			'last_updated': 0  # Timestamp of the last command update
		}

	def change_command(self, new_command : dict):
		"""Send a command to the plane.
		Args:
			new_command (dict): New command for the plane containing the same keys as in the constructor.
		"""
		if 'cmd' in new_command and 'args' in new_command and 'last_updated' in new_command:
			self.state['command'] = new_command
		else:
			raise ValueError("Command must contain 'cmd', 'args', and 'last_updated' keys.")
		
	def turn(self, desired_hdg: float, current_tick: int = 0):
		"""Turn the plane to a new heading.
		Args:
			desired_hdg (float): The desired heading in degrees.
			current_tick (int): The current simulation tick.
		"""
		if 0 <= desired_hdg < 360:
			self.change_command({
				'cmd': 'turn',
				'args': {'hdg': desired_hdg},
				'last_updated': current_tick
			})
		else:
			raise ValueError("Heading must be between 0 and 360 degrees.")
		
	def tick(self):
		super().tick()
		if self.state['command']['cmd'] == 'turn':
			# If the command is to turn, update the heading
			current_hdg = self.state['hdg']
			desired_hdg = self.state['command']['args'].get('hdg', current_hdg)
			if 0 <= desired_hdg < 360:
				if desired_hdg < current_hdg:
					self.state['hdg'] = current_hdg - 1 # Turn left
				elif desired_hdg > current_hdg:
					self.state['hdg'] = current_hdg + 1 # Turn right
				else:
					self.state['command']['cmd'] = None # Command completed
			else:
				raise ValueError("Heading must be between 0 and 360 degrees.")
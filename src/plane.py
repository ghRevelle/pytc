# Plane class for flight simulator
import shapely
import math
import geopy.distance
import numpy as np
import utils
from airport import Runway

class Plane:
	"""Plane class to represent a single aircraft in a flight simulation environment."""
	def __init__(self, init_state : dict):
		"""Initialize the Plane with its state.
		Args:
			init_state (dict): Initial state of the plane containing the following keys:
				'id' (str): Unique identifier for the plane (callsign).
				'type' (str): Plane model
				'turn_rate' (float): Turn rate of the plane in deg/sec (based only on model)
				'stall_speed' (float): Plane's minimum speed in kts (based only on model)

				'lat' (float): Latitude position of the plane in degrees.
				'lon' (float): Longitude position of the plane in degrees.
				'alt' (float): Altitude of the plane in meters.
				'hdg' (int): Heading of the plane in degrees.
				'gspd' (float): Ground speed of the plane in meters per second.
				'v_z' (float): Vertical speed (climb/sink rate) in meters per second.
				'traj' (list): List of the plane's trajectory points in world coordinates.
				'vel' (tuple): The plane's next world point
		"""
		self.state = init_state
		self.state['type'] = "Cessna" # default value

		if self.state['type'] == "Cessna":
			self.state['turn_rate'] = 3
			self.state['stall_speed'] = 62.4

	# getter-setters
	def get_state(self) -> object:
		"""Get the current state of the plane.
		Returns:
			object: The current state of the plane.
			Contains keys: 'id', 'lat', 'lon', 'alt', 'v_z', 'gspd', 'hdg', 'traj'.
		"""
		if not isinstance(self.state['hdg'], int):
			self.state['hdg'] = int(round(self.state['hdg']))
		return self.state
	
	def get_traj(self) -> object:
		"""Get the current trajectory of the plane.
		Returns:
			object: The list of plane trajectory points.
			Returns as a list.
		"""
		return self.state['traj']
	
	def get_traj_line(self):
		"""Get the current trajectory line of the plane.
		Returns:
			object: The trajectory line as a shapely LineString object.
		"""
		self._calculate_velocity()
		next_point = self.state['vel']
		return shapely.geometry.LineString([(self.state['lat'], self.state['lon']), (next_point.latitude, next_point.longitude)])

	def set_traj(self, traj):
		"""Set the current trajectory of the plane.
		Right now, used only for rendering purposes.
		"""
		self.state['traj'] = traj

	# class methods
	def _calculate_velocity(self):
		# Calculate the next position based on ground speed and heading
		self.state['vel'] = geopy.distance.distance(meters=self.state['gspd']).destination((self.state['lat'],self.state['lon']), bearing=self.state['hdg'])
	
	def apply_turn_rate_penalty(self, speed, turn_rate, stall_speed):
		"""Induce airspeed loss based on how fast the plane is turning.
		Args:
			speed: the plane's current groundspeed.
			turn_rate: the plane's current turnrate.
			stall_speed: the plane's minimum speed in kts (based only on model).
		Returns:
			float: the plane's new groundspeed after the penalty."""
		
		penalty = 0.02 * abs(turn_rate)
		return max(stall_speed, speed - penalty)

	def tick(self):
		"""
		Update the plane's position based on its ground speed and heading.
		Update the plane's groundspeed based on its vertrate and turnrate.
		"""
		self._calculate_velocity()
		# Update the plane's state with the new position and altitude
		self.state['lat'] = self.state['vel'].latitude
		self.state['lon'] = self.state['vel'].longitude
		self.state['alt'] += self.state['v_z']
		if self.state['alt'] < 0:
			self.state['alt'] = 0

		# Compute the heading unit vector by changing heading to radians
		hdg_rads = math.radians(self.state['hdg'])
		ux = math.cos(hdg_rads)
		uy = math.sin(hdg_rads)

		hdg_vec = (ux, uy)
		
		self.state['traj'] = [(self.state['lon'] + hdg_vec[1] / 1000 * i, self.state['lat'] + hdg_vec[0] / 1000 * i) for i in range(0, 11)]

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
		
	def turn(self, desired_hdg: int, current_tick: int = 0):
		"""Turn the plane to a new heading.
		Args:
			desired_hdg (int): The desired heading in degrees.
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
		
	def cleared_to_land(self, current_tick: int = 0):
		"""Put the plane on its final glideslope.
		By default, this slope is 850 ft / min at 140 kts gspd"""

		raise NotImplementedError()

	def get_turn_radius(self):
		"""Calculate the turn radius in meters of the plane based on its ground speed."""
		turn_rate_deg_per_sec = 1  # 1 deg per tick
		turn_rate_rad_per_sec = math.radians(turn_rate_deg_per_sec)
		if turn_rate_rad_per_sec == 0:
			return float('inf')
		return self.state['gspd'] / turn_rate_rad_per_sec
	
	def find_turn_initiation_distance(self, runway_line):
		"""
		Find the point where the plane should start turning to become parallel and colinear with the runway line.
		Handles turns greater than 180 degrees by choosing the correct offset direction.
		direction: 'left', 'right', or None (auto-choose shortest turn)
		"""
		turn_radius = self.get_turn_radius()
		plane_pos = (self.state['lon'], self.state['lat'])
		plane_hdg = self.state['hdg']
		distance = None
		# 1. Find the desired final heading (runway heading)
		print(runway_line.coords[0])
		ry1, rx1, _ = runway_line.coords[0]
		ry2, rx2, _ = runway_line.coords[1]
		runway_hdg = math.degrees(math.atan2(rx2 - rx1, ry2 - ry1)) % 360
		traj_line = self.get_traj_line()
		tx1, ty1 = traj_line.coords[0]
		tx2, ty2 = traj_line.coords[1]

		# Extend the length of the runway line x1000 in both directions
		extended_runway = shapely.geometry.LineString([(rx1 - 1000 * (rx2 - rx1), ry1 - 1000 * (ry2 - ry1)), (rx2 + 1000 * (rx2 - rx1), ry2 + 1000 * (ry2 - ry1))])
		extended_traj = shapely.geometry.LineString([(tx1 - 1000 * (tx2 - tx1), ty1 - 1000 * (ty2 - ty1)), (tx2 + 1000 * (tx2 - tx1), ty2 + 1000 * (ty2 - ty1))])

		intersection = utils.calculate_intersection(extended_runway, extended_traj)
		if intersection is None:
			return None
		total_distance = shapely.geometry.Point(plane_pos).distance(shapely.geometry.Point(intersection))

		angle_diff = 180 - (runway_hdg - plane_hdg)

		distance = total_distance - utils.meters_to_degrees(heading=self.state['hdg'], meters=turn_radius) / math.tan(math.radians(angle_diff / 2)) # distance in degrees
		# Compute the destination point along the heading
		# destination = geopy.distance.distance(nautical=distance * 111.32).destination(
		# 	(self.state['lat'], self.state['lon']), plane_hdg
		# )
		# distance_meters = geopy.distance.geodesic((self.state['lat'], self.state['lon']), (destination.latitude, destination.longitude)).meters
		distance_meters = utils.degrees_to_meters(heading=plane_hdg, degrees=distance)
		initiation_point = geopy.distance.distance(meters=distance).destination((self.state['lat'], self.state['lon']), bearing=plane_hdg)
		print("Distance:", utils.meters_to_degrees(heading=self.state['hdg'], meters=turn_radius) / math.tan(math.radians(angle_diff / 2)))
		print("Turning radius:", utils.meters_to_degrees(heading=self.state['hdg'], meters=turn_radius))
		return distance_meters

	def find_turn_initiation_time(self, runway_line, current_tick):
		"""Find the time to initiate a turn to align with a runway.
		Args:
			runway_line (shapely.geometry.LineString): The runway line to align with.
			turn_radius (float): The radius of the turn in the same units as the coordinates.
			direction (str): 'left' or 'right' indicating the direction of the turn.

		Returns:
			float: The time in seconds to initiate the turn.
		"""
		# Calculate the distance to the turn initiation point
		distance = self.find_turn_initiation_distance(runway_line)
		if distance is None:
			return None

		if distance < 0:
			raise ValueError("Distance cannot be negative.")
		return distance / self.state['gspd'] + current_tick # Time = distance / speed + current time

	def tick(self):
		super().tick()
		if self.state['command']['cmd'] == 'turn':
			# If the command is to turn, update the heading
			current_hdg = self.state['hdg']
			desired_hdg = self.state['command']['args'].get('hdg', current_hdg)
			if 0 <= desired_hdg < 360:
				diff = (desired_hdg - current_hdg + 360) % 360
				if abs(diff) <= self.state['turn_rate']:
					self.state['hdg'] = desired_hdg
					self.state['command']['cmd'] = None # Command completed
				elif diff <= 180:
					self.state['hdg'] = (current_hdg + self.state['turn_rate']) % 360 # Turn right
				else:
					self.state['hdg'] = (current_hdg - self.state['turn_rate']) % 360 # Turn left
				
				self.state['gspd'] = self.apply_turn_rate_penalty(self.state['gspd'], self.state['turn_rate'], self.state['stall_speed'])
			else:
				raise ValueError("Heading must be between 0 and 360 degrees.")
# Plane class for flight simulator
import shapely
import math
import geopy.distance
import numpy as np
import utils
from commands import *
from command_handlers import CommandProcessor
from planestates import *

class Plane:
	"""Plane class to represent a single aircraft in a flight simulation environment."""
	def __init__(self, init_state: dict):
		"""Initialize the Plane with its state.
		Args:
			init_state (dict): Initial state of the plane containing the following keys:
				'callsign' (str): The plane's callsign.
				'model' (str): Plane model

				'state' (Enum): Plane's current state (just AIR by default)
				'thistick' (List): List of bools tracking whether the plane:
				0	landed_this_tick,
				1	tookoff_this_tick,
				2	crashed_this_tick

				'turn_rate' (float): Turn rate of the plane in deg/sec (based on model)
				'stall_speed' (float): Plane's minimum speed in m/s (based on model)
				'nex_speed' (float): Plane's never-exceed speed in m/s (based on model)
				'crz_speed' (float): Plane's default speed in a traffic pattern (based on model)
				'ldg_speed' (float): Plane's target landing speed (based on model)

				'nex_alt' (float): Plane's never-exceed altitude in meters (based on model)
				'crz_alt' (float): Plane's typical approach altitude in meters (based on model)

				'asc_rate' (float): Plane's maximum no-speed-loss climb rate in m/s (based only on model)
				'dsc_rate' (float): Plane's maximum descent rate in m/s (asc_rate * 1.5)
				'acc_z_max' (float): Plane's maximum vertical acceleration in m/s^2 (based only on model)
				'acc_xy_max' (float, optional): Plane's maximum horizontal acceleration in m/s^2

				'lat' (float): Latitude position of the plane in degrees.
				'lon' (float): Longitude position of the plane in degrees.
				'alt' (float): Altitude of the plane in meters.
				'hdg' (int): Heading of the plane in degrees.
				'gspd' (float): Ground speed of the plane in meters per second.
				'v_z' (float): Vertical speed (climb/sink rate) in meters per second.
				'traj' (list): List of the plane's trajectory points in world coordinates.
				'vel' (tuple): The plane's next world point
		"""
		self.callsign = init_state['callsign']
		self.lat = init_state['lat']
		self.lon = init_state['lon']
		self.alt = init_state['alt']
		self.hdg = init_state['hdg']
		self.gspd = init_state['gspd']
		self.v_z = init_state['v_z']

		if self.alt > 0:
			self.state = PlaneState.AIR
		else:
			self.state = PlaneState.GROUND

		self.thistick = [False, False, False]

		self.acc_xy = init_state.get('acc_xy', 0.0)  # Optional, default to 0.0

		self.id = init_state['id']

		self.traj = None
		self._calculate_velocity()

		self.model = "A320" # default value

		match self.model:
			case "Cessna":
				self.stall_speed = 24.1789448 # m/s, or 62.4 kts / 1.94384
				self.nex_speed = 83.8546382418 # m/s, or 163 kts / 1.94384
				self.crz_speed = 48.8723351716 # m/s, or 95 kts / 1.94834
				self.ldg_speed = 33.4389662 # m/s, or 65 kts / 1.94384

				self.turn_rate = utils.get_turn_rate(25, self.crz_speed)

				self.nex_alt = 4114.8 # meters, or 13500 / 3.281
				self.crz_alt = 304.8 # meters, or 1000 / 3.281

				self.asc_rate = 3.556 # m/s
				self.dsc_rate = 5.334 # m/s, or asc_rate * 1.5
				self.acc_z_max = 0.5 # m/s^2
				self.acc_xy_max = 1.0 # m/s^2, total guess

			case "A320":
				self.stall_speed = 73.5656 # m/s or 143 kts / 1.94384
				self.nex_speed = 180.056 # m/s or 350 kts / 1.94834
				self.crz_speed = 82.3111 # m/s or 160 kts / 1.94834
				self.ldg_speed = 77.1667 # m/s or 150 kts / 1.94834

				self.turn_rate = utils.get_turn_rate(25, self.crz_speed) # degrees/sec

				self.nex_alt = 12131.04 # meters, or 39800 / 3.281
				self.crz_alt = 457.2 # meters, or 1500 / 3.281"""

				self.asc_rate = 12.7 # m/s
				self.dsc_rate = 19.05 # m/s, or asc_rate * 1.5
				self.acc_z_max = 1.5 # m/s^2, total guess
				self.acc_xy_max = 3 # m/s^2, this is an informed guesstimate

			#case "BE9L":

			#case "PA28":

		self.command = Command(command_type=CommandType.CRUISE, target_id=self.id, last_update=0, argument=None)  # Default command
		
		# Misc
		self.turn_start_time = -1 # Used for landing

	# getter-setters
	def get_state(self) -> object:
		"""Get the current state of the plane.
		Returns:
			object: The current state of the plane.
			Contains keys: 'callsign', 'lat', 'lon', 'alt', 'v_z', 'gspd', 'hdg', 'traj'.
		"""
		state_dict = {
        	'callsign': self.callsign,
        	'lat': self.lat,
        	'lon': self.lon,
        	'alt': self.alt,
        	'hdg': self.hdg,
        	'gspd': self.gspd,
        	'v_z': self.v_z,
			'traj': self.traj
		}
		
		# Convert heading to an integer
		state_dict['hdg'] = state_dict['hdg']
		
		return state_dict
	
	def get_traj(self) -> object:
		"""Get the current trajectory of the plane.
		Returns:
			object: The list of plane trajectory points.
			Returns as a list.
		"""
		return self.traj
	
	def get_traj_line(self):
		"""Get the current trajectory line of the plane.
		Returns:
			object: The trajectory line as a shapely LineString object.
		"""
		self._calculate_velocity()
		next_point = self.next_pt
		return shapely.geometry.LineString([(self.lon, self.lat), (next_point.longitude, next_point.latitude)])

	def set_traj(self, traj):
		"""Set the current trajectory of the plane.
		Right now, used only for rendering purposes.
		"""
		self.traj = traj

	# commands
	def change_command(self, new_command: Command):
		if not isinstance(new_command, Command):
			raise TypeError("Expected a Command object.")
		self.command = new_command

	def get_turn_radius(self):
		"""Calculate the turn radius in meters of the plane based on its ground speed."""
		turn_rate_rad_per_sec = math.radians(self.turn_rate)
		if turn_rate_rad_per_sec == 0:
			return float('inf')
		return self.gspd / turn_rate_rad_per_sec

	def find_turn_initiation_time(self, target_line: shapely.geometry.LineString, current_tick: int = 0): # TODO: move this somewhere else
		"""
		Find the time to initiate a turn to align with a target line. !!NOTE:!! This method does not work if the target is behind the plane.
		Args:
			target_line (shapely.geometry.LineString): The target line to align with. The direction of the line is the desired heading.
			current_tick (int): The current simulation tick.
		Returns:
			int: The tick at which the turn should be initiated, or None if it cannot be determined.
		"""
		if not isinstance(target_line, shapely.geometry.LineString):
			raise TypeError("target_line must be a shapely LineString object.")


		# 1. Get inputs

		turn_radius = utils.meters_to_degrees(heading=self.hdg, meters=self.get_turn_radius()) # turn radius in degrees
		# Get the plane's position and trajectory line
		plane_pos_xy = (self.lon, self.lat)
		traj_line = self.get_traj_line()
		tx1, ty1 = traj_line.coords[0]
		tx2, ty2 = traj_line.coords[1]
		# Get the target line coordinates
		rx1, ry1 = target_line.coords[0]
		rx2, ry2 = target_line.coords[1]
		# Get headings
		plane_hdg = self.hdg
		target_hdg = (450 - math.degrees(math.atan2(ry2 - ry1, rx2 - rx1))) % 360


		# 2. Find the intersection point of the extended target and trajectory lines

		# Extend the target and trajectory lines to ensure intersection
		# Find slopes
		# slope_target = (ry2 - ry1) / (rx2 - rx1) if (rx2 - rx1) != 0 else 99999 # HACK: close enough to infinity
		# slope_traj = (ty2 - ty1) / (tx2 - tx1) if (tx2 - tx1) != 0 else 99999
		# extend_distance = 3 # how many degrees of lat/lon to extend the lines by
		# # Extend lines
		# extended_target = shapely.geometry.LineString([(rx1 - extend_distance, ry1 - extend_distance*slope_target), (rx2 + extend_distance, ry2 + extend_distance*slope_target)])
		# extended_traj = shapely.geometry.LineString([(tx1 - extend_distance, ty1 - extend_distance*slope_traj), (tx2 + extend_distance, ty2 + extend_distance*slope_traj)])
		extended_target = utils.extend_line(target_line, 3)
		extended_traj = utils.extend_line(traj_line, 3)
		# Find intersection
		intersection = utils.calculate_intersection(extended_target, extended_traj)


		# 3. Find where to start the turn

		# Calculate the total distance to the intersection point in degrees
		total_distance = shapely.geometry.Point(plane_pos_xy).distance(shapely.geometry.Point(intersection))
		
		# Now the REAL calculation begins

		# Calculate the angle difference between the plane and target headings
		angle_diff = target_hdg - plane_hdg
		# Take half of the supplementary angle
		half_supp_diff = (180 - angle_diff) / 2

		# Calculate the distance to the turn initiation point
		# HACK: I don't know what abs() is doing here, but the second term must always be positive or we can't turn left
		distance = total_distance - turn_radius / abs(math.tan(math.radians(half_supp_diff))) # distance in degrees

		# Convert back to meters
		distance_meters = utils.degrees_to_meters(heading=plane_hdg, degrees=distance)

		# Check if the distance is valid
		if distance_meters is None:
			return None

		if distance_meters < 0:
			raise ValueError("Distance cannot be negative.")
		
		# Finally done
		return int(round(distance_meters / self.gspd + current_tick))  # Time = distance / speed + current time

	# class methods
	def _calculate_velocity(self):
		# Calculate the next position based on ground speed and heading
		self.next_pt = geopy.distance.distance(meters=self.gspd).destination((self.lat,self.lon), bearing=self.hdg)
	
	def _turn(self, current_hdg, desired_hdg):
		"""Turn the plane to a new heading."""
		diff = (desired_hdg - current_hdg + 360) % 360
		if abs(diff) <= self.turn_rate:
			self.hdg = desired_hdg
			return
		elif diff <= 180:
			self.hdg = (current_hdg + self.turn_rate) % 360
		else:
			self.hdg = (current_hdg - self.turn_rate) % 360

		# Apply turn rate penalty to ground speed
		# (commented out for now, we'll assume auto throttle handles it)
		# self.gspd = self._apply_turn_rate_penalty(
		# 	self.gspd,
		# 	self.turn_rate,
		# 	self.stall_speed
		# )

	def _calculate_tod(self, current_alt):
		"""Calculate the top of descent to a target position in nautical miles."""
		return current_alt * 3 / 1000  # Distance to start descent in nautical miles

	def _calculate_rod(self, current_speed):
		"""Calculate the rate of descent to a target position in meters per second."""
		return 5 * utils.mps_to_knots(current_speed) * 0.00508
	
	def _calculate_target_acc_descend(self, current_speed, current_alt):
		"""Calculate the target acceleration for descent based on current speed and altitude."""
		# Calculate the top of descent (m)
		tod = self._calculate_tod(current_alt) * 1852  # Convert to meters
		target_final_speed = self.ldg_speed
		# Calculate the target acceleration for descent using v² = u² + 2as
		# Rearranged to: a = (v² - u²) / (2s)
		acceleration = (target_final_speed**2 - current_speed**2) / (2 * tod)
		return acceleration

	def _apply_turn_rate_penalty(self, speed, turn_rate, stall_speed):
		"""Induce airspeed loss based on how fast the plane is turning.
		Args:
			speed: the plane's current groundspeed in m/s.
			turn_rate: the plane's current turnrate.
			stall_speed: the plane's minimum speed in kts (based only on model).
		Returns:
			float: the plane's new groundspeed in m/s after the penalty."""
		
		penalty = 0.04 * abs(turn_rate)
		return max(stall_speed, speed - penalty)

	def _apply_descend_boost(self, speed, v_speed, max_speed): # currently this function is not working (planes are gaining 0 airspeed from descent)
		"""Induce airspeed gain based on how fast the plane is descending.
		Args:
			speed: the plane's current groundspeed in m/s.
			vertical_speed: the plane's current vertical rate in m/s.
			max_speed: the plane's maximum speed in m/s.
		Returns:
			float: the plane's new groundspeed in m/s after the boost."""

		if v_speed >= 0:
			return speed

		boost = 0.02 * abs(v_speed) # Boost is proportional to the vertical speed
		# it should also be proportional to aircraft weight, but we don't have that info
		return min(speed + boost, max_speed)
	
	@staticmethod
	def proportional_change(current, target, min_value, max_value, max_change):
		"""Induce change in a plane's speed/acceleration state based on current values.
		Args:
			speed: the plane's current value in m/s.
			target: the plane's target value in m/s.
			min_speed: the plane's minimum value in m/s.
			max_speed: the plane's maximum value in m/s.
		Returns:
			float: the plane's new value in m/s after the change."""
		scale = (max_value - min_value) / 2
		if current < min_value or current > max_value:
			return min(max(current, min_value), max_value)
		elif np.isclose(current, target, atol=0.01):
			return target
		elif current > target:
			acc = -max_change * abs((current - target) / scale)
			return max(current + acc, min_value)
		else:
			acc = max_change * abs((current - target) / scale)
			return min(current + acc, max_value)

	def tick(self, tick):
		"""
		Update the plane's position based on its ground speed and heading.
		Update the plane's groundspeed based on its vertrate and turnrate.
		"""

		self.thistick = [False, False, False]

		# Process commands using the command processor
		if not hasattr(self, '_command_processor'):
			self._command_processor = CommandProcessor()
		
		self._command_processor.process_command(self, self.command, tick)
		
		# Update physics and position
		self._update_position()
		self._update_trajectory()

	def _update_position(self):
		"""Update the plane's physical state and position."""
		
		# 1. Apply speed modifications
		self.gspd = self._apply_descend_boost(
			self.gspd,
			self.v_z,
			self.nex_speed
		)
		self.gspd += self.acc_xy


		# 2. Sanity check speed and altitude
		if self.alt < 0:
			self.alt = 0
			self.v_z = 0  # Reset vertical speed if below ground level
		if self.gspd < 0:
			self.gspd = 0
		if self.gspd > self.nex_speed:
			self.gspd = self.nex_speed
		# Update velocity for the next tick
		self._calculate_velocity()
		
		# 3. Update position based on velocity
		self.lat = self.next_pt.latitude
		self.lon = self.next_pt.longitude
		self.alt += self.v_z
		

	def _update_trajectory(self):
		"""Update the plane's trajectory visualization."""
		# Compute the heading unit vector by changing heading to radians
		hdg_rads = math.radians(self.hdg)
		ux = math.cos(hdg_rads)
		uy = math.sin(hdg_rads)

		hdg_vec = (ux, uy)
		
		self.traj = [(self.lon + hdg_vec[1] / 1000 * i, self.lat + hdg_vec[0] / 1000 * i) for i in range(0, 11)]
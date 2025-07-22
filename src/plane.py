# Plane class for flight simulator
import shapely
import math
import geopy.distance
import numpy as np
import utils
from airport import Runway
from commands import *

class Plane:
	"""Plane class to represent a single aircraft in a flight simulation environment."""
	def __init__(self, init_state: dict):
		"""Initialize the Plane with its state.
		Args:
			init_state (dict): Initial state of the plane containing the following keys:
				'callsign' (str): The plane's callsign.
				'model' (str): Plane model
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

		self.acc_xy = init_state.get('acc_xy', 0.0)  # Optional, default to 0.0

		self.id = init_state['id']

		self.traj = None
		self._calculate_velocity()
		self.command: Optional[Command] = None

		self.model = "Cessna" # default value

		if self.model == "Cessna":
			self.turn_rate = 3
			self.stall_speed = 24.1789448 # m/s, or 62.4 kts / 1.94384
			self.nex_speed = 83.8546382418 # m/s, or 163 kts / 1.94384
			self.crz_speed = 48.8723351716 # m/s, or 95 kts / 1.94834
			self.ldg_speed = 33.4389662 # m/s, or 65 kts / 1.94384

			self.nex_alt = 4114.8 # meters, or 13500 / 3.281
			self.crz_alt = 304.8 # meters, or 1000 / 3.281

			self.asc_rate = 3.556 # m/s
			self.dsc_rate = 5.334 # m/s, or 3.556 * 1.5
			self.acc_z_max = 0.5 # m/s^2
			self.acc_xy_max = 1.0 # m/s^2, total guess

		
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
		if not isinstance(state_dict['hdg'], int):
			state_dict['hdg'] = int(round(state_dict['hdg']))
		
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
		next_point = self.vel
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

	def find_turn_initiation_time(self, target_line: shapely.geometry.LineString, current_tick: int = 0):
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
		slope_target = (ry2 - ry1) / (rx2 - rx1) if (rx2 - rx1) != 0 else 99999 # HACK: close enough to infinity
		slope_traj = (ty2 - ty1) / (tx2 - tx1) if (tx2 - tx1) != 0 else 99999
		extend_distance = 3 # how many degrees of lat/lon to extend the lines by
		# Extend lines
		extended_target = shapely.geometry.LineString([(rx1 - extend_distance, ry1 - extend_distance*slope_target), (rx2 + extend_distance, ry2 + extend_distance*slope_target)])
		extended_traj = shapely.geometry.LineString([(tx1 - extend_distance, ty1 - extend_distance*slope_traj), (tx2 + extend_distance, ty2 + extend_distance*slope_traj)])
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
		self.vel = geopy.distance.distance(meters=self.gspd).destination((self.lat,self.lon), bearing=self.hdg)
	
	def _turn(self, current_hdg, desired_hdg):
		"""Turn the plane to a new heading."""
		diff = (desired_hdg - current_hdg + 360) % 360
		if abs(diff) <= self.turn_rate:
			self.hdg = desired_hdg
		elif diff <= 180:
			self.hdg = (current_hdg + self.turn_rate) % 360
		else:
			self.hdg = (current_hdg - self.turn_rate) % 360

		# Apply turn rate penalty to ground speed
		self.gspd = self._apply_turn_rate_penalty(
			self.gspd,
			self.turn_rate,
			self.stall_speed
		)

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

		if self.command is None or self.command.command_type == CommandType.NONE:
			# When no command is given...

			# ...Try to achieve a vertrate proportional to the altitudinal error
			alt_error = self.crz_alt - self.alt
			v_z_target = max(-self.dsc_rate, min(self.asc_rate, alt_error * 0.1))

			self.v_z = self.proportional_change(
				current=self.v_z,
				target=v_z_target,
				min_value=-self.dsc_rate,
				max_value=self.asc_rate,
				max_change=self.acc_z_max
			)

			# ...Try to achieve a lateral acceleration proportional to the gspd error
			gspd_error = self.crz_speed - self.gspd
			gspd_target = max(-self.acc_xy_max, min(self.acc_xy_max, gspd_error * 0.1))

			# If the plane is NOT descending...
			if self.v_z > -self.dsc_rate / 2:
				# Turn on the speed controller
				self.acc_xy = self.proportional_change(
					current=self.acc_xy,
					target=gspd_target,
					min_value=-self.acc_xy_max,
					max_value=self.acc_xy_max,
					max_change= self.acc_xy_max
				)

		elif self.command.command_type == CommandType.TURN:
			current_hdg = self.hdg
			desired_hdg = self.command.argument

			if desired_hdg is None or not (0 <= desired_hdg < 360):
				raise ValueError("TURN command missing valid heading.")

			self._turn(current_hdg, desired_hdg)
			if self.hdg == desired_hdg:
				self.command.command_type = CommandType.NONE
		
		elif self.command.command_type == CommandType.CLEARED_TO_LAND:
			target_runway = self.command.argument

			# Check all the assumptions to stop python from complaining
			if not isinstance(target_runway, Runway):
				raise TypeError("Argument must be a Runway object.")
			if self.command.last_update is None or not isinstance(self.command.last_update, int):
				raise ValueError("Command last_update must be an integer tick value.")
			if self.turn_start_time is None or not isinstance(self.turn_start_time, int):
				self.turn_start_time = -1

			target_hdg = target_runway.hdg
			# Calculate the distance to the runway in nautical miles
			target_dist = utils.degrees_to_nautical_miles(heading = self.hdg, degrees = target_runway.get_line_xy().distance(shapely.geometry.Point((self.lon, self.lat))))

			if self.turn_start_time == -1: # Alignment not yet started
				self.turn_start_time = self.find_turn_initiation_time(target_runway.get_line_xy(), self.command.last_update)
				self.tod = self._calculate_tod(self.alt)  # Calculate top of descent
				self.rod = self._calculate_rod(self.gspd)  # Calculate rate of descent
				self.desired_acc_xy = self._calculate_target_acc_descend(self.gspd, self.alt)  # Calculate target horizontal acceleration for descent
			elif tick >= self.turn_start_time and self.command.last_update < self.turn_start_time:  # Time to align with the runway
				self._turn(self.hdg, target_hdg)
				if self.hdg == target_hdg:
					self.command.last_update = tick  # Update last update time when the turn is completed
			elif self.command.last_update >= self.turn_start_time and target_dist < self.tod and self.alt > 0:  # Already aligned, descend
				self.rod = self._calculate_rod(self.gspd)
				if target_dist < 1:
					self.v_z = self.proportional_change(
						current=self.v_z,
						target=-(self.alt*self.gspd/target_dist),  # Descend at the rate of descent
						min_value=-self.dsc_rate,
						max_value=self.asc_rate,
						max_change=self.acc_z_max
					)
				self.v_z = self.proportional_change(
					current=self.v_z,
					target=-self.rod,  # Descend at the rate of descent
					min_value=-self.dsc_rate,
					max_value=self.asc_rate,
					max_change=self.acc_z_max
				)
				if self.gspd > self.ldg_speed:
					self.acc_xy = self.proportional_change(
						current=self.acc_xy,
						target=self.desired_acc_xy,  # Maintain horizontal acceleration for descent
						min_value=-self.acc_xy_max,
						max_value=self.acc_xy_max,
						max_change=self.acc_xy_max
					)
				else:
					self.acc_xy = self.proportional_change(
						current=self.acc_xy,
						target=0,  # Stop horizontal acceleration when on the ground
						min_value=-self.acc_xy_max,
						max_value=self.acc_xy_max,
						max_change=self.acc_xy_max
					)
			elif self.alt <= 0 and self.gspd > 0:  # If the plane is on the ground, stop all movement
				self.v_z = 0 # Stop vertical movement
				self.acc_xy = -self.acc_xy_max # Slow down as quickly as possible
			elif self.gspd <= 0:  # If the plane is stopped, stop vertical movement
				print(f"Plane {self.callsign} with ID {self.id} has landed.")
				self.command.command_type = CommandType.NONE  # Done
				self.command.last_update = tick  # Update last update time to stop the command
		else:
			raise ValueError(f"Unknown command type: {self.command.command_type}")

		# Update the plane's state with the new position and speed
		self.lat = self.vel.latitude
		self.lon = self.vel.longitude
		self.alt += self.v_z
		self.gspd += self.acc_xy
		if self.alt < 0:
			self.alt = 0
		if self.gspd < 0:
			self.gspd = 0

		# Apply descent boost if descending
		self.gspd = self._apply_descend_boost(
			self.gspd,
			self.v_z,
			self.nex_speed
		)

		# check never-exceed speed
		if self.gspd > self.nex_speed:
			self.gspd = self.nex_speed

		self._calculate_velocity()


		# Compute the heading unit vector by changing heading to radians
		hdg_rads = math.radians(self.hdg)
		ux = math.cos(hdg_rads)
		uy = math.sin(hdg_rads)

		hdg_vec = (ux, uy)
		
		self.traj = [(self.lon + hdg_vec[1] / 1000 * i, self.lat + hdg_vec[0] / 1000 * i) for i in range(0, 11)]


###########################################################################################################################################
"""
class SimPlane(Plane):

	def __init__(self, init_state: dict, slot_manager: FixedSlotPlaneManager):

		super().__init__(init_state)
		self.id = slot_manager.assign_slot(self.state['callsign'])
		self.command: Optional[Command] = None

	def change_command(self, new_command: Command):
		if not isinstance(new_command, Command):
			raise TypeError("Expected a Command object.")
		self.command = new_command
	
	def turn(self, desired_hdg: int, current_tick: int = 0):
		if not (0 <= desired_hdg < 360):
			raise ValueError("Heading must be between 0 and 360 degrees.")
		
		# Create Command and assign it to self.command
		self.change_command(Command(
        	command_type=CommandType.TURN,
        	target_id=self.id,             # assumes `self.id` exists and is unique for the plane
        	last_update=current_tick,
        	argument=desired_hdg
    	))
		
	def cleared_to_land(self, current_tick: int = 0):

		raise NotImplementedError()

	def get_turn_radius(self):
		turn_rate_deg_per_sec = 1  # 1 deg per tick
		turn_rate_rad_per_sec = math.radians(turn_rate_deg_per_sec)
		if turn_rate_rad_per_sec == 0:
			return float('inf')
		return self.state['gspd'] / turn_rate_rad_per_sec
	
	def find_turn_initiation_distance(self, runway_line):
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
		# Calculate the distance to the turn initiation point
		distance = self.find_turn_initiation_distance(runway_line)
		if distance is None:
			return None

		if distance < 0:
			raise ValueError("Distance cannot be negative.")
		return distance / self.state['gspd'] + current_tick # Time = distance / speed + current time

	def tick(self):
		super().tick()
    
		if self.command is None:
			return
    
		if self.command.command_type == CommandType.TURN:
			current_hdg = self.state['hdg']
			desired_hdg = self.command.argument

			if desired_hdg is None or not (0 <= desired_hdg < 360):
				raise ValueError("TURN command missing valid heading.")

			diff = (desired_hdg - current_hdg + 360) % 360
			if abs(diff) <= self.state['turn_rate']:
				self.state['hdg'] = desired_hdg
				self.command = None  # Command completed
			elif diff <= 180:
				self.state['hdg'] = (current_hdg + self.state['turn_rate']) % 360
			else:
				self.state['hdg'] = (current_hdg - self.state['turn_rate']) % 360

			self.state['gspd'] = self.apply_turn_rate_penalty(
            	self.state['gspd'],
            	self.state['turn_rate'],
            	self.state['stall_speed']
        	)

"""
"""
def apply_command(command : Command, plane : SimPlane):
    if command.command_type == CommandType.LINE_UP_AND_WAIT:
        if plane.state['command'] == "holding_short":
            plane.status = "lined_up"
    
    elif command.command_type == CommandType.CLEARED_FOR_TAKEOFF:
        if plane.status == "lined_up":
            plane.status = "takeoff_roll"
            if command.argument:
                plane.heading = command.argument  # only if geography requires it
    
    elif command.command_type == CommandType.CLEARED_TO_LAND:
        if plane.status == "on_approach":
            plane.status = "descending"
    
    elif command.command_type == CommandType.GO_AROUND:
        if plane.status in ["on_approach", "descending"]:
            plane.go_around()  # handles re-entry delay, etc.
    
    elif command.command_type == CommandType.ABORT_TAKEOFF:
        if plane.status == "takeoff_roll":
            plane.abort_takeoff()  # returns to back of queue

"""
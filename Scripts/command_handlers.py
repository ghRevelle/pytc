# Command handler classes for better separation of concerns
from abc import ABC, abstractmethod
import math

import geopy
import numpy as np
from commands import Command, CommandType
import utils
import shapely.geometry
from airport import Runway
from plane import *
from planestates import *

class CommandHandler(ABC):
	"""Abstract base class for command handlers."""
	
	@abstractmethod
	def can_handle(self, command_type: CommandType) -> bool:
		"""Check if this handler can process the given command type."""
		pass
	
	@abstractmethod
	def execute(self, plane: Plane, command: Command, tick: int) -> None:
		"""Execute the command logic for the given plane."""
		pass
	
	# Shared runway alignment functionality
	@staticmethod
	def _validate_runway_command(target_runway, command, plane):
		"""Validate runway-related command parameters."""
		if not isinstance(target_runway, Runway):
			raise TypeError("Argument must be a Runway object.")
		if command.last_update is None or not isinstance(command.last_update, int):
			raise ValueError("Command last_update must be an integer tick value.")
		if plane.turn_start_time is None or not isinstance(plane.turn_start_time, int): # TODO: don't go around adding random attributes to the plane object
			plane.turn_start_time = -1
	
	@staticmethod
	def _calculate_runway_distance(plane, target_runway):
		"""Calculate distance to runway in nautical miles."""
		return utils.degrees_to_nautical_miles(
			heading=plane.hdg, 
			degrees=target_runway.get_line_xy().distance(
				shapely.geometry.Point((plane.lon, plane.lat))
			)
		)
	
	# def _initialize_runway_alignment(self, plane, target_runway, command):
	# 	"""Initialize runway alignment parameters.
	# 	DEPRECATED: Use the new RealignCommandHandler instead."""
	# 	plane.turn_start_time = plane.find_turn_initiation_time(target_runway.get_line_xy(), command.last_update)
	
	def _handle_runway_alignment(self, plane, target_hdg, tick, command):
		"""Handle the runway alignment phase."""
		plane._turn(plane.hdg, target_hdg)
		if math.isclose(plane.hdg, target_hdg, abs_tol=1e-5):
			command.last_update = tick
			return True  # Alignment complete
		return False  # Still aligning
	
	@staticmethod
	def _is_aligned_to_runway(plane, target_runway):
		"""Check if the plane is aligned with the target runway."""
		parallel = math.isclose(plane.hdg, target_runway.hdg, abs_tol=1e-5)
		online = utils.point_to_line_distance(
			utils.latlon_to_meters(plane.lat, plane.lon),
			utils.latlon_to_meters(target_runway.get_start_point_ll().latitude, target_runway.get_start_point_ll().longitude),
			utils.latlon_to_meters(target_runway.get_end_point_ll().latitude, target_runway.get_end_point_ll().longitude)
		) < 50
		return parallel and online


class RealignCommandHandler(CommandHandler):
	"""Handler for realigning planes to the runway centerline."""

	def __init__(self):
		self.init_dist = None
		self.dir = None  # Direction to turn: "left" or "right"
		self.parallel = False  # Flag to track if the parallel alignment is complete

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.REALIGN
	
	def execute(self, plane, command, tick) -> None:

		plane.state = PlaneState.AIR
		
		target_runway = command.argument
		if not isinstance(target_runway, Runway):
			raise ValueError("Invalid runway argument: must be a Runway object")
		
		target_hdg = target_runway.hdg

		plane.v_z = 0  # Ensure vertical speed is zero during realignment
		plane.acc_xy = 0  # Ensure horizontal acceleration is zero during realignment
		
		# If not already parallel to the runway...
		if not self.parallel:
			plane._turn(plane.hdg, target_hdg) # Turn towards the runway heading
			if np.isclose(plane.hdg, target_hdg, atol=1e-5):
				self.parallel = True
	
		# If the aircraft is already parallel to the runway...
		else:
			# Double-turn into the runway
			self.init_dist = self.init_dist or utils.point_to_line_distance(
				utils.latlon_to_meters(plane.lat, plane.lon),
				utils.latlon_to_meters(target_runway.get_start_point_ll().latitude, target_runway.get_start_point_ll().longitude),
				utils.latlon_to_meters(target_runway.get_end_point_ll().latitude, target_runway.get_end_point_ll().longitude)
			)
			current_dist = utils.point_to_line_distance(
				utils.latlon_to_meters(plane.lat, plane.lon),
				utils.latlon_to_meters(target_runway.get_start_point_ll().latitude, target_runway.get_start_point_ll().longitude),
				utils.latlon_to_meters(target_runway.get_end_point_ll().latitude, target_runway.get_end_point_ll().longitude)
			)
			self.dir = self.dir or self._get_direction((plane.lon, plane.lat), plane.hdg, target_runway.get_start_point_xy())
			# print(f"Plane {plane.callsign} is realigning to runway {target_runway.name} on tick {tick}, direction {self.dir}, current distance {current_dist:.2f} meters, initial distance {self.init_dist:.2f} meters.")
			if current_dist > self.init_dist / 2 and self.init_dist > 100:
				if self.dir == "left":
					plane._turn(plane.hdg, (target_runway.hdg - 90) % 360)
				elif self.dir == "right":
					plane._turn(plane.hdg, (target_runway.hdg + 90) % 360)
			else:
				plane._turn(plane.hdg, target_hdg)	

			
			if CommandHandler._is_aligned_to_runway(plane, target_runway): # check if already aligned and done
				# If already aligned, switch to cruise mode
				command.command_type = CommandType.CRUISE
				self.__init__()  # Reset state for next realignment

				print(f"{plane.callsign} is now aligned with runway {target_runway.name}")

				return
			
	@staticmethod
	def _get_direction(pos, heading, target_pos):
		"""Determine the direction to the target position relative to the plane's heading.
		Args:
			pos (tuple): The current position of the plane (x, y).
			heading_angle_rad (float): The current heading angle of the plane in radians.
			target_pos (tuple): The target position (x, y) to check the direction towards.
		Returns:
			str: The direction to the target position ("left", "right", or "straight ahead (or behind)").
		"""

		heading = utils.heading_angle_to_unit_vector(heading)
		to_target = (target_pos[0] - pos[0], target_pos[1] - pos[1])

		# 2D cross product
		cross = heading[0] * to_target[1] - heading[1] * to_target[0]
		if cross > 0:
			return "left"
		elif cross < 0:
			return "right"
		else:
			return "straight ahead"
			
	

class NoCommandHandler(CommandHandler):
	"""Handler for when no command is active."""
	
	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.NONE
	
	def execute(self, plane, command, tick) -> None:
		return

class CruiseCommandHandler(CommandHandler):
	"""Handler for plane cruise mode."""

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.CRUISE
	
	def execute(self, plane, command, tick) -> None:

		if 304.8 <= plane.alt <= 457.2: # 1000-1500 ft
			v_z_target = 0.0

			plane.v_z = plane.proportional_change(
				current=plane.v_z,
				target=v_z_target,
				min_value=-plane.dsc_rate,
				max_value=plane.asc_rate,
				max_change=plane.acc_z_max
			)

			plane.state = PlaneState.WAITING_FOR_LANDING
		else:
			plane.state = PlaneState.AIR

		if plane.alt < 304.8 or plane.alt > 457.2: # <1000 or >1500 ft
		# Try to achieve a vertrate proportional to the altitudinal error
			alt_error = plane.crz_alt - plane.alt
			v_z_target = max(-plane.dsc_rate, min(plane.asc_rate, alt_error * 0.1))

			plane.v_z = plane.proportional_change(
				current=plane.v_z,
				target=v_z_target,
				min_value=-plane.dsc_rate,
				max_value=plane.asc_rate,
				max_change=plane.acc_z_max
			)

		# Try to achieve a lateral acceleration proportional to the gspd error
		gspd_error = plane.crz_speed - plane.gspd
		gspd_target = max(-plane.acc_xy_max, min(plane.acc_xy_max, gspd_error * 0.1))

		# If the plane is NOT descending...
		if plane.v_z > -plane.dsc_rate / 2:
			# Turn on the speed controller
			plane.acc_xy = plane.proportional_change(
				current=plane.acc_xy,
				target=gspd_target,
				min_value=-plane.acc_xy_max,
				max_value=plane.acc_xy_max,
				max_change=plane.acc_xy_max
			)

class TurnCommandHandler(CommandHandler):
	"""Handler for turn commands."""
	
	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.TURN
	
	def execute(self, plane, command, tick) -> None:

		if abs(plane.alt - plane.crz_alt) < 20:
			plane.state = PlaneState.WAITING_FOR_LANDING
		else:
			plane.state = PlaneState.AIR

		current_hdg = plane.hdg
		desired_hdg = command.argument

		if desired_hdg is None or not (0 <= desired_hdg < 360):
			raise ValueError("TURN command missing valid heading.")

		plane._turn(current_hdg, desired_hdg)
		if plane.hdg == desired_hdg:
			command.command_type = CommandType.CRUISE
			
class LineUpAndWaitCommandHandler(CommandHandler):
	"""Handler for line up and wait for takeoff commands."""
	
	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.LINE_UP_AND_WAIT
	
	def execute(self, plane, command, tick) -> None:

		target_runway = command.argument
		if not isinstance(target_runway, Runway):
			raise ValueError("Invalid runway argument: must be a Runway object")

		plane.hdg = target_runway.hdg
		plane.lon = target_runway.get_start_point_xy()[0]
		plane.lat = target_runway.get_start_point_xy()[1]

		plane.state = PlaneState.WAITING_FOR_TAKEOFF

class LandingCommandHandler(CommandHandler):
	"""Handler for landing commands."""

	def __init__(self):
		self.tod = None  # Top of descent in nautical miles
		self.rod = None  # Rate of descent in feet per minute
		self.command_was_valid = False  # Flag to track if the command was valid
		self.has_snapped = False  # Flag to track if the plane has snapped to the runway position

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.CLEARED_TO_LAND
	
	@staticmethod
	def is_valid_command(command, plane):
		tod = plane._calculate_tod(plane.alt)
		current_dist = LandingCommandHandler._calculate_runway_distance(plane, command.argument)
		return current_dist >= tod and current_dist > 0
	
	@staticmethod
	def is_aligned(plane, command):
		target_runway = command.argument
		if not CommandHandler._is_aligned_to_runway(plane, target_runway):
			return False
		return True

	def execute(self, plane, command, tick) -> None:

		target_runway = command.argument
		
		plane.hdg = target_runway.hdg # Ensure the plane is aligned to the runway heading

		# Initialize landing parameters if needed
		if not hasattr(self, 'tod') or self.tod is None or not hasattr(self, 'rod') or self.rod is None or not hasattr(plane, 'desired_acc_xy') or plane.desired_acc_xy is None:
			self._initialize_landing(plane, target_runway, command) # TODO: STOP ADDING RANDOM ATTRIBUTES TO THE PLANE OBJECT

		target_dist = self._calculate_runway_distance(plane, target_runway)

		# DOES NOT ACTIVATE
		# if not self.command_was_valid:
		# 	self.command_was_valid = self.is_valid_command(command, plane)
		# 	if not self.command_was_valid:
		# 		self.__init__()  # Reset state for next landing
		# 		command.last_update = tick
		# 		command.command_type = CommandType.GO_AROUND
		# 		print(f"{plane.callsign} is going around due to invalid landing command at tick {tick}.")
		# 		return
	
		if target_dist < self.tod and plane.alt > 0:
			self._handle_descent_phase(plane, target_dist, target_runway)
		elif plane.alt <= 0 and plane.gspd > 0:
			self._handle_ground_phase(plane)
		elif plane.gspd <= 0:
			self._handle_landing_complete(plane, command, tick)
			self.__init__()  # Reset state for next landing
			plane.desired_acc_xy = None  # Reset desired horizontal acceleration after landing
		else:
			plane.v_z = 0  # Maintain vertical speed at zero if already on the ground
			plane.acc_xy = 0  # Maintain horizontal acceleration at zero if already on the ground
		

	def _initialize_landing(self, plane: Plane, target_runway: Runway, command: Command) -> None:
		"""Initialize landing parameters (extends base runway alignment)."""
		# Add landing-specific initialization
		
		self.tod = plane._calculate_tod(plane.alt)
		self.rod = plane._calculate_rod(plane.gspd)
		plane.desired_acc_xy = plane._calculate_target_acc_descend(plane.gspd, plane.alt)

	def _handle_descent_phase(self, plane: Plane, target_dist: float, target_runway: Runway) -> None:
		"""Handle the descent phase of landing."""
		
		# if target_dist < 1:
		# 	descent_rate = -(plane.alt * plane.gspd / target_dist)
		# else:
		descent_rate = -self.rod
		
		plane.v_z = descent_rate
		# plane.proportional_change(
		# 	current=plane.v_z,
		# 	target=descent_rate,
		# 	min_value=-plane.dsc_rate,
		# 	max_value=plane.asc_rate,
		# 	max_change=plane.acc_z_max
		# )
		
		# Handle speed control during descent
		if plane.gspd > plane.ldg_speed:
			plane.acc_xy = plane.proportional_change(
				current=plane.acc_xy,
				target=plane.desired_acc_xy,
				min_value=-plane.acc_xy_max,
				max_value=plane.acc_xy_max,
				max_change=plane.acc_xy_max
			)
		else:
			plane.acc_xy = plane.proportional_change(
				current=plane.acc_xy,
				target=0,
				min_value=-plane.acc_xy_max,
				max_value=plane.acc_xy_max,
				max_change=plane.acc_xy_max
			)

		if not self.has_snapped and target_dist < 0.1:
			# Snap to the runway position if within 0.1 nautical miles
			plane.lon = target_runway.get_start_point_xy()[0]
			plane.lat = target_runway.get_start_point_xy()[1]
			plane.hdg = target_runway.hdg
			self.has_snapped = True
			print(f"{plane.callsign} has snapped to runway {target_runway.name} at ({plane.lat}, {plane.lon})")

	def _handle_ground_phase(self, plane: Plane) -> None:
		"""Handle the ground phase after touchdown."""
		plane.v_z = 0
		plane.acc_xy = -plane.acc_xy_max
	
	def _handle_landing_complete(self, plane: Plane, command: Command, tick: int) -> None:
		"""Handle landing completion."""

		plane.state = PlaneState.LANDING
		plane.landed_this_tick = True

		plane.acc_xy = 0
		command.command_type = CommandType.TAXI
		command.last_update = tick

class TaxiCommandHandler(CommandHandler):
	"""Command handler for post-landing operations."""

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.TAXI

	def execute(self, plane, command, tick) -> None:
		plane.state = PlaneState.TAXIING
		if plane.time_waited == 0:
			print(f"{plane.callsign} taxiing off runway")
		if plane.time_waited == 90:
			plane.state = PlaneState.MARKED_FOR_DELETION
			print(f"{plane.callsign} exited runway")
		else:
			plane.time_waited += 1

class TakeoffCommandHandler(CommandHandler):
	"""Handler for takeoff commands."""

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.CLEARED_FOR_TAKEOFF
	
	def execute(self, plane, command, tick) -> None:

		# Ground roll -> accelerate to minimum takeoff speed
		if plane.gspd < plane.stall_speed:
			plane.acc_xy = plane.proportional_change(
				current=plane.acc_xy,
				target=plane.acc_xy_max,
				min_value=0,
				max_value=plane.acc_xy_max,
				max_change=plane.acc_xy_max
			)
			
		# Climb to cruising alt/speed
		else:
			plane.state = PlaneState.TAKINGOFF
			plane.tookoff_this_tick = True
			command.command_type = CommandType.CRUISE



#TODO: change go around to keep continuing straight and maintaining altitude
class GoAroundCommandHandler(CommandHandler):
	"""Handler for go-around commands."""

	def __init__(self):
		self.init_hdg = None

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.GO_AROUND

	def execute(self, plane, command, tick) -> None:

		plane.has_gone_around = True

		if abs(plane.alt - plane.crz_alt) < 20:
			plane.state = PlaneState.WAITING_FOR_LANDING
		else:
			plane.state = PlaneState.AIR

		if self.init_hdg is None:
			self.init_hdg = plane.hdg
		# Go-around procedure: climb and return to pattern
		if plane.alt < plane.crz_alt:  # Climb to pattern altitude
			plane.v_z = plane.proportional_change(
				current=plane.v_z,
				target=plane.asc_rate,  # Climb at max rate
				min_value=-plane.dsc_rate,
				max_value=plane.asc_rate,
				max_change=plane.acc_z_max
		)
		elif tick >= command.last_update and plane.hdg != self.init_hdg:  # Turn back to initial heading
			plane._turn(plane.hdg, self.init_hdg)
		elif plane.hdg == self.init_hdg and tick >= command.last_update:
			# If already back to initial heading, just cruise
			print(f"Plane {plane.callsign} has gone around and is following missed approach procedure")
			command.command_type = CommandType.CRUISE
			plane.acc_xy = 0
			self.init_hdg = None


class CommandProcessor:
	"""Main command processor that delegates to appropriate handlers."""
	
	def __init__(self):
		self.handlers = [
			NoCommandHandler(),
			TurnCommandHandler(),
			LineUpAndWaitCommandHandler(),
			LandingCommandHandler(),
			TakeoffCommandHandler(),
			GoAroundCommandHandler(),
			CruiseCommandHandler(),
			RealignCommandHandler(),
			TaxiCommandHandler()
		]
	
	def process_command(self, plane, command, tick):
		"""Process a command using the appropriate handler."""
		command_type = CommandType.NONE if command is None else command.command_type
		
		for handler in self.handlers:
			if handler.can_handle(command_type):
				handler.execute(plane, command, tick)
				return

		raise NotImplementedError(f"Unknown command type for plane {plane.callsign}: {command_type}")

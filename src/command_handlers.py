# Command handler classes for better separation of concerns
from abc import ABC, abstractmethod
import math

import geopy
import numpy as np
from commands import CommandType
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
	def execute(self, plane, command, tick) -> None:
		"""Execute the command logic for the given plane."""
		pass
	
	# Shared runway alignment functionality
	def _validate_runway_command(self, target_runway, command, plane):
		"""Validate runway-related command parameters."""
		if not isinstance(target_runway, Runway):
			raise TypeError("Argument must be a Runway object.")
		if command.last_update is None or not isinstance(command.last_update, int):
			raise ValueError("Command last_update must be an integer tick value.")
		if plane.turn_start_time is None or not isinstance(plane.turn_start_time, int): # TODO: don't go around adding random attributes to the plane object
			plane.turn_start_time = -1
	
	def _calculate_runway_distance(self, plane, target_runway):
		"""Calculate distance to runway in nautical miles."""
		return utils.degrees_to_nautical_miles(
			heading=plane.hdg, 
			degrees=target_runway.get_line_xy().distance(
				shapely.geometry.Point((plane.lon, plane.lat))
			)
		)
	
	def _initialize_runway_alignment(self, plane, target_runway, command):
		"""Initialize runway alignment parameters."""
		plane.turn_start_time = plane.find_turn_initiation_time(target_runway.get_line_xy(), command.last_update)
	
	def _handle_runway_alignment(self, plane, target_hdg, tick, command):
		"""Handle the runway alignment phase."""
		plane._turn(plane.hdg, target_hdg)
		if math.isclose(plane.hdg, target_hdg, abs_tol=1e-5):
			command.last_update = tick
			return True  # Alignment complete
		return False  # Still aligning
	
	def _is_aligned_to_runway(self, plane, target_runway):
		"""Check if the plane is aligned with the target runway."""
		return utils.is_collinear(
			shapely.geometry.LineString([(plane.lon, plane.lat), (plane.next_pt.lon, plane.next_pt.lat)]),
			target_runway.get_line_xy()
		)
	

class RealignCommandHandler(CommandHandler):
	"""Handler for realigning planes to the runway centerline."""

	def __init__(self):
		self.init_dist = None
		self.dir = None  # Direction to turn: "left", "right", or "straight ahead"

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.REALIGN
	
	def execute(self, plane, command, tick) -> None:

		plane.state = PlaneState.AIR

		target_runway = command.argument
		target_hdg = target_runway.hdg
		
		# Validation using shared method
		self._validate_runway_command(target_runway, command, plane)
		
		if abs(target_hdg - plane.hdg) == np.isclose(-0.01, 0.01):
			self.init_dist = self.init_dist if self.init_dist is not None else utils.point_to_great_circle_distance(
				geopy.Point(plane.lat, plane.lon),
				target_runway.get_start_point_ll(),
				target_runway.get_end_point_ll()
			)
			current_dist = utils.point_to_great_circle_distance(
				geopy.Point(plane.lat, plane.lon),
				target_runway.get_start_point_ll(),
				target_runway.get_end_point_ll()
			)
			self.dir = self.dir if self.dir is not None else self._get_direction((plane.lon, plane.lat), plane.hdg, target_runway.get_start_point_xy())
			
			print(f"Plane {plane.callsign} is realigning to runway {target_runway.name if hasattr(target_runway, 'name') else 'unknown'} at tick {tick}. Direction: {self.dir}, Current Distance: {current_dist}, Initial Distance: {self.init_dist}")
			
			if self.dir == "left":
				if current_dist > self.init_dist / 2:
					plane._turn(plane.hdg, (plane.hdg - 90) % 360)
				else:
					plane._turn(plane.hdg, target_hdg)
			elif self.dir == "right":
				if current_dist > self.init_dist / 2:
					plane._turn(plane.hdg, (plane.hdg + 90) % 360)
				else:
					plane._turn(plane.hdg, target_hdg)
			else:
				self.init_dist = None  # Reset initial distance if already aligned
				plane.dir = None  # Reset direction
				command.command_type = CommandType.CRUISE  # If already aligned, switch to cruise mode
		else:		
			# Initialize alignment if needed
			if plane.turn_start_time == -1:
				self._initialize_runway_alignment(plane, target_runway, command)
			
			# Execute runway alignment
			elif tick >= plane.turn_start_time and command.last_update < plane.turn_start_time:
				alignment_complete = self._handle_runway_alignment(plane, target_hdg, tick, command)
				if alignment_complete:
					print(f"Plane {plane.callsign} lined up to runway {target_runway.name if hasattr(target_runway, 'name') else 'unknown'} and waiting.")
			
			# If already aligned, just maintain position and heading
			elif command.last_update >= plane.turn_start_time:
				# Maintain runway heading and zero speed
				plane._turn(plane.hdg, target_hdg)
				plane.acc_xy = plane.proportional_change(
					current=plane.acc_xy,
					target=0,  # Stop acceleration
					min_value=-plane.acc_xy_max,
					max_value=plane.acc_xy_max,
					max_change=plane.acc_xy_max
				)


	@staticmethod
	def _get_direction(pos, heading_angle_rad, target_pos):
		"""Determine the direction to the target position relative to the plane's heading.
		Args:
			pos (tuple): The current position of the plane (x, y).
			heading_angle_rad (float): The current heading angle of the plane in radians.
			target_pos (tuple): The target position (x, y) to check the direction towards.
		Returns:
			str: The direction to the target position ("left", "right", or "straight ahead (or behind)").
		"""

		heading = utils.heading_angle_to_unit_vector(heading_angle_rad)
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
		plane.state = PlaneState.GROUND
		return None

class CruiseCommandHandler(CommandHandler):
	"""Handler for plane cruise mode."""

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.CRUISE
	
	def execute(self, plane, command, tick) -> None:

		if abs(plane.alt - plane.crz_alt) < 5:
			plane.state = PlaneState.WAITING_FOR_LANDING
		else:
			plane.state = PlaneState.AIR

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

		if abs(plane.alt - plane.crz_alt) < 5:
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
		
		# Validation using shared method
		self._validate_runway_command(target_runway, command, plane)

		plane.hdg = target_runway.hdg
		plane.lon = target_runway.get_start_point_xy()[0]
		plane.lat = target_runway.get_start_point_xy()[1]

		plane.state = PlaneState.WAITING_FOR_TAKEOFF

class LandingCommandHandler(CommandHandler):
	"""Handler for landing commands."""
	
	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.CLEARED_TO_LAND
	
	def execute(self, plane, command, tick) -> None:

		target_runway = command.argument
		
		# Validation using shared method
		self._validate_runway_command(target_runway, command, plane)
		
		target_hdg = target_runway.hdg
		target_dist = self._calculate_runway_distance(plane, target_runway)
		
		# Initialize landing parameters if needed
		if plane.turn_start_time == -1:
			self._initialize_landing(plane, target_runway, command)
		
		# Execute landing phases
		elif tick >= plane.turn_start_time and command.last_update < plane.turn_start_time:
			alignment_complete = self._handle_runway_alignment(plane, target_hdg, tick, command)
			# Continue to descent phase after alignment
		elif command.last_update >= plane.turn_start_time and target_dist < plane.tod and plane.alt > 0:
			self._handle_descent_phase(plane, target_dist)
		elif plane.alt <= 0 and plane.gspd > 0:
			self._handle_ground_phase(plane)
		elif plane.gspd <= 0:
			self._handle_landing_complete(plane, command, tick)
	
	def _initialize_landing(self, plane, target_runway, command):
		"""Initialize landing parameters (extends base runway alignment)."""
		# Use shared runway alignment initialization
		self._initialize_runway_alignment(plane, target_runway, command)
		# Add landing-specific initialization
		plane.tod = plane._calculate_tod(plane.alt)
		plane.rod = plane._calculate_rod(plane.gspd)
		plane.desired_acc_xy = plane._calculate_target_acc_descend(plane.gspd, plane.alt)
	
	def _handle_descent_phase(self, plane, target_dist):
		"""Handle the descent phase of landing."""
		plane.rod = plane._calculate_rod(plane.gspd)
		
		if target_dist < 1:
			descent_rate = -(plane.alt * plane.gspd / target_dist)
		else:
			descent_rate = -plane.rod
		
		plane.v_z = plane.proportional_change(
			current=plane.v_z,
			target=descent_rate,
			min_value=-plane.dsc_rate,
			max_value=plane.asc_rate,
			max_change=plane.acc_z_max
		)
		
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
	
	def _handle_ground_phase(self, plane):
		"""Handle the ground phase after touchdown."""
		plane.v_z = 0
		plane.acc_xy = -plane.acc_xy_max
	
	def _handle_landing_complete(self, plane, command, tick):
		"""Handle landing completion."""

		plane.state = PlaneState.LANDING
		plane.thistick[0] = True

		#print(f"Plane {plane.callsign} with ID {plane.id} has landed.")
		plane.acc_xy = 0
		command.command_type = CommandType.NONE
		command.last_update = tick

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
			
		# Climb until 1000 ft
		else:
			plane.state = PlaneState.TAKINGOFF
			plane.thistick[1] = True

			command.command_type = CommandType.CRUISE

			print(f"Plane {plane.callsign} takeoff complete. Now cruising.")

class GoAroundCommandHandler(CommandHandler):
	"""Handler for go-around commands."""

	def __init__(self):
		self.init_hdg = None
		self.target_hdg = None
		self.has_turned = False

	def can_handle(self, command_type: CommandType) -> bool:
		return command_type == CommandType.GO_AROUND

	def execute(self, plane, command, tick) -> None:

		if abs(plane.alt - plane.crz_alt) < 5:
			plane.state = PlaneState.WAITING_FOR_LANDING
		else:
			plane.state = PlaneState.AIR

		if self.init_hdg is None:
			self.init_hdg = plane.hdg
		if self.target_hdg is None:
			self.target_hdg = (self.init_hdg + 180) % 360
		# Go-around procedure: climb and return to pattern
		if plane.alt < plane.crz_alt:  # Climb to pattern altitude
			plane.v_z = plane.proportional_change(
				current=plane.v_z,
				target=plane.asc_rate,  # Climb at max rate
				min_value=-plane.dsc_rate,
				max_value=plane.asc_rate,
				max_change=plane.acc_z_max
		)
		elif not self.has_turned:  # Turn away from runway
			plane.v_z = 0  # Stop vertical movement
			plane.acc_xy = 0  # Stop horizontal acceleration
			plane._turn(plane.hdg, self.target_hdg)
			if math.isclose(plane.hdg, self.target_hdg, abs_tol=1e-2):
				plane.hdg = self.target_hdg
				command.last_update = tick + 500  # random delay to simulate go-around time
				self.has_turned = True
		elif tick >= command.last_update and self.has_turned and plane.hdg != self.init_hdg:  # Turn back to initial heading
			plane._turn(plane.hdg, self.init_hdg)
		elif plane.hdg == self.init_hdg and tick >= command.last_update:
			# If already back to initial heading, just cruise
			print(f"Plane {plane.callsign} has completed go-around and is returning to pattern.")
			command.command_type = CommandType.CRUISE
			plane.acc_xy = 0
			self.has_turned = False
			self.init_hdg = None
			self.target_hdg = None


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
			RealignCommandHandler()
		]
	
	def process_command(self, plane, command, tick):
		"""Process a command using the appropriate handler."""
		command_type = CommandType.NONE if command is None else command.command_type
		
		for handler in self.handlers:
			if handler.can_handle(command_type):
				handler.execute(plane, command, tick)
				return
		
		raise NotImplementedError(f"Unknown command type: {command_type}")

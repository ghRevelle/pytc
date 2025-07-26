# Flight simulator class
from plane import Plane
from planestates import *
from pygame_display import *
from airport import *
from commands import *
from plane_manager import PlaneManager
from command_handlers import LandingCommandHandler
import time
import utils

class FlightSimulator:
	"""A simple flight simulator to demonstrate plane movement and display."""

	# Simulation speed. In real life, 1 tick = 1 second
	base_tps = 25

	def __init__(self, display_size=(640, 480), airport=None, plane_manager=None, rolling_initial_state=None):
		"""Initialize the flight simulator with a display size, optional airport layout.
		Args:
			display_size (tuple): Size of the display window (width, height).
			airport (Airport): Optional airport layout.
		"""
		self.display_size = display_size

		# Start at tick 0
		self.current_tick = 0

		if airport is None:
			raise TypeError("Missing Airport Layout")

		if plane_manager is None:
			raise TypeError("Missing Plane Manager")
		
		self.plane_manager = plane_manager
		self.plane_manager.set_airport(airport)

		self.rolling_initial_state = rolling_initial_state if rolling_initial_state is not None else []
		for plane_state in self.rolling_initial_state:
			if plane_state['time_added'] == 0:
				self.add_plane_to_manager(plane_state)

		# Initialize the Pygame display
		self.pg_display = Pygame_Display(*display_size)
		# Setup the airport on the display
		self.pg_display.setup_airport(airport)
		# Empty command queue
		self.command_queue = []

		self.crashed_planes = []  # List to keep track of crashed planes

		self.invalid_command_executed = False  # Flag for invalid command execution
		self.valid_command_executed = False  # Flag for valid command execution


	def get_tps(self):
		"""Get the effective ticks per second, accounting for turbo mode."""
		if hasattr(self.pg_display, 'turbo_mode') and self.pg_display.turbo_mode:
			return self.base_tps * 10
		return self.base_tps

	# Add a plane to the plane manager
	def add_plane_to_manager(self, plane: dict):
		"""Add a plane to the simulator."""
		self.plane_manager.add_plane(plane)
		print(f"tick: {self.current_tick}")

	# Delete a plane from the plane manager
	def delete_plane_from_manager(self, id = None, callsign = None):
		if id is not None:
			self.plane_manager.delete_plane(id)
		elif callsign is not None:
			self.plane_manager.delete_plane_by_callsign(callsign)
		else:
			raise ValueError("Must input an ID or callsign for deletion.")

	# Send a command to a plane
	def command_plane(self, command: Command):
		"""Send a command to a specific plane instantly.
		Args:
			command (Command class): The command to send
		"""
		for plane in self.plane_manager.planes:
			if plane.id == command.target_id:
				plane.change_command(command)

	# Add a command to a command queue
	def add_command(self, command: Command):
		"""Add a command to the command queue.
		Args:
			command (Command class): Command to be added to the queue.
		"""
		if not isinstance(command, Command):
			raise ValueError("Command must be a Command object.")
		self.command_queue.append(command)

	# Function to support testing by allowing commands by callsign
	def add_command_by_callsign(self, callsign: str, command_type: CommandType, last_update: int, argument):
		target_id = self.plane_manager.get_id(callsign)
		self.add_command(Command(command_type, target_id, last_update, argument))

	def check_command_validity(self, command: Command):
		"""Check the validity of an executed command."""
		plane = self.plane_manager.planes[command.target_id]
		command_type = command.command_type
		if command_type == CommandType.CLEARED_FOR_TAKEOFF:
			if plane.state != PlaneState.WAITING_FOR_TAKEOFF:
				print(f"Invalid command: {command.command_type} for {plane.callsign}. Expected state: WAITING_FOR_TAKEOFF, was: {plane.state}")
				self.invalid_command_executed = True
		elif command_type == CommandType.CLEARED_TO_LAND:
			if plane.state != PlaneState.WAITING_FOR_LANDING or not LandingCommandHandler.is_valid_command(command, plane):
				print(f"Invalid command: {command.command_type} for {plane.callsign}. Expected state: WAITING_FOR_LANDING, was: {plane.state}")
				self.invalid_command_executed = True
		elif command_type == CommandType.LINE_UP_AND_WAIT:
			if plane.state != PlaneState.QUEUED:
				print(f"Invalid command: {command.command_type} for {plane.callsign}. Expected state: QUEUED, was: {plane.state}")
				self.invalid_command_executed = True
		return

	def tick(self):
		"""Run a single tick of the simulation."""

		for event in pygame.event.get(): # Check for quit events
			if event.type == pygame.QUIT:
				self.pg_display.stop_display()
				self.current_tick = 0
				return
		
		self.invalid_command_executed = False  # Reset invalid command flag for this tick
		self.valid_command_executed = False  # Reset valid command flag for this tick
		for command in self.command_queue:  # Process all commands in the queue
			if self.current_tick == command.last_update:
				self.check_command_validity(command)  # Check if the command is valid
				if not self.invalid_command_executed:
					self.command_plane(command)
					self.print_command(command)  # Print the command for debugging
					self.plane_manager.airport.pop_top_of_queue() if command.command_type == CommandType.LINE_UP_AND_WAIT else None
					self.command_queue.remove(command)  # Remove command after execution
					if 1 <= command.command_type.value <= 5:  # Only reward for valid DRL-issued commands (Enums 1 to 5)
						self.valid_command_executed = True
		
		# Update all plane states using list comprehension
		plane_states = [plane.tick(self.current_tick).get_state() for plane in self.plane_manager.planes]

		# Check all planes for crashes and deletions
		planes = self.plane_manager.planes
		for i, plane1 in enumerate(planes):
			# Delete planes that have flown out of radar
			# Assuming 15 nautical miles is the maximum distance for radar visibility
			if utils.distance_from_base(plane1.lat, plane1.lon) > 27780 and plane1.state != PlaneState.QUEUED: # 15 nautical miles
				plane1.state = PlaneState.MARKED_FOR_DELETION
				print(f"{plane1.callsign} has flown out of radar")
				print(f"tick: {self.current_tick}")

			for plane2 in planes[i+1:]: # avoid checking a plane against itself
				# Skip ground planes to avoid collision detection in airport queue
				if plane1.state == plane2.state == PlaneState.QUEUED:
					continue
				
				check_distance = utils.calculate_craft_distance(plane1.lat, plane1.lon, plane2.lat, plane2.lon, plane1.alt, plane2.alt)
				
				# <= 300 meters is considered a near-collision; the DRL is punished as if the planes crashed
				if check_distance <= 300 and check_distance > 30:
					plane1.crashed_this_tick = plane2.crashed_this_tick = True
					print(f"{plane1.callsign} and {plane2.callsign} had a close call at tick {self.current_tick}.")
					
				# <= 30 meters is considered a collision; the DRL is punished, and the planes get destroyed
				elif check_distance <= 30:
					self.crashed_planes.extend([plane1, plane2])
					print(f"{plane1.callsign} and {plane2.callsign} crashed at tick {self.current_tick}.")
			
			if plane1.state == PlaneState.MARKED_FOR_DELETION:
				self.plane_manager.delete_plane(plane1.id)

		# Update display once with all plane states
		self.pg_display.update_display(plane_states)

		reward = self.compute_reward()  # Compute the reward for this tick
		if abs(reward) > 0.1:
			print(f"Reward for tick {self.current_tick}: {reward}")

		while self.crashed_planes:
			self.plane_manager.delete_plane(self.crashed_planes.pop().id)

		# Reset plane flags using list comprehension
		[setattr(plane, attr, False) for plane in self.plane_manager.planes 
		 for attr in ['landed_this_tick', 'tookoff_this_tick', 'crashed_this_tick']]
			
		self.current_tick += 1  # Increment the tick count

		# Check for planes to add to state
		for plane_state in self.rolling_initial_state:
			if plane_state['time_added'] == self.current_tick:
				self.add_plane_to_manager(plane_state)

		# Check for end of simulation
		if self.plane_manager.check_end_state():
			print(f"Ending simulation at tick {self.current_tick}")
			self.pg_display.stop_display()
			self.current_tick = 0
			return

		effective_tps = self.get_tps()
		time.sleep(1 / effective_tps)  # Control the simulation speed with turbo mode
			
	def compute_reward(self):
		reward = 0.0

		for plane in self.plane_manager.planes:
			# Reward for successful landings
			if plane.landed_this_tick == True:
				print(f"{plane.callsign} has landed")
				reward += 50.0

			# Reward for successful takeoff
			if plane.tookoff_this_tick == True:
				print(f"{plane.callsign} has taken off")
				reward += 50.0

			# Penalty for crashing
			if plane.crashed_this_tick == True:
				print(f"{plane.callsign} just crashed")
				reward -= 1000.0

    	# Penalty for invalid or illegal commands
		if self.invalid_command_executed:
			reward -= 10.0

		# Reward for valid command execution
		# This is to encourage the DRL to issue valid commands
		if self.valid_command_executed:
			reward += 5.0

    	# Small time pressure penalty per plane still in air
		reward -= 0.01 * len(self.plane_manager.planes)

		return reward

	def print_command(self, command: Command):
		if command.command_type == CommandType.CLEARED_FOR_TAKEOFF:
			print(f"{self.plane_manager.get_callsign(command.target_id)} cleared for takeoff on runway {command.argument.name}")
		elif command.command_type == CommandType.CLEARED_TO_LAND:
			print(f"{self.plane_manager.get_callsign(command.target_id)} cleared to land on runway {command.argument.name}")
		elif command.command_type == CommandType.LINE_UP_AND_WAIT:
			print(f"{self.plane_manager.get_callsign(command.target_id)} lined up on runway {command.argument.name}")
		elif command.command_type == CommandType.REALIGN:
			print(f"{self.plane_manager.get_callsign(command.target_id)} realigning to runway {command.argument.name}")
		elif command.command_type == CommandType.GO_AROUND:
			print(f"{self.plane_manager.get_callsign(command.target_id)} going around")
		elif command.command_type == CommandType.ABORT_TAKEOFF:
			print(f"{self.plane_manager.get_callsign(command.target_id)} aborting takeoff")
		print(f"tick: {self.current_tick}")
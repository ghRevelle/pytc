# Flight simulator class
from plane import Plane
from pygame_display import *
from airport import *
from commands import *
from plane_manager import PlaneManager
import time

class FlightSimulator:
	"""A simple flight simulator to demonstrate plane movement and display."""

	# Simulation speed. In real life, 1 tick = 1 second
	ticks_per_second = 20

	def __init__(self, display_size=(640, 480), planes=None, airport=None, plane_manager=None):
		"""Initialize the flight simulator with a display size, optional planes, optional airport layout.
		Args:
			display_size (tuple): Size of the display window (width, height).
			planes (list): Optional list of Plane objects to simulate.

		"""

		if plane_manager is None:
			raise TypeError("Missing Plane Manager")

		# Initialize the airport layout
		self.airport = airport
		# Initialize the Pygame display
		self.pg_display = Pygame_Display(*display_size)
		# Setup the airport on the display
		self.pg_display.setup_airport(self.airport)
		# Empty command queue
		self.command_queue = []
		# Start at tick 0
		self.tick = 0
		# Initialize the slot manager
		self.plane_manager = plane_manager

	# Add a plane to the plane manager
	def add_plane_to_manager(self, plane: dict):
		"""Add a plane to the simulator."""
		self.plane_manager.add_plane(plane)

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
	def add_command_by_callsign(self, callsign: str, command_type: CommandType, last_update: int, argument: Optional[int]):
		target_id = self.plane_manager.get_id(callsign)
		self.add_command(Command(command_type, target_id, last_update, argument))

	# Advance the simulator by 1 tick
	def next_tick(self):
		for event in pygame.event.get(): # Check for quit events
			if event.type == pygame.QUIT:
				self.pg_display.stop_display()
				return
		for command in self.command_queue:  # Process all commands in the queue
			if self.tick == command.last_update:
				self.command_plane(command)
		# Update all plane states
		plane_states = []
		for plane in self.plane_manager.planes:
			plane.tick()
			plane_states.append(plane.get_state())
			
		# Update display once with all plane states
		if plane_states:
			self.pg_display.update_display(plane_states)

		self.tick += 1
		time.sleep(1 / self.ticks_per_second)  # Control the simulation speed

	# Run the simulator for a number of ticks
	def run(self, ticks=500):
		"""Run the flight simulation for a specified number of ticks."""
		while self.tick < ticks:
			for event in pygame.event.get(): # Check for quit events
				if event.type == pygame.QUIT:
					self.pg_display.stop_display()
					self.tick = 0
					return
			for command in self.command_queue:  # Process all commands in the queue
				if self.tick == command.last_update:
					self.command_plane(command)
			# Update all plane states
			plane_states = []
			for plane in self.plane_manager.planes:
				plane.tick()
				plane_states.append(plane.get_state())
			
			# Update display once with all plane states
			if plane_states:
				self.pg_display.update_display(plane_states)

			time.sleep(1 / self.ticks_per_second)  # Control the simulation speed
			self.tick += 1  # Increment the tick count
		self.tick = 0  # Reset tick after running
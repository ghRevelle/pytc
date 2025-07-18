# Flight simulator class
from plane import Plane
from pygame_display import *
from airport import *
from commands import *
from slot_manager import FixedSlotPlaneManager
import time

class FlightSimulator:
	"""A simple flight simulator to demonstrate plane movement and display."""
	# TODO: remember to deal with skipped ticks
	ticks_per_second = 20
	"""Number of ticks per second for the simulation."""

	def __init__(self, display_size=(640, 480), planes=None, airport=None):
		"""Initialize the flight simulator with a display size, optional planes, optional airport layout.
		Args:
			display_size (tuple): Size of the display window (width, height).
			planes (list): Optional list of Plane objects to simulate.

		"""
		# Initialize the list of planes
		self.planes = planes if planes is not None else []
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
		self.slot_manager = FixedSlotPlaneManager()

	def add_plane(self, plane: Plane):
		"""Add a plane to the simulator."""
		self.planes.append(plane)

	def command_plane(self, command: dict):
		"""Send a command to a specific plane instantly.
		Args:
			command (dict): The command to send (e.g., 'turn').
		"""
		for plane in self.planes:
			if plane.id == command.target_id:
				plane.change_command(command)

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
		target_id = self.slot_manager.get_slot(callsign)
		self.add_command(Command(command_type, target_id, last_update, argument))

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
			for plane in self.planes:
				plane.tick()
				plane_states.append(plane.get_state())
			
			# Update display once with all plane states
			if plane_states:
				self.pg_display.update_display(plane_states)

			time.sleep(1 / self.ticks_per_second)  # Control the simulation speed
			self.tick += 1  # Increment the tick count
		self.tick = 0  # Reset tick after running
from plane import Plane
from pygame_display import *
import time

class FlightSimulator:
	"""A simple flight simulator to demonstrate plane movement and display."""
	# TODO: remember to deal with skipped ticks
	tick = 0
	"""Current tick of the simulation."""
	ticks_per_second = 60
	"""Number of ticks per second for the simulation."""
	command_queue = []
	
	def __init__(self, display_size=(640, 480), planes=None):
		"""Initialize the flight simulator with a display size and optional planes.
		Args:
			display_size (tuple): Size of the display window (width, height).
			planes (list): Optional list of Plane objects to simulate.
		"""
		# Initialize the list of planes
		self.planes = planes if planes is not None else []
		# Initialize the Pygame display
		self.pg_display = Pygame_Display(*display_size)

	def add_plane(self, plane: Plane):
		"""Add a plane to the simulator."""
		self.planes.append(plane)

	def command_plane(self, command: dict):
		"""Send a command to a specific plane instantly.
		Args:
			command (dict): The command to send (e.g., 'turn').
		"""
		for plane in self.planes:
			if plane.get_state()['id'] == command['id']:
				plane.change_command(command)
	def add_command(self, command: dict):
		"""Add a command to the command queue.
		Args:
			command (dict): Command to be added to the queue.
		"""
		if 'id' not in command or 'cmd' not in command or 'args' not in command or 'last_updated' not in command:
			raise ValueError("Command must contain 'id', 'cmd', 'args', and 'last_updated' keys.")
		self.command_queue.append(command)

	def run(self, ticks=500):
		"""Run the flight simulation for a specified number of ticks."""
		
		while self.tick < ticks:
			for event in pygame.event.get(): # Check for quit events
				if event.type == pygame.QUIT:
					self.pg_display.stop_display()
					self.tick = 0
					return
			for command in self.command_queue:  # Process all commands in the queue
				if self.tick == command['last_updated']:
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
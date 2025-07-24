# Flight simulator class
from plane import Plane
from planestates import *
from pygame_display import *
from airport import *
from commands import *
from plane_manager import PlaneManager
import time

class FlightSimulator:
	"""A simple flight simulator to demonstrate plane movement and display."""

	# Simulation speed. In real life, 1 tick = 1 second
	base_tps = 25
	current_tick = 0

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
		self.current_tick = 0
		# Initialize the slot manager
		self.plane_manager = plane_manager

	def get_tps(self):
		"""Get the effective ticks per second, accounting for turbo mode."""
		if hasattr(self.pg_display, 'turbo_mode') and self.pg_display.turbo_mode:
			return self.base_tps * 10
		return self.base_tps

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
	def add_command_by_callsign(self, callsign: str, command_type: CommandType, last_update: int, argument):
		target_id = self.plane_manager.get_id(callsign)
		self.add_command(Command(command_type, target_id, last_update, argument))

	# Run the simulator for a number of ticks
	def run(self, ticks=500):
		"""Run the flight simulation for a specified number of ticks."""
		while self.current_tick < ticks:
			if self.current_tick == 200:
				self.plane_manager.delete_plane_by_callsign('UA4')  # Example of deleting a plane at tick 200
			elif self.current_tick == 300:
				self.plane_manager.add_plane({ # Example of adding a new plane at tick 300
					'callsign': 'UA7',
					'lat': 0.01,
					'lon': 0.01,
					'alt': 5000,
					'v_z': 0,
					'gspd': 50,
					'hdg': 90
				})
			self.tick()
			effective_tps = self.get_tps()
			time.sleep(1 / effective_tps)  # Control the simulation speed with turbo mode
			
		self.current_tick = 0  # Reset tick after running

	def tick(self):
		"""Run a single tick of the simulation."""
		# Initialize the list of crashed planes
		crashed_planes = []

		for plane in crashed_planes:
			self.plane_manager.delete_plane(plane.id)

		for event in pygame.event.get(): # Check for quit events
			if event.type == pygame.QUIT:
				self.pg_display.stop_display()
				self.current_tick = 0
				return
		for command in self.command_queue:  # Process all commands in the queue
			if self.current_tick == command.last_update:
				self.command_plane(command)
		# Update all plane states
		plane_states = []
		for plane in self.plane_manager.planes:
			plane.tick(self.current_tick)
			plane_states.append(plane.get_state())

		# Check all planes for crashes
		for i in range(0, len(self.plane_manager.planes)):
			for j in range(0, len(self.plane_manager.planes)):
				if self.plane_manager.planes[i] != self.plane_manager.planes[j]:
					plane1 = self.plane_manager.planes[i]
					plane2 = self.plane_manager.planes[j]

					# does not count close planes on the ground as being a collision or near-collision
					if plane1.state == PlaneState.GROUND and plane2.state == PlaneState.GROUND:
						continue
					
					check_distance = utils.calculate_craft_distance(plane1.lat, plane1.lon, plane2.lat, plane2.lon, plane1.alt, plane2.alt)

					if check_distance <= 300:
						plane1.thistick[2] = True
						plane2.thistick[2] = True
					
					elif check_distance <= 30:
						plane1.thistick[2] = True
						plane2.thistick[2] = True

						crashed_planes.append(plane1)
						crashed_planes.append(plane2)
		
		# Update display once with all plane states
		if plane_states:
			self.pg_display.update_display(plane_states)
		
		#print(compute_reward(self, self))
			
		self.current_tick += 1  # Increment the tick count
	

	# Pseudocode implementation of a reward function for reinforcement learning
	def compute_reward(self, env_state, command_executed, sim_tick):
		reward = 0.0

		for plane in env_state.planes:
			# Reward for successful landings
			if plane.thistick[0] == True:
				print(f"A plane just landed.")
				reward += 10.0

			# Reward for successful takeoff
			if plane.thistick[1] == True:
				print(f"A plane just took off.")
				reward += 10.0

    	# Penalty for invalid or illegal commands
		if command_executed.is_invalid:
			reward -= 10.0

		if command_executed.caused_conflict:
			reward -= 50.0

    	# Penalty for a crash
		for plane in env_state.planes:
			if plane.thistick[2] == True:
				print(f"Plane with ID {plane.id} just crashed.")
				reward -= 100.0
			
    	# Small time pressure penalty per plane still in air
		reward -= 0.01 * env_state.num_planes_in_air

		return reward

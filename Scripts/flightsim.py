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

	def __init__(self, display_size=(640, 480), airport=None, plane_manager=None, rolling_initial_state=None):
		"""Initialize the flight simulator with a display size, optional airport layout.
		Args:
			display_size (tuple): Size of the display window (width, height).
			airport (Airport): Optional airport layout.
		"""
		self.display_size = display_size
		self.airport = airport
		self.plane_manager = plane_manager

		self.rolling_initial_state = rolling_initial_state if rolling_initial_state is not None else []
		for plane_state in self.rolling_initial_state:
			if plane_state['time_added'] == 0:
				self.add_plane_to_manager(plane_state)

		if airport is None:
			raise TypeError("Missing Airport Layout")

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

		self.crashed_planes = []  # List to keep track of crashed planes

	def pass_airport_to_pm(self, airport):
		self.plane_manager.set_airport(airport)

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
			self.tick()
			
		self.current_tick = 0  # Reset tick after running

	def tick(self):
		"""Run a single tick of the simulation."""

		for event in pygame.event.get(): # Check for quit events
			if event.type == pygame.QUIT:
				self.pg_display.stop_display()
				self.current_tick = 0
				return
		for command in self.command_queue:  # Process all commands in the queue
			if self.current_tick == command.last_update:
				self.command_plane(command)
				self.command_queue.remove(command)  # Remove command after execution
		
		# Update all plane states using list comprehension
		plane_states = [plane.tick(self.current_tick).get_state() for plane in self.plane_manager.planes]

		# Check all planes for crashes and deletions
		planes = self.plane_manager.planes
		for i, plane1 in enumerate(planes):
			for plane2 in planes[i+1:]: # avoid checking a plane against itself
				# Skip ground planes to avoid collision detection in airport queue
				if plane1.state == plane2.state == PlaneState.QUEUED:
					continue
				
				check_distance = utils.calculate_craft_distance(plane1.lat, plane1.lon, plane2.lat, plane2.lon, plane1.alt, plane2.alt)
				
				# <= 300 meters is considered a near-collision; the DRL is punished as if the planes crashed
				if check_distance <= 300:
					plane1.crashed_this_tick = plane2.crashed_this_tick = True
					
					# <= 30 meters is considered a collision; the DRL is punished, and the planes get destroyed
					if check_distance <= 30:
						self.crashed_planes.extend([plane1, plane2])
			
			if plane1.state == PlaneState.MARKED_FOR_DELETION:
				self.plane_manager.delete_plane(plane1.id)

		# Update display once with all plane states
		if plane_states:
			self.pg_display.update_display(plane_states)

		while self.crashed_planes:
			self.plane_manager.delete_plane(self.crashed_planes.pop().id)
		
		#print(compute_reward(self, self))

		# Reset plane flags using list comprehension
		[setattr(plane, attr, False) for plane in self.plane_manager.planes 
		 for attr in ['landed_this_tick', 'tookoff_this_tick', 'crashed_this_tick']]
			
		self.current_tick += 1  # Increment the tick count

		# Check for planes to add to state
		for plane_state in self.rolling_initial_state:
			if plane_state['time_added'] == self.current_tick:
				self.add_plane_to_manager(plane_state)

		effective_tps = self.get_tps()
		time.sleep(1 / effective_tps)  # Control the simulation speed with turbo mode
			
	

	# Pseudocode implementation of a reward function for reinforcement learning
	def compute_reward(self, env_state, command_executed, sim_tick):
		reward = 0.0

		for plane in env_state.planes:
			# Reward for successful landings
			if plane.landed_this_tick == True:
				print(f"{plane.callsign} has landed. (ID: {plane.id})")
				reward += 10.0

			# Reward for successful takeoff
			if plane.tookoff_this_tick == True:
				print(f"{plane.callsign} has taken off. (ID: {plane.id})")
				reward += 10.0

			# Penalty for crashing
			if plane.crashed_this_tick == True:
				print(f"{plane.callsign} just crashed. (ID: {plane.id})")
				reward -= 100.0

    	# Penalty for invalid or illegal commands
		if command_executed.is_invalid:
			reward -= 10.0

		if command_executed.caused_conflict:
			reward -= 50.0
			
    	# Small time pressure penalty per plane still in air
		reward -= 0.01 * env_state.num_planes_in_air

		return reward

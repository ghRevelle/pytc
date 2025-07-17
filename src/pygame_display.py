import pygame
import numpy as np
from geopy import units, distance

class Pygame_Display:	
	"""A class to handle the Pygame display for the plane simulation."""
	plane_colors = {}
	trails = {}
	last_states = {}
	debug_labels = True
	trail_length = 100  # Maximum length of the trail in number of points
	def __init__(self, w=1280, h=720):
		pygame.init()
		self.w = w
		self.h = h
		self.x_c = self.w // 2
		self.y_c = self.h // 2
		self.lon_c = 0
		self.lat_c = 0
		self.zoom = 2500


		self.screen = pygame.display.set_mode((self.w, self.h))
		self.bg = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent background
		self.airport_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent surface for the airport
		self.trail_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent surface for trails
		self.traj_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent surface for trajectories
		self.fg = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent surface for drawing

	def update_plane_state(self, state):
		"""Update a single plane's state without redrawing the entire display."""
		# Handle events only once per frame
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.stop_display()
				
		# Initialize plane color if not exists
		if state['id'] not in self.plane_colors:
			self.plane_colors[state['id']] = np.random.randint(0, 255, size=3).tolist()

		# Initialize trajectory if not exists
		#if state['traj'] not in self.plane_colors:
		#	self.plane_colors[state['traj']] = [(state['lon'] + state['vel'].longitude * i, state['lat'] + state['vel'].latitude * i) for i in range(0, 11)]

		# Initialize trail if not exists
		if state['id'] not in self.trails:
			self.trails[state['id']] = []
			
		# Add current position to trail
		self.trails[state['id']].append((state['lon'], state['lat']))

		if len(self.trails[state['id']]) > self.trail_length:
			self.trails[state['id']] = self.trails[state['id']][-self.trail_length:]

	def render(self):
		"""Render all planes, their trails, and their trajectories to the display."""
		# Clear the surfaces
		self.fg.fill((0, 0, 0, 0))
		self.trail_surface.fill((0, 0, 0, 0))
		self.airport_surface.fill((0, 0, 0, 0))
		self.bg.fill((0, 0, 0))

		# Draw all trails and planes
		for plane_id, trail in self.trails.items():
			if not trail:
				continue
				
			color = self.plane_colors[plane_id]
			plane_x, plane_y = self.wgs84_to_xy(
				self.last_states.get(plane_id, {}).get('lon', self.x_c),
				self.last_states.get(plane_id, {}).get('lat', self.y_c)
			)

			# Draw trail dots (excluding the last/current position)
			for pos in trail[:-1]:
				x, y = self.wgs84_to_xy(pos[0], pos[1])
				pygame.draw.circle(self.fg, color, (x, y), 1)

			# Draw the plane's trajectory
			for pos in self.last_states[plane_id]['traj'][:-1]:
				x, y = self.wgs84_to_xy(pos[0], pos[1])
				pygame.draw.circle(self.fg, color, (x, y), 2)

			# Draw triangle at the current position
			if trail:
				lon, lat = trail[-1]
				x, y = self.wgs84_to_xy(lon, lat)

				if hasattr(self, 'last_states') and plane_id in self.last_states:
					hdg = self.last_states[plane_id]['hdg']
				else:
					hdg = 0  # Default heading north
					
				angle = np.deg2rad(hdg-90)
				# Triangle points
				size = 8
				points = [
					(x + size * np.cos(angle), y + size * np.sin(angle)),
					(x + size * np.cos(angle + 2.5), y + size * np.sin(angle + 2.5)),
					(x, y),
					(x + size * np.cos(angle - 2.5), y + size * np.sin(angle - 2.5)),
				]
				pygame.draw.polygon(self.fg, color, points)

			# Draw labels

			# ID label
			id_font = pygame.font.Font(None, 24)
			id_label = id_font.render(plane_id, True, color)
			self.fg.blit(id_label, (plane_x - id_label.get_width() // 2, plane_y - id_label.get_height() // 2 + 25))

			# Debug labels
			if self.debug_labels:
				# Draw labels only if debug_labels is True
				# Convert state to display units
				display_state = self.state_to_display(self.last_states[plane_id])
				# Altitude label
				alt_font = pygame.font.Font(None, 18)
				alt_label = alt_font.render(f"Alt: {display_state['alt']:.0f}ft", True, color)
				# Heading label
				heading_label = alt_font.render(f"Hdg: {display_state['hdg']:.0f}°", True, color)
				# Velocity label
				vel_label = alt_font.render(f"Gspd: {display_state['gspd']:.0f} kts", True, color)
				# Blit labels at the position
				self.fg.blit(vel_label, (plane_x - vel_label.get_width() // 2, plane_y - vel_label.get_height() // 2 + 65))
				self.fg.blit(alt_label, (plane_x - alt_label.get_width() // 2, plane_y - alt_label.get_height() // 2 + 45))
				self.fg.blit(heading_label, (plane_x - heading_label.get_width() // 2, plane_y - heading_label.get_height() // 2 - 25))		

		# Draw airport
		
		# Draw runways
		for runway in self.airport.runways.values():
			start_x, start_y = self.wgs84_to_xy(runway.start_point[1], runway.start_point[0])
			end_x, end_y = self.wgs84_to_xy(runway.end_point[1], runway.end_point[0])
			color = (255, 255, 255) if not runway.is_occupied else (255, 0, 0)
			# Calculate the angle of the runway
			dx = end_x - start_x
			dy = end_y - start_y
			length = np.hypot(dx, dy)
			if length == 0:
				continue  # Avoid division by zero

			runway_width = self.nm_to_xy(units.nautical(feet=200))  # pixels, adjust as needed
			angle = np.arctan2(dy, dx)

			# Calculate the four corners of the rectangle
			offset_x = (runway_width / 2) * np.sin(angle)
			offset_y = (runway_width / 2) * -np.cos(angle)

			points = [
				(start_x - offset_x, start_y - offset_y),
				(start_x + offset_x, start_y + offset_y),
				(end_x + offset_x, end_y + offset_y),
				(end_x - offset_x, end_y - offset_y),
			]
			pygame.draw.polygon(self.airport_surface, color, points)

		# Draw nautical mile circles
		for i in range(2, 12, 2):  # Draw circles at 2, 4, 6, 8, and 10 NM
			radius = self.nm_to_xy(i)  # I have no idea why this is the conversion factor, but it works
			pygame.draw.circle(self.bg, (0, 255, 0, 255), (self.x_c, self.y_c), radius, 1)
			# Draw the radius label
			radius_label = pygame.font.Font(None, 18).render(f"{i} NM", True, (0, 255, 0))
			self.bg.blit(radius_label, (self.x_c + radius - radius_label.get_width() // 2 + 5, self.y_c - radius_label.get_height() // 2))
		# Draw the center point
		pygame.draw.circle(self.bg, (255, 0, 0), (self.x_c, self.y_c), 5)

		# +X = right, +Y = down
		# +lon = east, +lat = north

		# Layer surfaces and update the display
		self.screen.blit(self.bg, (0, 0))  # Draw the background
		self.screen.blit(self.airport_surface, (0, 0))
		self.screen.blit(self.trail_surface, (0, 0))  # Draw the trails
		self.screen.blit(self.fg, (0, 0))
		pygame.display.flip()

	def update_display(self, states):
		"""Update the display with multiple plane states.
		Args:
			states: Either a single state dict or a list of state dicts.
		"""
		# Handle both single state and list of states
		if isinstance(states, dict):
			states = [states]

		# Initialize last_states if it doesn't exist
		if not hasattr(self, 'last_states'):
			self.last_states = {}

		# Update all plane states
		for state in states:
			self.update_plane_state(state)
			# Store the last state for each plane
			self.last_states[state['id']] = state
		
		# Render everything once
		self.render()

	def stop_display(self):
		pygame.quit()

	def wgs84_to_xy(self, lon, lat) -> tuple:
		"""Convert WGS84 coordinates to display coordinates."""
		x = int((lon - self.lon_c) * self.zoom + self.x_c)
		y = int((self.lat_c - lat) * self.zoom + self.y_c) # y is inverted in display coordinates
		return x, y
	
	def state_to_display(self, state) -> dict:
		"""Convert a plane state to display units."""
		display_state = state.copy()
		display_state['gspd'] = int(state['gspd'] * 1.94384)  # Convert m/s to kts
		display_state['alt'] = int(state['alt'] * 3.28084)  # Convert meters to feet
		return display_state
	
	def setup_airport(self, airport):
		"""Setup the airport layout on the display."""
		if not hasattr(self, 'airport'):
			self.airport = airport
		else:
			self.airport.runways.update(airport.runways)

	def nm_to_xy(self, dist):
		"""Convert distance in nautical miles to pixels."""
		return int(dist * 0.0168 * self.zoom)
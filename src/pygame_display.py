import pygame
import numpy as np

class Pygame_Display:
	"""Center coordinates for the display."""
	plane_colors = {}
	trails = {}
	last_states = {}
	debug_labels = True
	"""A class to handle the Pygame display for the plane simulation."""
	def __init__(self, w=1280, h=720):
		pygame.init()
		self.w = w
		self.h = h
		self.x_c = self.w // 2
		self.y_c = self.h // 2
		self.bg = pygame.display.set_mode((self.w, self.h))
		self.clock = pygame.time.Clock()
		self.bg.fill((0, 0, 0))  # Fill the screen with black
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

		# Initialize trail if not exists
		if state['id'] not in self.trails:
			self.trails[state['id']] = []
			
		# Add current position to trail
		self.trails[state['id']].append((state['lon'], state['lat']))
		
		# Limit trail length
		max_trail_length = 200
		if len(self.trails[state['id']]) > max_trail_length:
			self.trails[state['id']] = self.trails[state['id']][-max_trail_length:]

	def render(self):
		"""Render all planes and their trails to the display."""
		# Clear the surfaces
		self.fg.fill((0, 0, 0, 0))
		self.bg.fill((0, 0, 0))

		# Draw all trails and planes
		for plane_id, trail in self.trails.items():
			if not trail:
				continue
				
			color = self.plane_colors[plane_id]
			
			# Draw trail dots (excluding the last/current position)
			for pos in trail[:-1]:
				x = int(pos[0] * 100 + self.x_c)
				y = int(pos[1] * 100 + self.y_c)
				pygame.draw.circle(self.fg, color, (x, y), 2)

			# Draw triangle at the current position
			if trail:
				lon, lat = trail[-1]
				x = int(lon * 100 + self.x_c)
				y = int(lat * 100 + self.y_c)
				
				# We need the heading for the triangle, so we'll store it
				# For now, we'll use a default pointing north if heading is not available
				if hasattr(self, 'last_states') and plane_id in self.last_states:
					hdg = self.last_states[plane_id]['hdg']
				else:
					hdg = 0  # Default heading north
					
				angle = np.deg2rad(90 - hdg)
				# Triangle points
				size = 13
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
				self.fg.blit(id_label, (x - id_label.get_width() // 2, y - id_label.get_height() // 2 + 25))
				
				# Debug labels
				if self.debug_labels:
					# Draw labels only if debug_labels is True
					# Altitude label
					alt_font = pygame.font.Font(None, 18)
					alt_label = alt_font.render(f"Alt: {self.last_states[plane_id]['alt']:.0f}m", True, color)
					# Heading label
					heading_label = alt_font.render(f"Hdg: {hdg:.0f}°", True, color)
					# Velocity label
					vel_label = alt_font.render(f"Gspd: {self.last_states[plane_id]['gspd']:.0f} m/s", True, color)
					# Blit labels at the position
					self.fg.blit(vel_label, (x - vel_label.get_width() // 2, y - vel_label.get_height() // 2 + 65))
					self.fg.blit(alt_label, (x - alt_label.get_width() // 2, y - alt_label.get_height() // 2 + 45))
					self.fg.blit(heading_label, (x - heading_label.get_width() // 2, y - heading_label.get_height() // 2 - 25))

		# Blit the foreground onto the background and update display
		self.bg.blit(self.fg, (0, 0))
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
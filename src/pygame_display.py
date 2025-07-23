import pygame
import numpy as np
import utils
from geopy import units, distance

class Pygame_Display:	
	"""A class to handle the Pygame display for the plane simulation."""
	plane_colors = {}
	trails = {}
	last_states = {}
	debug_labels = True
	trail_length = 25  # Maximum length of the trail in number of points
	def __init__(self, w=1280, h=720):
		pygame.init()
		self.w = w
		self.h = h
		self.x_c = self.w // 2
		self.y_c = self.h // 2
		# Center on the coordinates we're optimized for
		self.lon_c = -103.06126  # Longitude center (Rapid City area)
		self.lat_c = 44.04882    # Latitude center (Rapid City area)
		self.zoom = 10  # Much lower zoom for aviation-appropriate scale

		# FPS tracking
		self.clock = pygame.time.Clock()
		self.fps_font = pygame.font.Font(None, 24)
		
		# Tick counter (separate from the real tick counter in FlightSimulator but it's just for display anyway)
		self.tick_count = 0
		
		# Turbo mode state (10x speed when V is held)
		self.turbo_mode = False
		
		# Cache fonts for plane labels
		self.id_font = pygame.font.Font(None, 24)
		self.debug_font = pygame.font.Font(None, 18)
		self.nm_label_font = pygame.font.Font(None, 18)
		
		# Flags for static content caching
		self._airport_rendered = False
		self._background_rendered = False

		self.screen = pygame.display.set_mode((self.w, self.h))
		self.bg = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent background
		self.airport_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent surface for the airport
		self.traj_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent surface for trajectories
		self.fg = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Create a transparent surface for drawing

	def _update_single_plane_state(self, state):
		"""Update a single plane's state without handling events."""
		# Initialize plane color if not exists
		if state['callsign'] not in self.plane_colors:
			random_color_key = np.random.randint(0,3)
			palette = []
			for i in range(3):
				if i == random_color_key:
					palette.append(np.random.randint(150,256))
					# Ensures that at least one of RGB is 150 or more so the color is visible
				else:
					palette.append(np.random.randint(0,256))
			self.plane_colors[state['callsign']] = palette

		# Initialize trail if not exists
		if state['callsign'] not in self.trails:
			self.trails[state['callsign']] = []
			
		# Add current position to trail
		self.trails[state['callsign']].append((state['lon'], state['lat']))

		if len(self.trails[state['callsign']]) > self.trail_length:
			self.trails[state['callsign']] = self.trails[state['callsign']][-self.trail_length:]

	def handle_events(self):
		"""Handle Pygame events."""
		pygame.event.pump()  # Process events without blocking

		pressed_keys = pygame.key.get_pressed()
		if pressed_keys[pygame.K_ESCAPE]:
			self.stop_display()
		if pressed_keys[pygame.K_EQUALS]:
			self.zoom *= 1.01  # Zoom in
			self._clear_rendered_content()  # Need to redraw when zoom changes
		if pressed_keys[pygame.K_MINUS]:
			self.zoom *= 0.99  # Zoom out
			self._clear_rendered_content()  # Need to redraw when zoom changes
		
		# Check for turbo mode (10x speed while V is held)
		self.turbo_mode = pressed_keys[pygame.K_v]
		
		# Handle viewport movement with arrow keys
		# Movement speed inversely proportional to zoom (more zoomed in = slower movement)
		move_speed = 0.001 / (self.zoom / 2500)  # Base movement speed scaled by zoom
		
		if pressed_keys[pygame.K_UP]:
			self.lat_c += move_speed
			self._clear_rendered_content()  # Need to redraw when viewport moves
		if pressed_keys[pygame.K_DOWN]:
			self.lat_c -= move_speed
			self._clear_rendered_content()
		if pressed_keys[pygame.K_LEFT]:
			self.lon_c -= move_speed
			self._clear_rendered_content()
		if pressed_keys[pygame.K_RIGHT]:
			self.lon_c += move_speed
			self._clear_rendered_content()

	def _clear_rendered_content(self):
		"""Mark static content as needing redraw (called when zoom changes)."""
		self._airport_rendered = False
		self._background_rendered = False
	def render(self):
		"""Render all planes, their trails, and their trajectories to the display."""
		# Clear only the foreground surface (planes and trails change every frame)
		self.fg.fill((0, 0, 0, 0))
		
		# Only clear and redraw static content when necessary
		if not self._background_rendered:
			self._render_background()
			self._background_rendered = True
			
		if not self._airport_rendered:
			self._render_airport()
			self._airport_rendered = True

		# Draw all trails and planes
		for plane_id, trail in self.trails.items():
			if not trail:
				continue
				
			color = self.plane_colors[plane_id]
			
			# Get plane state once
			plane_state = self.last_states.get(plane_id, {})
			plane_lon = plane_state.get('lon', self.x_c)
			plane_lat = plane_state.get('lat', self.y_c)
			plane_x, plane_y = self.wgs84_to_xy(plane_lon, plane_lat)

			# Draw trail dots (excluding the last/current position)
			# Batch coordinate conversion for better performance
			trail_coords = []
			for pos in trail[:-1]:
				x, y = self.wgs84_to_xy(pos[0], pos[1])
				trail_coords.append((x, y))
			
			# Draw all trail dots
			for x, y in trail_coords:
				pygame.draw.circle(self.fg, color, (x, y), 1)

			# Draw the plane's trajectory...
			# ...as a series of dots (we'll use this later)
			#for pos in self.last_states[plane_id]['traj'][:-1]:
			#	x, y = self.wgs84_to_xy(pos[0], pos[1])
			#	pygame.draw.circle(self.fg, color, (x, y), 2)

			# ...as a singular line (using this for visibility now)
			if 'traj' in plane_state and len(plane_state['traj']) >= 2:
				firpos = plane_state['traj'][0]
				x, y = self.wgs84_to_xy(firpos[0], firpos[1])

				laspos = plane_state['traj'][-1]
				u, v = self.wgs84_to_xy(laspos[0], laspos[1])

				pygame.draw.line(self.fg, color, (x, y), (u, v))

			# Draw triangle at the current position (reuse coordinates from trail)
			if trail:
				# Use the already calculated plane_x, plane_y instead of recalculating
				hdg = plane_state.get('hdg', 0)  # Default heading north
					
				angle = np.deg2rad(hdg-90)
				# Triangle points
				size = 8
				points = [
					(plane_x + size * np.cos(angle), plane_y + size * np.sin(angle)),
					(plane_x + size * np.cos(angle + 2.5), plane_y + size * np.sin(angle + 2.5)),
					(plane_x, plane_y),
					(plane_x + size * np.cos(angle - 2.5), plane_y + size * np.sin(angle - 2.5)),
				]
				pygame.draw.polygon(self.fg, color, points)

			# Draw labels

			# ID label (using cached font)
			id_label = self.id_font.render(plane_id, True, color)
			self.fg.blit(id_label, (plane_x - id_label.get_width() // 2, plane_y - id_label.get_height() // 2 + 25))

			# Debug labels
			if self.debug_labels:
				# Draw labels only if debug_labels is True
				# Convert state to display units
				display_state = self.state_to_display(plane_state)
				# Altitude, heading, and velocity labels (using cached font)
				alt_label = self.debug_font.render(f"Alt: {display_state['alt']:.0f}ft", True, color)
				heading_label = self.debug_font.render(f"Hdg: {display_state['hdg']:.0f}°", True, color)
				vel_label = self.debug_font.render(f"Gspd: {display_state['gspd']:.0f} kts", True, color)
				# Blit labels at the position
				self.fg.blit(vel_label, (plane_x - vel_label.get_width() // 2, plane_y - vel_label.get_height() // 2 + 65))
				self.fg.blit(alt_label, (plane_x - alt_label.get_width() // 2, plane_y - alt_label.get_height() // 2 + 45))
				self.fg.blit(heading_label, (plane_x - heading_label.get_width() // 2, plane_y - heading_label.get_height() // 2 - 25))

		# Layer surfaces and update the display
		self.screen.blit(self.bg, (0, 0))  # Draw the background
		self.screen.blit(self.airport_surface, (0, 0))
		self.screen.blit(self.fg, (0, 0))
		
		# Draw FPS counter in top right corner
		fps = self.clock.get_fps()
		fps_text = self.fps_font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
		fps_rect = fps_text.get_rect()
		fps_rect.topright = (self.w - 10, 10)
		self.screen.blit(fps_text, fps_rect)
		
		# Draw tick counter in top left corner
		tick_display = f"Tick: {self.tick_count}"
		if self.turbo_mode:
			tick_display += " (TURBO)"
		tick_text = self.fps_font.render(tick_display, True, (255, 255, 0) if self.turbo_mode else (255, 255, 255))
		tick_rect = tick_text.get_rect()
		tick_rect.topleft = (10, 10)
		self.screen.blit(tick_text, tick_rect)
		
		# Draw keybind info and zoom info in bottom left corner
		keybind_lines = [
			"ESC: Quit",
			"+/-: Zoom In/Out",
			"Arrow Keys: Move Viewport",
			"V: Turbo Mode (10x speed)",
			f"Zoom: {self.zoom:.0f}"
		]
		
		for i, line in enumerate(keybind_lines):
			text = self.debug_font.render(line, True, (255, 255, 255))
			text_rect = text.get_rect()
			text_rect.bottomleft = (10, self.h - 10 - (i * 20))
			self.screen.blit(text, text_rect)
		
		pygame.display.flip()

	def _render_background(self):
		"""Render static background elements (nautical mile circles, center point)."""
		self.bg.fill((0, 0, 0))
		
		# Convert display center to screen coordinates
		origin_x, origin_y = self.wgs84_to_xy(self.lon_c, self.lat_c)
		
		# Draw nautical mile circles centered at display center
		for i in range(2, 12, 2):  # Draw circles at 2, 4, 6, 8, and 10 NM
			radius = self.nm_to_xy(i)
			pygame.draw.circle(self.bg, (0, 255, 0, 255), (origin_x, origin_y), radius, 1)
			# Draw the radius label (using cached font)
			radius_label = self.nm_label_font.render(f"{i} NM", True, (0, 255, 0))
			self.bg.blit(radius_label, (origin_x + radius - radius_label.get_width() // 2 + 5, origin_y - radius_label.get_height() // 2))
		# Draw the center point at display center
		pygame.draw.circle(self.bg, (255, 0, 0), (origin_x, origin_y), 5)

	def _render_airport(self):
		"""Render static airport elements (runways)."""
		self.airport_surface.fill((0, 0, 0, 0))
		
		# Draw runways
		for runway in self.airport.runways.values():
			# Use the proper methods to get coordinates from geopy.Point objects
			start_x, start_y = self.wgs84_to_xy(runway.start_point.longitude, runway.start_point.latitude)
			end_x, end_y = self.wgs84_to_xy(runway.end_point.longitude, runway.end_point.latitude)
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

		# Handle events once per frame
		self.handle_events()

		# Create a set of current plane IDs for O(1) lookup
		current_plane_ids = {state['callsign'] for state in states}

		# Single iteration: update states and trails, remove stale planes
		planes_to_remove = []
		for plane_id in self.last_states:
			if plane_id not in current_plane_ids:
				planes_to_remove.append(plane_id)

		# Remove stale planes
		for plane_id in planes_to_remove:
			if plane_id in self.trails:
				del self.trails[plane_id]
			if plane_id in self.plane_colors:
				del self.plane_colors[plane_id]
			del self.last_states[plane_id]

		# Update current planes
		for state in states:
			self._update_single_plane_state(state)
			self.last_states[state['callsign']] = state

		# Increment tick counter
		self.tick_count += 1

		# Render everything once
		self.render()
		
		# Update FPS clock
		self.clock.tick()

	def stop_display(self):
		pygame.quit()

	def wgs84_to_xy(self, lon, lat) -> tuple:
		"""Convert WGS84 coordinates to display coordinates using proper coordinate conversion."""
		# Convert lat/lon to meters using our location-specific conversion
		x_meters, y_meters = utils.latlon_to_meters(lat, lon, origin_lat=self.lat_c, origin_lon=self.lon_c)
		
		# Convert meters to screen coordinates
		# The zoom factor acts as pixels per meter - higher zoom = more zoomed in
		# Default zoom of 2500 means we're showing roughly 2500 pixels per meter, which is way too zoomed in
		# Let's use a more reasonable scale: zoom factor as pixels per kilometer
		scale_factor = self.zoom / 1000.0  # Convert zoom from "pixels per meter" to "pixels per kilometer"
		x = int(x_meters * scale_factor + self.x_c)
		y = int(-y_meters * scale_factor + self.y_c)  # Negative y because screen y increases downward
		return x, y
	
	def state_to_display(self, state) -> dict:
		"""Convert a plane state to display units."""
		display_state = state.copy()
		display_state['gspd'] = int(utils.mps_to_knots(state['gspd']))  # Convert m/s to kts
		display_state['alt'] = int(state['alt'] * 3.28084)  # Convert meters to feet
		return display_state
	
	def setup_airport(self, airport):
		"""Setup the airport layout on the display."""
		if not hasattr(self, 'airport'):
			self.airport = airport
		else:
			self.airport.runways.update(airport.runways)
		# Mark airport content as needing redraw
		self._airport_rendered = False

	def nm_to_xy(self, dist):
		"""Convert distance in nautical miles to pixels."""
		# Convert nautical miles to meters, then to pixels using the same scale as wgs84_to_xy
		meters = dist * 1852.0  # 1 nautical mile = 1852 meters
		scale_factor = self.zoom / 1000.0  # Same scale factor as in wgs84_to_xy
		return int(meters * scale_factor)
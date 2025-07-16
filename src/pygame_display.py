import pygame
import numpy as np

class Pygame_Display:
	"""Center coordinates for the display."""
	plane_colors = {}
	"""A class to handle the Pygame display for the plane simulation."""
	def __init__(self, w=1280, h=720):
		pygame.init()
		self.w = w
		self.h = h
		self.x_c = self.w // 2
		self.y_c = self.h // 2
		self.screen = pygame.display.set_mode((self.w, self.h))
		self.clock = pygame.time.Clock()
		self.screen.fill((0, 0, 0))  # Fill the screen with black

	def update_display(self, state):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.stop_display()
		if state['id'] not in self.plane_colors:
			self.plane_colors[state['id']] = np.random.randint(0, 255, size=3).tolist()
		# Draw the plane's position
		pygame.draw.circle(self.screen, self.plane_colors[state['id']], (int(state['lon'] * 100 + self.x_c), int(state['lat'] * 100 + self.y_c)), 5)
		pygame.display.flip()

	def stop_display(self):
		pygame.quit()
import pygame
import numpy as np

class Pygame_Display:
	h, w = 800, 600
	x_c, y_c = w // 2, h // 2
	plane_colors = {}
	"""A class to handle the Pygame display for the plane simulation."""
	def __init__(self):
		pygame.init()
		self.screen = pygame.display.set_mode((self.w, self.h))
		self.clock = pygame.time.Clock()
		self.screen.fill((255, 255, 255))

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
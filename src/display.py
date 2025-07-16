import matplotlib.pyplot as plt

positions = {}

def add_position(plane_state):
	if plane_state['id'] not in positions:
		positions[plane_state['id']] = []
	positions[plane_state['id']].append((plane_state['lon'], plane_state['lat']))

def plot_positions(id):
	print(positions)
	plt.scatter(x=positions[id][:][0], y=positions[id][:][1], color='blue')

def display():
	plt.show()
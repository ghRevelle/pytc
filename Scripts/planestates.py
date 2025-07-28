from enum import Enum

# the DRL will gather its rewards/punishments via checking these plane states
class PlaneState(Enum):
	AIR = 0
	LANDING = 1 # landing -> tells the DRL that the craft is landing IN THIS TICK
	WAITING_FOR_LANDING = 2 # waiting for landing -> once planes reach their final cruising altitude
	REALIGNING = 3 # if the plane is currently realigning
	QUEUED = 4 # queued -> only used for planes in the queue
	TAXIING = 5 # taxiing -> only used for planes that have just landed
	TAKINGOFF = 6 # taking off -> tells the DRL that the craft is taking off IN THIS TICK
	WAITING_FOR_TAKEOFF = 7 # waiting for takeoff -> once planes are out on the runway
	MARKED_FOR_DELETION = 8 # marked for deletion -> used for planes that have completed taxiing
	CRASHED = 9
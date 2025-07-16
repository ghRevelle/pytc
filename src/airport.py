# Airport and runway class for flight simulator
import geopy.distance
import math


class Runway:
    def __init__(self, start_point, end_point, is_occupied=False):
        """Initilize a runway with the following attributes:
            start_point (tuple) : latitude, longitude of start point
            end_point (tuple) : latitude, longitude of end point
            is_occupied (bool) : if the runway is occupied by a plane being on it
        """
        self.start_point = start_point
        """Start point of the runway in (lat, lon) format"""
        self.end_point = end_point
        """End point of the runway in (lat, lon) format"""
        self.hdg = 90 - math.degrees(math.atan2(
            end_point[1] - start_point[1], end_point[0] - start_point[0]))
        """Heading of the runway in degrees"""
        self.length = geopy.distance.distance(start_point, end_point).feet
        """Length of the runway in feet"""
        self.is_occupied = is_occupied
        """If the runway is occupied by a plane being on it"""

class Airport:
    def __init__(self, runways : dict={}):
        """Initialize the airport with its runway layout
        Args:
            runways : dictionary mapping runway names to runway class
        """
        if not isinstance(runways, dict):
            runways = {}
        else:
            self.runways = runways
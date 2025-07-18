# Airport and runway class for flight simulator
import geopy.distance
import math
import shapely
import utils

class Runway:
    def __init__(self, start_point, end_point, is_occupied=False):
        """Initilize a runway with the following attributes:
            start_point (tuple) : latitude, longitude of start point
            end_point (tuple) : latitude, longitude of end point
            is_occupied (bool) : if the runway is occupied by a plane being on it
        """
        self.start_point = geopy.Point(start_point[0], start_point[1])
        """Start point of the runway in (lat, lon) format"""
        self.end_point = geopy.Point(end_point[0], end_point[1])
        """End point of the runway in (lat, lon) format"""
        # self.hdg = 90 - math.degrees(math.atan2(
        #     end_point[1] - start_point[1], end_point[0] - start_point[0]))
        self.hdg = utils.calculate_bearing(start_point, end_point)
        """Heading of the runway in degrees"""
        self.length = geopy.distance.distance(start_point, end_point).feet
        """Length of the runway in feet"""
        self.is_occupied = is_occupied
        """If the runway is occupied by a plane being on it"""
    
    def get_start_point(self) -> geopy.Point:
        """Get the start point of the runway."""
        return self.start_point

    def get_end_point(self) -> geopy.Point:
        """Get the end point of the runway."""
        return self.end_point

    def get_line(self):
        """Get the runway line as a shapely LineString object.
        Returns:
            object: The runway line as a shapely LineString object.
        """
        return shapely.geometry.LineString([self.start_point, self.end_point])

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
# Airport and runway class for flight simulator
import geopy.distance
import math
import shapely
import utils

class Runway:
    def __init__(self, start_point, end_point, name=None, is_occupied=False):
        """Initilize a runway with the following attributes:
            start_point (tuple) : latitude, longitude of start point
            end_point (tuple) : latitude, longitude of end point
            is_occupied (bool) : if the runway is occupied by a plane being on it
        """
        self.start_point = geopy.Point(start_point[0], start_point[1])
        """Start point of the runway in (lat, lon) format"""
        self.end_point = geopy.Point(end_point[0], end_point[1])
        """End point of the runway in (lat, lon) format"""
        self.name = name
        """Name of the runway"""
        # self.hdg = 90 - math.degrees(math.atan2(
        #     end_point[1] - start_point[1], end_point[0] - start_point[0]))
        self.hdg = utils.calculate_bearing(start_point, end_point)
        """Heading of the runway in degrees"""
        self.length = geopy.distance.distance(start_point, end_point).feet
        """Length of the runway in feet"""
        self.is_occupied = is_occupied
        """If the runway is occupied by a plane being on it"""

    def get_start_point_ll(self) -> geopy.Point:
        """Get the start point of the runway as a geopy.Point(latitude, longitude)."""
        return self.start_point

    def get_end_point_ll(self) -> geopy.Point:
        """Get the end point of the runway as a geopy.Point(latitude, longitude)."""
        return self.end_point

    def get_start_point_xy(self) -> tuple[float, float]:
        """Get the start point of the runway as a tuple (longitude, latitude).
        Returns:
            tuple: The start point of the runway in (longitude, latitude) format.
        """
        return (self.start_point.longitude, self.start_point.latitude)

    def get_end_point_xy(self) -> tuple[float, float]:
        """Get the end point of the runway as a tuple (longitude, latitude).
        Returns:
            tuple: The end point of the runway in (longitude, latitude) format.
        """
        return (self.end_point.longitude, self.end_point.latitude)

    def get_line_xy(self):
        """Get the runway line as a shapely LineString object using the latitude and longitude of the start and end points.
        Returns:
            object: The runway line as a shapely LineString object.
        """
        return shapely.geometry.LineString([
            (self.start_point.longitude, self.start_point.latitude), 
            (self.end_point.longitude, self.end_point.latitude)
        ])

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
        
        self.queue = []

    def add_to_queue(self, plane_id, runway):
        """Add a plane to the takeoff queue for a specific runway."""
        self.queue.append((plane_id, runway))

    def remove_from_queue(self, plane_id):
        """Remove a plane from the takeoff queue."""
        self.queue = [item for item in self.queue if item[0] != plane_id]

    def get_top_of_queue(self):
        return self.queue[0][0]

    def pop_top_of_queue(self):
        pop_id = self.queue.pop(0)[0]

        print(f"Plane id at the tope of the queue: {pop_id}")

        return pop_id
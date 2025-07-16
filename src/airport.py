# Airport class for flight simulator

class Runway:
    def __init__(self, start_point, end_point, hdg, length, is_occupied=False):
        """Initilize a runway with the following attributes:
            start_point (tuple) : latitude, longitude of start point
            end_point (tuple) : 
        """
        self.start_point = start_point
        self.end_point = end_point
        self.hdg = hdg
        self.length = length
        self.is_occupied = is_occupied

class Airport:
    def __init__(self, runways : dict):
        """Initialize the airport with its runway layout
        Args:
            runways : dictionary mapping runway names to runway class
        """
        self.runways = runways
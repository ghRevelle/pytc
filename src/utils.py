import shapely, math

def calculate_intersection(line1: shapely.geometry.LineString, line2: shapely.geometry.LineString) -> tuple:
		"""Calculate the intersection of the plane's trajectory with a runway.
		Args:
			runway (Runway): The runway to check for intersection.
		Returns:
			bool: True if the trajectory intersects with the runway, False otherwise.
		"""
		intersection = line1.intersection(line2)
		if intersection is None or intersection.is_empty:
			raise ValueError("No intersection found.")
		else:
			return intersection.coords[0]
		
def meters_to_degrees(heading: float, meters: float) -> float:
		"""
		Convert a distance in meters along a given heading to degrees (approximate, WGS84).
		heading: degrees from north (0 = north, 90 = east)
		meters: distance in meters
		Returns: distance in degrees
		"""
		# Project meters onto latitude and longitude axes
		dlat = meters * math.cos(math.radians(heading)) / 111320.0
		dlon = meters * math.sin(math.radians(heading)) / 111320.0
		# Return the total angular distance (Euclidean in degree space)
		return math.hypot(dlat, dlon)

def degrees_to_meters(heading: float, degrees: float) -> float:
		"""
		Convert a distance in degrees along a given heading to meters (approximate, WGS84).
		heading: degrees from north (0 = north, 90 = east)
		degrees: distance in degrees
		Returns: distance in meters
		"""
		dlat = degrees * math.cos(math.radians(heading))
		dlon = degrees * math.sin(math.radians(heading))
		meters = math.hypot(dlat * 111320.0, dlon * 111320.0)
		return meters

def mps_to_knots(mps: float) -> float:
	return mps * 1.94384

def knots_to_mps(knots: float) -> float:
	return knots / 1.94384

def calculate_bearing(start_point, end_point):
        """Calculate the bearing from (lat1, lon1) to (lat2, lon2) in degrees."""
        lat1, lon1 = start_point
        lat2, lon2 = end_point
        dLon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        x = math.sin(dLon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dLon))
        bearing = math.degrees(math.atan2(x, y))
        return int(round((bearing + 360) % 360))
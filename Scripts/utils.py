import geopy
import shapely, math
import numpy as np

def lerp(amt, low, high):
	"""Linearly interpolates between two numbers.
	Args:
		current: The amount to interpolate between low and high.
		low: The value at the low end of the lerp range.
		high: The value at the high end of the lerp range.
	Returns:
		float: The interpolated value.
	"""

	return low + (high - low) * amt

def get_turn_rate(bank_angle, gspd):
	"""Determines an aircraft's turn rate based on its maximum bank angle and current groundspeed.
	Args:
		bank_angle: The aircraft's maximum commercial bank angle.
		gspd: The aircraft's cruising speed.
	Returns:
		float: The aircraft's turn rate in degrees/sec."""
	
	return abs(1091 * math.tan(bank_angle) / gspd)


def calculate_craft_distance(lat1, lon1, lat2, lon2, alt1, alt2):
	"""Finds the 3D distance between two aircraft.
	Args:
		plane1: The first aircraft to compare.
		plane2: The second aircraft to compare.
	Returns:
		float: The 3D distance between the aircraft in meters."""
	
	# Latitude difference in meters (constant ~111,320 m per degree)
	dlat_meters = abs(lat1 - lat2) * get_meters_per_degree_lat()
	
	# Longitude difference in meters (varies with latitude)
	# Use average latitude for the longitude conversion
	avg_lat = (lat1 + lat2) / 2
	dlon_meters = abs(lon1 - lon2) * get_meters_per_degree_lon(avg_lat)
	
	# Calculate horizontal distance
	horizontal_distance = math.sqrt(dlat_meters**2 + dlon_meters**2)
	
	# Altitude difference (requires alt in meters)
	altitude_difference = alt1 - alt2
	
	# Calculate 3D distance using Pythagorean theorem
	return math.sqrt(horizontal_distance**2 + altitude_difference**2)

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

def get_meters_per_degree_lat() -> float:
		"""Get the meters per degree of latitude.
		Returns:
			float: Meters per degree of latitude."""
		return 110883.79  # Average meters per degree latitude (WGS84)

def get_meters_per_degree_lon(lat: float=44.04882) -> float:
		"""Get the meters per degree of longitude at a given latitude.
		Args:
			lat (float): Latitude in degrees.
		Returns:
			float: Meters per degree of longitude at the specified latitude."""
		return 110883.79 * math.cos(math.radians(lat))

def meters_to_degrees(heading: float, meters: float) -> float:
		"""
		Convert a distance in meters along a given heading to degrees (approximate, WGS84).
		heading: degrees from north (0 = north, 90 = east)
		meters: distance in meters
		Returns: distance in degrees
		"""
		# Project meters onto latitude and longitude axes
		dlat = meters * math.cos(math.radians(heading)) / get_meters_per_degree_lat()
		dlon = meters * math.sin(math.radians(heading)) / get_meters_per_degree_lon()
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
		meters = math.hypot(dlat * get_meters_per_degree_lat(), dlon * get_meters_per_degree_lon())
		return meters

def degrees_to_nautical_miles(heading: float, degrees: float) -> float:
		"""
		Convert a distance in degrees along a given heading to nautical miles (approximate, WGS84).
		heading: degrees from north (0 = north, 90 = east)
		degrees: distance in degrees
		Returns: distance in nautical miles
		"""
		return degrees_to_meters(heading, degrees) / 1852.0

def mps_to_knots(mps: float) -> float:
	return mps * 1.94384

def knots_to_mps(knots: float) -> float:
	return knots / 1.94384

def calculate_bearing(start_point, end_point) -> float:
        """Calculate the bearing from (lat1, lon1) to (lat2, lon2) in degrees."""
        lat1, lon1 = start_point
        lat2, lon2 = end_point
        dLon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        x = math.sin(dLon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dLon))
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line segment.
    
    Args:
        point: (x, y) coordinates of the point
        line_start: (x, y) coordinates of the line start
        line_end: (x, y) coordinates of the line end
    
    Returns:
        float: Perpendicular distance from point to line
    """
    # Convert inputs to numpy arrays for vectorized operations
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    
    # Vector from line start to end
    ab = b - a
    
    # Vector from line start to point
    ap = p - a
    
    # Calculate cross product magnitude (2D cross product gives area of parallelogram)
    cross_product = np.abs(np.cross(ap, ab))
    
    # Calculate line segment length
    line_length = np.linalg.norm(ab)
    
    # Avoid division by zero (degenerate line)
    if line_length == 0:
        raise ValueError("The line segment must have distinct endpoints.")
    
    # Distance is area divided by base length
    return cross_product / line_length

def heading_angle_to_unit_vector(angle):
	angle = math.radians(450 - angle)  # Convert to standard mathematical angle in radians
	return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

def latlon_to_meters(lat: float, lon: float, origin_lat: float = 32.7329, origin_lon: float = -117.1897) -> tuple:
	"""
	Convert latitude and longitude to meters using a local coordinate system.
	Optimized for 32.7329° N / 117.1897° W region.
	Args:
		lat (float): Latitude in degrees
		lon (float): Longitude in degrees
		origin_lat (float): Origin latitude for the local coordinate system (default: 32.7329)
		origin_lon (float): Origin longitude for the local coordinate system (default: -117.1897)
	Returns:
		tuple: (x_meters, y_meters) - coordinates in meters relative to origin
	"""
	# Calculate differences from origin
	dlat = lat - origin_lat
	dlon = lon - origin_lon
	
	# Convert latitude difference to meters (constant ~111,320 m per degree)
	y_meters = dlat * get_meters_per_degree_lat()
	
	# Convert longitude difference to meters (varies with latitude)
	# Use average latitude for the longitude conversion
	avg_lat = (lat + origin_lat) / 2
	x_meters = dlon * get_meters_per_degree_lon(avg_lat)
	
	return (x_meters, y_meters)

def meters_to_latlon(x_meters: float, y_meters: float, origin_lat: float = 32.7329, origin_lon: float = -117.1897) -> tuple:
	"""
	Convert meters back to latitude and longitude coordinates.
	Optimized for 32.7329° N / 117.1897° W region.
	Args:
		x_meters (float): X coordinate in meters
		y_meters (float): Y coordinate in meters
		origin_lat (float): Origin latitude for the local coordinate system (default: 32.7329)
		origin_lon (float): Origin longitude for the local coordinate system (default: -117.1897)
	Returns:
		tuple: (latitude, longitude) in degrees
	"""
	# Convert y_meters back to latitude difference
	dlat = y_meters / get_meters_per_degree_lat()
	lat = origin_lat + dlat
	
	# Convert x_meters back to longitude difference
	# Use the calculated latitude for the longitude conversion
	avg_lat = (lat + origin_lat) / 2
	dlon = x_meters / (get_meters_per_degree_lon(avg_lat))
	lon = origin_lon + dlon
	
	return (lat, lon)

def meters_to_feet(meters: float) -> float:
	"""Convert meters to feet."""
	return meters * 3.28084

def extend_line(line: shapely.geometry.LineString, distance: float) -> shapely.geometry.LineString:
	"""
	Extend a line in both directions by a fixed distance.
	Args:
		line (shapely.geometry.LineString): The line to extend.
		distance (float): The distance to extend the line in both directions.
	"""
	# Get the start and end points of the line
	start = line.coords[0]
	end = line.coords[-1]

	# Calculate the direction vector of the line
	direction = np.array(end) - np.array(start)
	direction /= np.linalg.norm(direction)  # Normalize the vector

	# Extend the line in both directions
	new_start = np.array(start) - direction * distance
	new_end = np.array(end) + direction * distance

	return shapely.geometry.LineString([new_start, new_end])

def distance_from_base(lat: float, lon: float, base_lat: float = 32.7329, base_lon: float = -117.1897) -> float:
	"""
	Calculate the distance from a given latitude and longitude to a base point.
	Args:
		lat (float): Latitude of the point.
		lon (float): Longitude of the point.
		base_lat (float): Latitude of the base point (default: 32.7329).
		base_lon (float): Longitude of the base point (default: -117.1897).
	Returns:
		float: Distance in meters from the base point.
	"""
	x_meters, y_meters = latlon_to_meters(lat, lon, base_lat, base_lon)
	return math.sqrt(x_meters**2 + y_meters**2)

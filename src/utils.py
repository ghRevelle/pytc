import shapely, math
from plane import Plane

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

def calculate_craft_distance(plane1 : Plane, plane2 : Plane):
	"""Finds the 3D distance between two aircraft.
	Args:
		plane1: The first aircraft to compare.
		plane2: The second aircraft to compare.
	Returns:
		float: The 3D distance between the aircraft in meters."""
	
	# Latitude difference in meters (constant ~111,320 m per degree)
	dlat_meters = abs(plane1.lat - plane2.lat) * 111320.0
	
	# Longitude difference in meters (varies with latitude)
	# Use average latitude for the longitude conversion
	avg_lat = (plane1.lat + plane2.lat) / 2
	dlon_meters = abs(plane1.lon - plane2.lon) * 111320.0 * math.cos(math.radians(avg_lat))
	
	# Calculate horizontal distance
	horizontal_distance = math.sqrt(dlat_meters**2 + dlon_meters**2)
	
	# Altitude difference (requires alt in meters)
	altitude_difference = abs(plane1.alt - plane2.alt)
	
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
		
def meters_to_degrees(heading: int, meters: float) -> float:
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

def degrees_to_meters(heading: int, degrees: float) -> float:
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

def degrees_to_nautical_miles(heading: int, degrees: float) -> float:
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
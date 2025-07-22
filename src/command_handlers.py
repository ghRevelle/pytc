# Command handler classes for better separation of concerns
from abc import ABC, abstractmethod
from commands import CommandType
import numpy as np
import utils
import shapely.geometry
from airport import Runway

class CommandHandler(ABC):
    """Abstract base class for command handlers."""
    
    @abstractmethod
    def can_handle(self, command_type: CommandType) -> bool:
        """Check if this handler can process the given command type."""
        pass
    
    @abstractmethod
    def execute(self, plane, command, tick) -> None:
        """Execute the command logic for the given plane."""
        pass

class NoCommandHandler(CommandHandler):
    """Handler for when no command is active."""
    
    def can_handle(self, command_type: CommandType) -> bool:
        return command_type == CommandType.NONE
    
    def execute(self, plane, command, tick) -> None:
        return None

class CruiseCommandHandler(CommandHandler):
    """Handler for plane cruise mode."""

    def can_handle(self, command_type: CommandType) -> bool:
        return command_type == CommandType.CRUISE
    
    def execute(self, plane, command, tick) -> None:
        # Try to achieve a vertrate proportional to the altitudinal error
        alt_error = plane.crz_alt - plane.alt
        v_z_target = max(-plane.dsc_rate, min(plane.asc_rate, alt_error * 0.1))

        plane.v_z = plane.proportional_change(
            current=plane.v_z,
            target=v_z_target,
            min_value=-plane.dsc_rate,
            max_value=plane.asc_rate,
            max_change=plane.acc_z_max
        )

        # Try to achieve a lateral acceleration proportional to the gspd error
        gspd_error = plane.crz_speed - plane.gspd
        gspd_target = max(-plane.acc_xy_max, min(plane.acc_xy_max, gspd_error * 0.1))

        # If the plane is NOT descending...
        if plane.v_z > -plane.dsc_rate / 2:
            # Turn on the speed controller
            plane.acc_xy = plane.proportional_change(
                current=plane.acc_xy,
                target=gspd_target,
                min_value=-plane.acc_xy_max,
                max_value=plane.acc_xy_max,
                max_change=plane.acc_xy_max
            )

class TurnCommandHandler(CommandHandler):
    """Handler for turn commands."""
    
    def can_handle(self, command_type: CommandType) -> bool:
        return command_type == CommandType.TURN
    
    def execute(self, plane, command, tick) -> None:
        current_hdg = plane.hdg
        desired_hdg = command.argument

        if desired_hdg is None or not (0 <= desired_hdg < 360):
            raise ValueError("TURN command missing valid heading.")

        plane._turn(current_hdg, desired_hdg)
        if plane.hdg == desired_hdg:
            command.command_type = CommandType.CRUISE

class LandingCommandHandler(CommandHandler):
    """Handler for landing commands."""
    
    def can_handle(self, command_type: CommandType) -> bool:
        return command_type == CommandType.CLEARED_TO_LAND
    
    def execute(self, plane, command, tick) -> None:
        target_runway = command.argument
        
        # Validation
        self._validate_landing_command(target_runway, command, plane)
        
        target_hdg = target_runway.hdg
        target_dist = self._calculate_runway_distance(plane, target_runway)
        
        # Initialize landing parameters if needed
        if plane.turn_start_time == -1:
            self._initialize_landing(plane, target_runway, command)
        
        # Execute landing phases
        elif tick >= plane.turn_start_time and command.last_update < plane.turn_start_time:
            self._handle_runway_alignment(plane, target_hdg, tick, command)
        elif command.last_update >= plane.turn_start_time and target_dist < plane.tod and plane.alt > 0:
            self._handle_descent_phase(plane, target_dist)
        elif plane.alt <= 0 and plane.gspd > 0:
            self._handle_ground_phase(plane)
        elif plane.gspd <= 0:
            self._handle_landing_complete(plane, command, tick)
    
    def _validate_landing_command(self, target_runway, command, plane):
        """Validate landing command parameters."""
        
        if not isinstance(target_runway, Runway):
            raise TypeError("Argument must be a Runway object.")
        if command.last_update is None or not isinstance(command.last_update, int):
            raise ValueError("Command last_update must be an integer tick value.")
        if plane.turn_start_time is None or not isinstance(plane.turn_start_time, int):
            plane.turn_start_time = -1
    
    def _calculate_runway_distance(self, plane, target_runway):
        """Calculate distance to runway in nautical miles."""
        
        return utils.degrees_to_nautical_miles(
            heading=plane.hdg, 
            degrees=target_runway.get_line_xy().distance(
                shapely.geometry.Point((plane.lon, plane.lat))
            )
        )
    
    def _initialize_landing(self, plane, target_runway, command):
        """Initialize landing parameters."""
        plane.turn_start_time = plane.find_turn_initiation_time(target_runway.get_line_xy(), command.last_update)
        plane.tod = plane._calculate_tod(plane.alt)
        plane.rod = plane._calculate_rod(plane.gspd)
        plane.desired_acc_xy = plane._calculate_target_acc_descend(plane.gspd, plane.alt)
    
    def _handle_runway_alignment(self, plane, target_hdg, tick, command):
        """Handle the runway alignment phase."""
        plane._turn(plane.hdg, target_hdg)
        if plane.hdg == target_hdg:
            command.last_update = tick
    
    def _handle_descent_phase(self, plane, target_dist):
        """Handle the descent phase of landing."""
        plane.rod = plane._calculate_rod(plane.gspd)
        
        if target_dist < 1:
            descent_rate = -(plane.alt * plane.gspd / target_dist)
        else:
            descent_rate = -plane.rod
        
        plane.v_z = plane.proportional_change(
            current=plane.v_z,
            target=descent_rate,
            min_value=-plane.dsc_rate,
            max_value=plane.asc_rate,
            max_change=plane.acc_z_max
        )
        
        # Handle speed control during descent
        if plane.gspd > plane.ldg_speed:
            plane.acc_xy = plane.proportional_change(
                current=plane.acc_xy,
                target=plane.desired_acc_xy,
                min_value=-plane.acc_xy_max,
                max_value=plane.acc_xy_max,
                max_change=plane.acc_xy_max
            )
        else:
            plane.acc_xy = plane.proportional_change(
                current=plane.acc_xy,
                target=0,
                min_value=-plane.acc_xy_max,
                max_value=plane.acc_xy_max,
                max_change=plane.acc_xy_max
            )
    
    def _handle_ground_phase(self, plane):
        """Handle the ground phase after touchdown."""
        plane.v_z = 0
        plane.acc_xy = -plane.acc_xy_max
    
    def _handle_landing_complete(self, plane, command, tick):
        """Handle landing completion."""
        print(f"Plane {plane.callsign} with ID {plane.id} has landed.")
        plane.acc_xy = 0
        command.command_type = CommandType.NONE
        command.last_update = tick

class TakeoffCommandHandler(CommandHandler):
    """Handler for takeoff commands."""

    def can_handle(self, command_type: CommandType) -> bool:
        return command_type == CommandType.CLEARED_FOR_TAKEOFF
    
    def execute(self, plane, command, tick) -> None:
        print(f"Plane with ID {plane.id} is cleared for takeoff.")

        # Ground roll -> accelerate to minimum takeoff speed
        if plane.gspd < plane.stall_speed:
            plane.acc_xy = plane.proportional_change(
                current=plane.acc_xy,
                target=plane.acc_xy_max,
                min_value=0,
                max_value=plane.acc_xy_max,
                max_change=plane.acc_xy_max
            )
            
        # Climb until 1000 ft
        else:
            command.command_type = CommandType.CRUISE
            print(f"Plane {plane.callsign} takeoff complete. Now cruising.")


class CommandProcessor:
    """Main command processor that delegates to appropriate handlers."""
    
    def __init__(self):
        self.handlers = [
            NoCommandHandler(),
            TurnCommandHandler(),
            LandingCommandHandler(),
            TakeoffCommandHandler(),
			CruiseCommandHandler(),
        ]
    
    def process_command(self, plane, command, tick):
        """Process a command using the appropriate handler."""
        command_type = CommandType.NONE if command is None else command.command_type
        
        for handler in self.handlers:
            if handler.can_handle(command_type):
                handler.execute(plane, command, tick)
                return
        
        raise NotImplementedError(f"Unknown command type: {command_type}")

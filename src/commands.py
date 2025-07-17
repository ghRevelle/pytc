from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from plane import *

class CommandType(Enum):
    LINE_UP_AND_WAIT = 0     # line up on the runway and wait
    CLEARED_FOR_TAKEOFF = 1  # cleared for takeoff; optional argument heading if geography requires tower to dictate what heading to fly after takeoff
    CLEARED_TO_LAND = 2      # cleared to land
    GO_AROUND = 3            # begin go around procedure; simulated behavior based on typical go around procedure, including simulated time loss and return to approach
    ABORT_TAKEOFF = 4        # abort takeoff; assumption: plane moves to the end of the runway to taxiway, reenters queue at the back


@dataclass
class Command:
    command_type: CommandType
    target_id: int                   # index of the plane; planes and their callsigns will be mapped to an integer ID
    argument: Optional[int] = None   # runway ID, optional heading for takeoff

def apply_command(command : Command, plane : SimPlane):
    if command.command_type == CommandType.LINE_UP_AND_WAIT:
        if plane.state['command'] == "holding_short":
            plane.status = "lined_up"
    
    elif command.command_type == CommandType.CLEARED_FOR_TAKEOFF:
        if plane.status == "lined_up":
            plane.status = "takeoff_roll"
            if command.argument:
                plane.heading = command.argument  # only if geography requires it
    
    elif command.command_type == CommandType.CLEARED_TO_LAND:
        if plane.status == "on_approach":
            plane.status = "descending"
    
    elif command.command_type == CommandType.GO_AROUND:
        if plane.status in ["on_approach", "descending"]:
            plane.go_around()  # handles re-entry delay, etc.
    
    elif command.command_type == CommandType.ABORT_TAKEOFF:
        if plane.status == "takeoff_roll":
            plane.abort_takeoff()  # returns to back of queue



from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class CommandType(Enum):
    LINE_UP_AND_WAIT = 0     # line up on the runway and wait
    CLEARED_FOR_TAKEOFF = 1  # cleared for takeoff; optional argument heading if geography requires tower to dictate what heading to fly after takeoff
    CLEARED_TO_LAND = 2      # cleared to land
    GO_AROUND = 3            # begin go around procedure; simulated behavior based on typical go around procedure, including simulated time loss and return to approach
    ABORT_TAKEOFF = 4        # abort takeoff; assumption: plane moves to the end of the runway to taxiway, reenters queue at the back
    TURN = 5                 # rarely used; will use for testing

@dataclass
class Command:
    command_type: CommandType
    target_id: int                   # index of the plane; planes and their callsigns will be mapped to an integer ID
    argument: Optional[int] = None   # runway ID, optional heading for takeoff

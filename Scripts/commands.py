from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class CommandType(Enum):
    NONE = 0                 # no new command
    LINE_UP_AND_WAIT = 1     # line up on the runway and wait
    CLEARED_FOR_TAKEOFF = 2  # cleared for takeoff; optional argument heading if geography requires tower to dictate what heading to fly after takeoff
    CLEARED_TO_LAND = 3      # cleared to land
    GO_AROUND = 4            # begin go around procedure; simulated behavior based on typical go around procedure, including simulated time loss and return to approach
    ABORT_TAKEOFF = 5        # abort takeoff; assumption: plane moves to the end of the runway to taxiway, reenters queue at the back
    TURN = 6                 # rarely used; will use for 
    REALIGN = 7              # realign plane to runway centerline; used for planes that are not aligned with the runway after landing or takeoff
    CRUISE = 8               # default for planes in the air
    TAXI = 9                 # planes which have just landed and are taxiing away

@dataclass
class Command:
    command_type: CommandType
    target_id: int                   # index of the plane; planes and their callsigns will be mapped to an integer ID
    last_update: int                 # when the command was last updated
    argument: Optional[int] = None   # runway ID, optional heading for takeoff

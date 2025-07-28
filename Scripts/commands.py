from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from airport import Runway

class CommandType(Enum):
    NONE = 0                 # no new command
    LINE_UP_AND_WAIT = 1     # line up on the runway and wait
    CLEARED_TO_LAND = 2      # cleared to land
    GO_AROUND = 3            # begin go around procedure; simulated behavior based on typical go around procedure, including simulated time loss and return to approach
    ABORT_TAKEOFF = 4        # abort takeoff; assumption: plane moves to the end of the runway to taxiway, reenters queue at the back
    TURN = 5                 # rarely used; will use for 
    REALIGN = 6              # realign plane to runway centerline; used for planes that are not aligned with the runway after landing or takeoff
    CRUISE = 7               # default for planes in the air
    TAXI = 8                 # planes which have just landed and are taxiing away
    CLEARED_FOR_TAKEOFF = 9  # cleared for takeoff; optional argument heading if geography requires tower to dictate what heading to fly after takeoff

@dataclass
class Command:
    command_type: CommandType
    target_id: int                    # index of the plane; planes and their callsigns will be mapped to an integer ID
    last_update: int                  # when the command was last updated
    argument: Optional[Runway] = Runway((32.73, -117.175), (32.73713, -117.20433), 27) # runway object

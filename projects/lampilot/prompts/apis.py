# Ego APIs

def get_ego_vehicle() -> Vehicle:
    """
    Return the ego vehicle.
    """


def get_desired_time_headway() -> float:
    """
    Return the desired time headway,
    IDM parameter in seconds,
    the minimum possible time to the vehicle in front,
    default 1.5.
    """


def get_target_speed() -> float:
    """
    Return the target speed,
    IDM parameter in m/s,
    the velocity the vehicle would drive at in free traffic.
    """


def say(text: str):
    """Provide a text message to the passenger.
    >>> say("Making a left lane change.")
    """


def is_safe_enter(lane: Lane, safe_decel: float = 5) -> bool:
    """
    Return True if the ego vehicle can safely enter the specified lane from the current position now, False otherwise.
    safe_decel is the maximum deceleration that both the ego vehicle and the rear vehicle in the target lane can afford.
    """


# Control APIs

def set_desired_time_headway(desired_time_headway: float):
    """
    >>> set_desired_time_headway(1.5)
    """


def set_target_speed(target_speed: float):
    """
    >>> set_target_speed(30)
    """


def set_target_lane(target_lane: Lane):
    """
    >>> set_target_lane(target_lane)
    """


def autopilot() -> List[float, float]:
    """
    Return a control command (acceleration, steering angle) according to the current policy.
    By default it follows the IDM model,
    it will stop the car if there is a stop sign or a red traffic light.
    """


def recover_from_stop():
    """
    Resume driving after stopping at a stop sign or a red traffic light.
    >>> recover_from_stop()
    """


# Perception APIs
def get_speed_of(veh: Vehicle) -> float:
    """
    Return the speed of the vehicle in m/s.
    """


def get_lane_of(veh: Vehicle) -> Lane:
    """
    Return the lane of the vehicle.
    """


def detect_front_vehicle_in(lane: Lane, distance: float = 100) -> Vehicle:
    """
    Return the closest vehicle in front of the ego vehicle in the specified lane within the specified distance,
    None if no vehicle.
    """


def detect_rear_vehicle_in(lane: Lane, distance: float = 100) -> Vehicle:
    """
    Return the closest vehicle behind the ego vehicle in the specified lane within the specified distance,
    None if no vehicle.
    """


def get_distance_between_vehicles(veh1: Vehicle, veh2: Vehicle) -> float:
    """
    Return the distance between two vehicles in meters, positive if veh1 is in front of veh2, negative otherwise.
    """


def get_left_lane(veh: Vehicle) -> Lane:
    """
    Return the left lane of the vehicle, None if no lane is detected.
    """


def get_right_lane(veh: Vehicle) -> Lane:
    """
    Return the right lane of the vehicle, None if no lane is detected.
    """


def get_left_to_right_cross_traffic_lanes() -> List[Lane]:
    """
    Return the left to right cross traffic lanes of the ego vehicle, Empty list if no lane is detected.
    """


def get_right_to_left_cross_traffic_lanes() -> List[Lane]:
    """
    Return the right to left cross traffic lanes of the ego vehicle, Empty list if no lane is detected.
    """


def detect_stop_sign_ahead() -> float:
    """
    Return the distance to the stop sign ahead, -1 if no stop sign is detected.
    """


# Route APIs

def turn_left_at_next_intersection():
    """
    Change the planned route to turn left at the next intersection.
    >>> turn_left_at_next_intersection()
    """


def turn_right_at_next_intersection():
    """
    Change the planned route to turn right at the next intersection.
    >>> turn_right_at_next_intersection()
    """


def go_straight_at_next_intersection():
    """
    Change the planned route to go straight at the next intersection.
    >>> go_straight_at_next_intersection()
    """

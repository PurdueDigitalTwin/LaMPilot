def overtake_using_left_lane():
    """
    Command: Go around the car in front of you using the left lane.
    Context Info: My current speed is 31.3 m/s. I am driving on a highway with 2 lanes in my direction, and I am in the 1st lane from the right. There is a car in front of me in my lane, at a distance of 94.8 m, with a speed of 29.3 m/s.
    Returns a generator.
    """

    # Check if there is a left lane available
    ego_vehicle = get_ego_vehicle()
    current_lane = get_lane_of(ego_vehicle)
    left_lane = get_left_lane(ego_vehicle)
    target_vehicle = detect_front_vehicle_in(current_lane)

    if left_lane is None:
        say("There is no left lane to change into.")
        return
    if target_vehicle is None:
        say("There is no vehicle in front of me.")
        return

    # Check if it is safe to enter the left lane
    while True:
        if is_safe_enter(left_lane):
            set_target_lane(left_lane)
            break
        yield autopilot()

    # Monitor the distance to the target vehicle
    while True:
        distance_to_target = get_distance_between_vehicles(ego_vehicle, target_vehicle)
        if distance_to_target < 0:  # The target vehicle is still in front of the ego vehicle
            set_target_speed(get_speed_of(target_vehicle) * 1.5)
            yield autopilot()
        else:
            break
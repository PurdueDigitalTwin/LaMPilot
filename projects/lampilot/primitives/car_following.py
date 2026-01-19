def decrease_gap_to_25_meters():
    """
    Command: Decrease the gap to 25 meters.
    Context Info: My current speed is 31.3 m/s. I am driving on a highway with 2 lanes in my direction, and I am in the 1st lane from the right. There is a car in front of me in my lane, at a distance of 57.7 m, with a speed of 21.6 m/s.
    Returns a generator.
    """
    ego_vehicle = get_ego_vehicle()
    current_lane = get_lane_of(ego_vehicle)
    desired_gap = 25
    while True:
        front_vehicle = detect_front_vehicle_in(current_lane)
        if front_vehicle:
            front_vehicle_speed = get_speed_of(front_vehicle)
            set_target_speed(front_vehicle_speed * 1.2)
            current_time_headway = get_desired_time_headway()
            current_gap = get_distance_between_vehicles(front_vehicle, ego_vehicle)
            if current_gap > desired_gap:
                set_desired_time_headway(current_time_headway - 0.05)
            else:
                set_desired_time_headway(current_time_headway + 0.05)
            yield autopilot()
        else:
            return

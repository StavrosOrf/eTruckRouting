def reward_move_to_next_node():
    """Small reward for moving to next customer node."""
    return 2.0


def reward_finish_charging():
    """Medium reward for completing charging."""
    return 5.0


def reward_arrive_destination():
    """Big reward for reaching destination."""
    return 50.0


def penalty_wait_at_charger(wait_time: float):
    """Small penalty based on waiting time at charging station."""
    return -0.5 * wait_time


def penalty_run_out_of_energy():
    """Big penalty for running out of energy."""
    return -100.0


def penalty_time_elapsed(time_elapsed: float):
    """Penalty to optimize for minimum time."""
    return -0.1 * time_elapsed


def reward_efficient_route(distance_saved: float):
    """Bonus for taking efficient routes."""
    return 0.5 * distance_saved

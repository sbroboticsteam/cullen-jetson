# basic setup
# won't exist in actual code, but is needed as placeholder right now
import random
from math import cos, sin


class PathSegment:
    def __init__(self, x, y, angle, velocity, dt):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = velocity
        self.dt = dt


class Position:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle


def get_current_position():
    return Position(0, 0, 0)


def update_wheels(left, right):
    pass


# Actual Ramsete Code:

# aggressiveness > 0, damping >= 0, <= 1
aggr = 0
damp = 0
width = 0  # width between wheels aka wheelbase


def initialize_constants(aggressiveness_factor, damping_factor, robot_width):
    aggr = aggressiveness_factor
    damp = damping_factor
    width = robot_width


def calc_next_wheel_update(desired: PathSegment, is_last=False):
    left = 0
    right = 0
    if is_last:
        # if this is the last part of the path, slow to a stop.
        update_wheels(left, right)
        return

    # otherwise, there's some math to do
    current = get_current_position()  # current position

    # turning rate is angle over dt
    desired_turning = desired.angle / desired.dt
    # calculate a factor that uses the damping factor
    damper = calc_damper(desired.velocity, desired_turning)
    # room for optimization: some repeated calculations here....but it would be even more ugly
    velocity = calc_velocity(desired, current, damper)  # calculate velocity command
    turning_rate = calc_turning_rate(desired, current, damper, desired_turning)  # calculate turning command

    # calculate left and right velocities based on velocity and turning commands
    left = (-width * turning_rate) / 2 + velocity
    right = (+width * turning_rate) / 2 + velocity

    update_wheels(left, right)


def calc_damper(desired_velocity, desired_turning):
    return 2 * damp * ((desired_turning ** 2 + (aggr * desired_velocity ** 2)) ** 1 / 2)


def calc_velocity(desired: PathSegment, current: Position, damper):
    return desired.velocity * cos(desired.angle - current.angle) + \
           damper * ((desired.x - current.x) * cos(current.angle) + (desired.y - current.y) * sin(current.angle))


def calc_turning_rate(desired: PathSegment, current: Position, damper, desired_turning):
    return desired_turning + aggr * desired.velocity * sinc(desired.angle - current.angle) * \
           ((desired.y - current.y) * cos(current.angle) - (desired.x - current.x) * sin(current.angle)) + \
           damper * (desired.angle - current.angle)


def sinc(x):
    return sin(x) / x
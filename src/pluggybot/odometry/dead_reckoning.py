# This script provides dead reckoning using change in wheel angle over time.

import math


class dead_reckoner:
  def __init__(self, wheel_radius, track_width):
    self.wheel_radius = wheel_radius
    self.track_width = track_width
    self.x = 0
    self.y = 0
    self.theta = 0
  
  def update(self, left_ang_vel, right_ang_vel, dt):

    # Velocity is the average of wheel surface speeds
    velocity = self.wheel_radius * (left_ang_vel + right_ang_vel) / 2

    # Angular velocity (of the bot) is difference of wheel speeds, divided by track width
    bot_ang_vel = self.wheel_radius * (right_ang_vel - left_ang_vel) / self.track_width

    # Update position + angle info
    self.theta += bot_ang_vel * dt
    self.x += velocity * math.cos(bot_ang_vel[0]) * dt
    self.y += velocity * math.sin(bot_ang_vel[0]) * dt


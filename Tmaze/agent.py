#!/usr/bin/env python
import math
import numpy as np
import time
from typing import Tuple, List

Point = Tuple[float, float]
Rect  = Tuple[float, float, float, float] # (xmin, ymin, xmax, ymax)

SQRT2 = math.sqrt(2.0)

## Corner sensor logic
def _t_bound(dx: int, dy: int,
             xc: float, yc: float,
             W: float) -> float:
    # Earliest point at which the ray exits the outer bound
    tx = (W - xc) if dx > 0 else xc
    ty = (W - yc) if dy > 0 else yc
    return min(tx, ty) # first contact with the square


def _first_hit_on_ray(dx: int, dy: int,
                      center: Point,
                      rects: List[Rect],
                      W: float) -> Tuple[str, float, Point]:
    """
    Returns: ('rect' | 'boundary', t, (x, y))
        what: what is hit first
        t: parameter along the ray
        point: contact coordinates
    """
    xc, yc   = center
    t_best   = _t_bound(dx, dy, xc, yc, W)
    hit_kind = "boundary"
    hit_pt   = (xc + dx * t_best, yc + dy * t_best)

    for xmin, ymin, xmax, ymax in rects:
        # --- slab test on x ---
        t0x = (xmin - xc) / dx
        t1x = (xmax - xc) / dx
        t_entry_x, t_exit_x = sorted((t0x, t1x))

        # --- slab test on y ---
        t0y = (ymin - yc) / dy
        t1y = (ymax - yc) / dy
        t_entry_y, t_exit_y = sorted((t0y, t1y))

        # entry/exit for the intersection of both slabs
        t_entry = max(t_entry_x, t_entry_y)
        t_exit  = min(t_exit_x,  t_exit_y)

        # Does the ray actually hit the rectangle?
        if t_exit >= 0 and t_entry <= t_exit and t_entry >= 0:
            if t_entry < t_best:  # blocks earlier → keep it
                t_best   = t_entry
                hit_kind = "rect"
                hit_pt   = (xc + dx * t_best, yc + dy * t_best)

    return hit_kind, t_best, hit_pt

class XYAgent:
    def __init__(self, init_pos=np.array([1.5,1.5]), max_delta=0.5, agent_size=0.1, world=None, max_sensor_range=0.0):
        self.init_pos = init_pos.copy()
        self.pos = init_pos.copy()
        self.max_delta = max_delta
        self.size = agent_size
        self.world = world
        self.max_sensor_range = max_sensor_range

    @property
    def sense(self):
        if self.world is None:
            return None
        else:
            return self.world.get_color(self.pos)

    def move(self, pos, force=False, update=True, delta_pos=False):
        pos = np.asarray(pos)
        vec_multiplier = 1.0
        if not delta_pos:
            delta = self.pos-pos
        else:
            delta = pos
        if not force and np.any(np.abs(delta) > self.max_delta):
            vec_multiplier = self.max_delta / np.linalg.norm(delta)
        if self.world is not None and not force:
            next_pos = self.world.collision_check(self.pos, self.pos-(vec_multiplier*delta), padding=self.size)
        else:
            next_pos = self.pos - (vec_multiplier*delta)
        if update:
            self.pos = next_pos
        return next_pos if not delta_pos else (vec_multiplier*delta)

    def reset(self):
        self.pos = self.init_pos.copy()

    @property
    def corner_sensors(self):
        # Analyze all four rays and return distance to hit
        rays = {"045": ( 1,  1),
                "315": (-1,  1),
                "135": ( 1, -1),
                "225": (-1, -1)}
        corner_sensors = {}

        for name, (dx, dy) in rays.items():
            _, t, _ = _first_hit_on_ray(dx, dy, (self.pos[0], self.pos[1]), self.world.walls, self.world.width) # ignoring what and ipt
            corner_sensors[name] = min((t * SQRT2), self.max_sensor_range)
        return corner_sensors

class SoftPID:
    def __init__(self, Kp, Ki=0.0, Kd=0.0, windup=None, errsum_decay=0.99, bias=None, active_threshold=None):
        """
        General multidimensional PID controller

        @Kp Proportional coefficient
        @Ki Integral coefficient
        @Kd Derivative coefficient
        @windup Windup guard
        @errsum_decay Integral error decay
        @bias Constant offset
        @active_threshold Disable controller when max error falls below this threshold
        """
        self.active = True
        self.Kp = np.asarray(Kp)
        self.Ki = np.asarray(Ki)
        self.Kd = np.asarray(Kd)
        self.errsum = np.zeros_like(self.Ki)
        self.windup = np.asarray(windup) if windup is not None else np.asarray(Ki*2)
        self.errsum_decay = errsum_decay
        self.bias = bias if bias is not None else np.zeros_like(self.Kp)
        self.err_threshold = active_threshold
        self.last_err = np.zeros_like(self.Kd)
        self.last_time = time.monotonic() * 1000 # ms

    def reset(self, active=True):
        self.active = active
        self.errsum = np.zeros_like(self.errsum)
        self.last_err = np.zeros_like(self.last_err)
        self.last_time = time.monotonic() * 1000 # ms

    def move(self, current_pos, target_pos, delta_time=None):
        """
        Takes the feedback value, setpoint and returns a new process value

        @current_pos Currently measured process value (feedback value)
        @target_pos Set point
        @delta_time Interval since last move. Leave as None to calculate from last move call
        """
        if self.active:
            sp = np.asarray(target_pos)
            fv = np.asarray(current_pos)
            now = time.monotonic() * 1000 # ms
            dt = delta_time if delta_time is not None else now - self.last_time
            err = sp - fv
            if self.err_threshold is not None and max(abs(err)) < self.err_threshold:
                pv = fv
                self.reset()
            else:
                self.errsum = np.clip((self.errsum_decay*self.errsum) + (err*dt), a_min=-self.windup, a_max=self.windup)
                derr = (err-self.last_err)/dt
                pv = fv + self.bias + (self.Kp*err) + (self.Ki*self.errsum) + (self.Kd*derr)
                self.last_err = err
                self.last_time = now
        else:
            pv = np.asarray(target_pos)
            self.reset(active=False)
        return pv

    def move_future(self, start_pos, targets, keep_steps=1, delta_time=None):
        """
        From the current feedback value, given a list of targets, return a list of process values.
        Process value is fedback directly for future steps

        @start_pos Currently measured process value (feedback value)
        @targets List of setpoints
        @keep_steps Assume this step is used, save error values
        @delta_time Interval since last move. Leave as None to calculate from last move call
        """
        if self.active:
            errsum = self.errsum
            last_err = self.last_err
            output = []
            input = start_pos
            for move, target in enumerate(targets):
                sp = np.asarray(target)
                fv = np.asarray(input)
                now = time.monotonic() * 1000 # ms
                if move == 0:
                    dt = delta_time if delta_time is not None else now - self.last_time # dt is set at the beginning
                err = sp - fv
                errsum = np.clip((self.errsum_decay*errsum) + (err*dt), a_min=-self.windup, a_max=self.windup)
                derr = (err-last_err) / dt
                last_err = err
                if move == keep_steps-1:
                    self.errsum = errsum
                    self.last_err = last_err
                    self.last_time = now
                pv = fv + self.bias + (self.Kp*err) + (self.Ki*errsum) + (self.Kd*derr)
                output.append(pv)
                input = pv # loopback
        else:
            output = targets
            self.reset(active=False)
        return output

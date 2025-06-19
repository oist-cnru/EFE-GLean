#!/usr/bin/env python
import math
import numpy as np

rng = np.random.default_rng()

from typing import Tuple, Optional, List
Point = Tuple[float, float]
Rect  = Tuple[float, float, float, float] # xmin, ymin, xmax, ymax

def _point_in_rect(px, py, rect):
    xmin, ymin, xmax, ymax = rect
    return xmin <= px <= xmax and ymin <= py <= ymax

def _segment_intersection(p1, p2, q1, q2, eps=1e-12):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = q1; x4,y4 = q2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < eps:
        return None, None                    # parallel
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))/denom
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2))/denom
    if 0 <= t <= 1 and 0 <= u <= 1:          # on both segments
        return (x1 + t*(x2-x1), y1 + t*(y2-y1)), t
    return None, None

def last_safe_circle_position(p1: Point, p2: Point,
                              rect: Rect, r: float,
                              eps: float = 1e-9) -> Optional[Point]:
    """
    Returns the furthest point on the path p1, p2 where a circle of radius r is still completely outside the rectangle
    If the circle never collides, returns p2
    If the circle starts already intersecting, returns p1
    """
    # 1  Grow rectangle by r
    grown = (rect[0]-r, rect[1]-r, rect[2]+r, rect[3]+r)

    # 2  No collision at the end? – we’re done
    if not _point_in_rect(*p2, grown):
        return p2

    # 3  Find earliest entry parameter t∈[0,1]
    dx, dy = (p2[0]-p1[0], p2[1]-p1[1])
    edges = [((grown[0], grown[1]), (grown[0], grown[3])),  # left
             ((grown[0], grown[3]), (grown[2], grown[3])),  # top
             ((grown[2], grown[3]), (grown[2], grown[1])),  # right
             ((grown[2], grown[1]), (grown[0], grown[1]))]  # bottom

    ts: List[float] = []
    for q1, q2 in edges:
        _, t = _segment_intersection(p1, p2, q1, q2)
        if t is not None:
            ts.append(t)

    if not ts:                             # path starts inside grown rect
        return p1                          # already colliding at start

    t_entry = min(ts)                      # earliest contact
    safe_t  = max(0.0, t_entry - eps)
    return (p1[0] + safe_t*dx,
            p1[1] + safe_t*dy)

def check_overlap(rect_x, rect_y, rect_w, rect_h, circ_x, circ_y, circ_r):
    """
    Check if a rectangle and a circle overlap.

    Args:
        rect_x (float): x-coordinate of the bottom left corner of the rectangle.
        rect_y (float): y-coordinate of the bottom left corner of the rectangle.
        rect_w (float): width of the rectangle.
        rect_h (float): height of the rectangle.
        circ_x (float): x-coordinate of the center of the circle.
        circ_y (float): y-coordinate of the center of the circle.
        circ_r (float): radius of the circle.

    Returns:
        bool: Whether the rectangle and the circle overlap.
    """

    # Find the closest point to the circle within the rectangle
    closest_x = max(rect_x, min(circ_x, rect_x + rect_w))
    closest_y = max(rect_y, min(circ_y, rect_y + rect_h))

    # Calculate the distance between the circle's center and this point
    distance_x = circ_x - closest_x
    distance_y = circ_y - closest_y

    # If the distance is less than or equal to the circle's radius, they overlap
    return math.sqrt(distance_x**2 + distance_y**2) <= circ_r

class TMaze:
    def __init__(self, size=(3.0,3.0), bwidth=1.0, goal_lr=None, rgb_color=False):
        # Colors
        if rgb_color:
            self.red = [1.0, 0.0, 0.0]
            self.green = [0.0, 1.0, 0.0]
            self.blue = [0.0, 0.0, 1.0]
            self.clear = [1.0, 1.0, 1.0]
        else:
            self.red = [0, 0, 1, 0]
            self.green = [0, 1, 0, 0]
            self.blue = [1, 0, 0, 0]
            self.clear = [0, 0, 0, 1]

        # Goal position (Left, Right)
        self.goal_lr = None

        # Outer bounds
        self.width, self.height = size
        self.path_width = bwidth
        # Inner walls
        # self.x_walls = [[0.0, (self.width/2)-(bwidth/2)], [(self.width/2)+(bwidth/2), self.width]]
        # self.y_walls = [0.0, self.height-bwidth]
        self.walls = [(0.0, 0.0, (self.width/2)-(bwidth/2), self.height-bwidth), ((self.width/2)+(bwidth/2), 0.0, self.width, self.height-bwidth)] # list of (xmin, ymin, xmax, ymax) defining obstacles: left wall, right wall, (optional) obstacles

        # Colored regions [(xmin,ymin,xmax,ymax),color] (colors are defined later)
        self.colors = [{"pos":(0.0,self.height-bwidth,bwidth/2,self.height),"color":self.clear}, {"pos":(self.width-bwidth/2,self.height-bwidth,self.width,self.height),"color":self.clear}, {"pos":((self.width/2)-(bwidth/2),0.0,(self.width/2)+(bwidth/2),bwidth/2),"color":self.clear}]

        self.set_goal(goal_lr)

    def set_goal(self, lr=None, invert_cs=False):
        if lr is None:
            if rng.random() < 0.5:
                self.goal_lr = [1, 0]
            else:
                self.goal_lr = [0, 1]
        elif len(lr) == 2:
            self.goal_lr = lr
        else:
            if lr == "left":
                self.goal_lr = [1, 0]
            elif lr == "right":
                self.goal_lr = [0, 1]
        # BC: Blue => Goal TL, Green => Goal TR
        if self.goal_lr[0] == 1:
            self.colors[0]["color"] = self.red
            self.colors[1]["color"] = self.clear
            self.colors[2]["color"] = self.blue if not invert_cs else self.green
        elif self.goal_lr[1] == 1:
            self.colors[0]["color"] = self.clear
            self.colors[1]["color"] = self.red
            self.colors[2]["color"] = self.green if not invert_cs else self.blue

    def set_obstacle(self, x, y, w=None, h=None):
        if w is None:
            w = self.path_width/2
        if h is None:
            h = w
        if len(self.walls) < 3:
            self.walls.append(None)
        self.walls[-1] = (x, y, x+w, y+h)

    def rm_obstacle(self):
        if len(self.walls) >= 3:
            obs = self.walls.pop()
        else:
            obs = None
        return obs
    
    def random_goal(self):
        self.set_goal()
        return [1, 0] if self.tlr_color[0][-1] == self.red else [0, 1]

    def collision_check(self, cur_pos, tgt_pos, end_pos_only=False, padding=0.1):
        # Check outer bounds
        if cur_pos[0]-padding < 0.0 or cur_pos[0]+padding > self.width or cur_pos[1]-padding < 0.0 or cur_pos[1]+padding > self.height:
            print("Invalid starting position", cur_pos)
            return np.asarray(cur_pos)
        if tgt_pos[0]-padding < 0.0 or tgt_pos[0]+padding > self.width or tgt_pos[1]-padding < 0.0 or tgt_pos[1]+padding > self.height:
            tgt_pos[0] = max(padding, min(tgt_pos[0], self.width-padding))
            tgt_pos[1] = max(padding, min(tgt_pos[1], self.height-padding))
            # print("Clipped target position to outer bounds", tgt_pos)
        
        if end_pos_only:
            # Only check if the end pos overlaps a wall (i.e. no teleporting into a wall)
            teleport_failed = False
            for w in self.walls:
                if check_overlap(w[0], w[1], w[2]-w[1], w[3]-w[1], tgt_pos[0], tgt_pos[1], padding):
                    teleport_failed = True
                    break
            if not teleport_failed:
                return np.asarray(tgt_pos)
        
        # Check if the path crosses an internal wall
        for w in self.walls:
            chk_pos = last_safe_circle_position(cur_pos, tgt_pos, (w[0], w[1], w[2], w[3]), padding)
            if not np.allclose(chk_pos, tgt_pos):
                # print("Collision at", chk_pos)
                tgt_pos = chk_pos
        return np.asarray(tgt_pos)

    def get_color(self, pos, padding=0.1):
        # Check whether the agent overlaps with one of the colored sections
        for c in self.colors:
            if check_overlap(c["pos"][0], c["pos"][1], c["pos"][2]-c["pos"][0], c["pos"][3]-c["pos"][1], pos[0], pos[1], padding):
                return c["color"]
        # Default color
        return self.clear
    
    def convert_color(self, color):
        if type(color) == "str":
            if color.lower() == "red":
                return self.red
            elif color.lower() == "blue":
                return self.blue
            elif color.lower() == "green":
                return self.green
            elif color.lower() == "clear" or color.lower() == "white" or color.lower() == "none":
                return self.clear
        else:
            if color == self.red:
                return "red"
            elif color == self.blue:
                return "blue"
            elif color == self.green:
                return "green"
            elif color == self.clear:
                return "white"
        return None

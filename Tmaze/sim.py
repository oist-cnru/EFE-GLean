import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

def bearing_to_angle(bearing_deg):
    # Bearing (° from North, clockwise) → Cartesian angle (° from +x, CCW)
    return (450.0 - bearing_deg) % 360.0

def angle_to_bearing(angle_deg):
    # Cartesian angle (° from +x, CCW) → Bearing (° from North, clockwise)
    return (450.0 - angle_deg) % 360.0

class Sim2D:
    def __init__(self, world, robot, paths=0, path_length=60, points=0, largetxt=False):
        self.world = world
        self.agent = robot
        self.bg = None
        self.draw_walls = []
        self.draw_colors = []
        self.draw_agent = None
        # Convert agent sensor data for visualization
        self.dsensors = {}
        self.draw_dsensors = []
        self.draw_paths = []
        self.draw_points = []

        ## Setup plot
        if largetxt:
            font = {"size": 16}
            plt.rc("font", **font)
        self.fig, self.ax = plt.subplots(figsize=(9,9))
        self.ax.set_xlim(left=0.0, right=self.world.width)
        self.ax.set_ylim(bottom=0.0, top=self.world.height)
        self.fig.set_dpi(100)
        if largetxt:
            self.fig.tight_layout()

        # Internal walls
        for w in self.world.walls:
            self.draw_walls.append(patches.Rectangle([w[0], w[1]], w[2]-w[0], w[3]-w[1], facecolor="0.6"))
            self.ax.add_patch(self.draw_walls[-1])

        # Save background
        self.fig.show()
        self.fig.canvas.start_event_loop(0.01)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox) # save static bg
        
        # Agent
        if self.agent is not None:
            self.draw_agent = patches.Circle(robot.pos, 0.1, edgecolor="black", facecolor="white")
            self.ax.add_patch(self.draw_agent)

        # Corner sensors
        if self.agent.max_sensor_range > 0.0:
            for k, mag in self.agent.corner_sensors.items():
                # Compute limited vectors
                theta = math.radians(bearing_to_angle(float(k)))
                vec = (mag*math.cos(theta)+self.agent.pos[0], mag*math.sin(theta)+self.agent.pos[1])
                self.dsensors[str(int(math.degrees(theta)))] = vec

                self.draw_dsensors.append(lines.Line2D([self.agent.pos[0], vec[0]], [self.agent.pos[1], vec[1]], color="pink", linestyle=':', linewidth=2))
                self.ax.add_line(self.draw_dsensors[-1])

        # Colored floors
        for c in self.world.colors:
            self.draw_colors.append(patches.Rectangle([c["pos"][0], c["pos"][1]], c["pos"][2]-c["pos"][0], c["pos"][3]-c["pos"][1], facecolor=self.world.convert_color(c["color"]), zorder=0))
            self.ax.add_patch(self.draw_colors[-1])
        
        # Paths
        for _ in range(paths):
            self.draw_paths.append(self.ax.plot(np.zeros(path_length), np.zeros(path_length), animated=False, alpha=0.0)[0])
        
        for _ in range(points):
            self.draw_points.append(patches.Circle((0, 0), 0.04, color="black", linestyle=":", alpha=0.0))
            self.ax.add_patch(self.draw_points[-1])

    def add_wall(self, idx=-1, color="0.6"):
        w = self.world.walls[idx]
        self.draw_walls.append(patches.Rectangle([w[0], w[1]], w[2]-w[0], w[3]-w[1], facecolor=color))
        self.ax.add_patch(self.draw_walls[-1])
    
    def toggle_wall_visibility(self, idx=-1, visible=True):
        if visible:
            self.draw_walls[idx].set(alpha=1.0)
        else:
            self.draw_walls[idx].set(alpha=0.0)

    def draw(self, redraw_agent=True, redraw_colors=False, redraw_walls=False, paths=[], highlight_path=-1, dashed_path=-1, points=[], block_time=0.1, savepath=None):
        self.fig.canvas.restore_region(self.bg)
        if redraw_agent:
            self.draw_agent.set_center(self.agent.pos)
            self.ax.draw_artist(self.draw_agent)
            if self.agent.max_sensor_range > 0.0:
                for d,kv in zip(self.draw_dsensors, self.agent.corner_sensors.items()):
                    k, mag = kv
                    # Compute limited vectors
                    theta = math.radians(bearing_to_angle(float(k)))
                    vec = (mag*math.cos(theta)+self.agent.pos[0], mag*math.sin(theta)+self.agent.pos[1])
                    self.dsensors[str(int(math.degrees(theta)))] = vec

                    d.set_data([self.agent.pos[0], vec[0]], [self.agent.pos[1], vec[1]])
                    self.ax.draw_artist(d)
        if redraw_colors:
            for d,c in zip(self.draw_colors, self.world.colors):
                d.set_facecolor(self.world.convert_color(c["color"]))
                self.ax.draw_artist(d)
        if redraw_walls:
            for d,w in zip(self.draw_walls, self.world.walls):
                d.set_bounds(w[0], w[1], w[2]-w[0], w[3]-w[1])
                self.ax.draw_artist(d)
        for idx,p in enumerate(paths):
            if idx >= len(self.draw_paths):
                break
            self.draw_paths[idx].set_data(p[0], p[1])
            if idx == highlight_path:
                self.draw_paths[idx].set(color="black", alpha=1.0, zorder=99)
            elif idx == dashed_path:
                self.draw_paths[idx].set(color="orange", linestyle="--", linewidth=3, alpha=1.0, zorder=100)
            else:
                self.draw_paths[idx].set(color="gray", alpha=0.1)
        for idx,p in enumerate(points):
            if idx >= len(self.draw_points):
                break
            self.draw_points[idx].set_center(p)
            self.draw_points[idx].set(alpha=0.8)
        # self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(block_time)

        if savepath is not None:
            self.fig.savefig(savepath)

    def close(self):
        plt.close(self.fig)

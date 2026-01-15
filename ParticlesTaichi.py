import taichi as ti
import numpy as np


@ti.data_oriented
class ParticlesTaichi:
    def __init__(self, count, sim_width, sim_height, cell_size):
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.cell_size = cell_size

        self.screen_w = sim_width * cell_size
        self.screen_h = sim_height * cell_size

        # Initialize Lanes
        line_spacing = 9
        num_lines = int(sim_height // line_spacing)
        self.num_lines = num_lines

        particles_per_line = count // num_lines
        total_count = particles_per_line * num_lines
        self.count = total_count

        unique_lanes_np = np.linspace(
            2, sim_height-2, num_lines).astype(np.float32)

        # Create Initial Positions
        y_np = np.repeat(unique_lanes_np, particles_per_line)
        x_np = np.random.rand(total_count).astype(np.float32) * sim_width
        pos_np = np.stack((x_np, y_np), axis=1)

        # Taichi Fields
        self.pos = ti.Vector.field(2, dtype=float, shape=total_count)
        self.pos.from_numpy(pos_np)

        self.lanes = ti.field(dtype=float, shape=num_lines)
        self.lanes.from_numpy(unique_lanes_np)

        self.screen_buf = ti.Vector.field(
            3, dtype=ti.u8, shape=(self.screen_w, self.screen_h))

    @ti.func
    def get_respawn_pos(self):
        lane_idx = int(ti.random() * self.num_lines)
        if lane_idx >= self.num_lines:
            lane_idx = self.num_lines - 1

        y = self.lanes[lane_idx]
        x = ti.random() * (self.sim_width * 0.1)

        return ti.Vector([x, y])

    @ti.kernel
    def update(self, u: ti.template(), cylinder: ti.template()):
        for i in self.pos:
            p = self.pos[i]

            ix, iy = int(p.x), int(p.y)

            # Check Bounds or Walls
            if (ix < 0 or ix >= self.sim_width or
                iy < 0 or iy >= self.sim_height or
                    cylinder[ix, iy] == 1):

                self.pos[i] = self.get_respawn_pos()
                continue

            # Advect
            vel = u[ix, iy]
            p += vel

            # Wrap Y
            if p.y < 0:
                p.y += self.sim_height
            if p.y >= self.sim_height:
                p.y -= self.sim_height

            self.pos[i] = p

    @ti.kernel
    def render(self, u: ti.template(), cylinder: ti.template(), max_speed: float):
        # Draw Background or Walls
        for i, j in self.screen_buf:
            sim_x = i // self.cell_size
            sim_y = j // self.cell_size

            if sim_x < self.sim_width and sim_y < self.sim_height:
                if cylinder[sim_x, sim_y] == 1:
                    self.screen_buf[i, j] = ti.Vector(
                        [100, 100, 100]).cast(ti.u8)
                else:
                    self.screen_buf[i, j] = ti.Vector([0, 0, 0]).cast(ti.u8)

        # Draw Particles
        for i in self.pos:
            sx = int(self.pos[i].x * self.cell_size)
            sy = int(self.pos[i].y * self.cell_size)

            if 0 <= sx < self.screen_w and 0 <= sy < self.screen_h:

                ix, iy = int(self.pos[i].x), int(self.pos[i].y)

                vel = ti.Vector([0.0, 0.0])
                if 0 <= ix < self.sim_width and 0 <= iy < self.sim_height:
                    vel = u[ix, iy]

                speed = vel.norm()
                t = speed / max_speed
                t = min(1.0, max(0.0, t))

                r, g, b = 0, 0, 0
                if t < 0.5:
                    local_t = t * 2.0
                    r = 255
                    g = int(255 * local_t)
                else:
                    local_t = (t - 0.5) * 2.0
                    r = int(255 * (1.0 - local_t))
                    g = 255

                self.screen_buf[sx, sy] = ti.Vector([r, g, b]).cast(ti.u8)

    def export_visuals(self, out_arr):
        out_arr[:] = self.screen_buf.to_numpy()

import taichi as ti
import numpy as np


@ti.data_oriented
class FluidTaichi:
    def __init__(self, width, height, viscosity=0.02):
        self.width = width
        self.height = height

        # Fields
        self.rho = ti.field(dtype=float, shape=(width, height))
        self.u = ti.Vector.field(2, dtype=float, shape=(width, height))
        self.f = ti.Vector.field(9, dtype=float, shape=(width, height))
        self.f_new = ti.Vector.field(9, dtype=float, shape=(width, height))
        self.cylinder = ti.field(dtype=int, shape=(width, height))

        # Scalar Outputs
        self.drag_val = ti.field(dtype=float, shape=())
        self.lift_val = ti.field(dtype=float, shape=())
        self.max_v_sq = ti.field(dtype=float, shape=())

        self.rgb_buf = ti.Vector.field(3, dtype=ti.u8, shape=(width, height))

        # Constants
        self.w = ti.Vector([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.ex = ti.Vector([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.ey = ti.Vector([0, 0, -1, 0, 1, -1, -1, 1, 1])
        self.omega = 1.0 / (3.0 * viscosity + 0.5)

        self.reset()

    def reset(self):
        self.cylinder.fill(0)
        self.init_flow()

    @ti.kernel
    def init_flow(self):
        for i, j in self.rho:
            self.rho[i, j] = 1.0
            self.u[i, j] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                self.f[i, j][k] = self.w[k]
                self.f_new[i, j][k] = self.w[k]

    @ti.kernel
    def set_inlet(self, u_speed: float):
        for j in range(self.height):
            u_vec = ti.Vector([u_speed, 0.0])
            u_sq = u_speed**2
            for k in ti.static(range(9)):
                eu = u_vec.dot(ti.Vector([self.ex[k], self.ey[k]]))
                feq = self.w[k] * 1.0 * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
                self.f[0, j][k] = feq
                self.f[1, j][k] = feq

    @ti.kernel
    def step_kernel(self):
        self.drag_val[None] = 0.0
        self.lift_val[None] = 0.0
        self.max_v_sq[None] = 0.0

        # Streaming
        for i, j in self.f:
            for k in ti.static(range(9)):
                prev_x = (i - self.ex[k] + self.width) % self.width
                prev_y = (j - self.ey[k] + self.height) % self.height
                self.f_new[i, j][k] = self.f[prev_x, prev_y][k]

        # Collision & Forces
        for i, j in self.f_new:
            if self.cylinder[i, j] == 1:
                # Bounce Back
                for k in ti.static(range(9)):
                    inv = k
                    if k == 1:
                        inv = 3
                    elif k == 2:
                        inv = 4
                    elif k == 3:
                        inv = 1
                    elif k == 4:
                        inv = 2
                    elif k == 5:
                        inv = 7
                    elif k == 6:
                        inv = 8
                    elif k == 7:
                        inv = 5
                    elif k == 8:
                        inv = 6

                    val_in = self.f_new[i, j][k]
                    self.f[i, j][inv] = val_in

                    if val_in > 0:
                        dx, dy = self.ex[k], self.ey[k]
                        ti.atomic_add(self.drag_val[None], 2.0 * val_in * dx)
                        ti.atomic_add(self.lift_val[None], 2.0 * val_in * dy)

                self.u[i, j] = ti.Vector([0.0, 0.0])

            else:
                f_vec = self.f_new[i, j]
                rho = f_vec.sum()
                u_vec = ti.Vector([0.0, 0.0])

                for k in ti.static(range(9)):
                    u_vec += ti.Vector([self.ex[k], self.ey[k]]) * f_vec[k]

                if rho > 0:
                    u_vec /= rho

                u_sq = u_vec.norm_sqr()
                ti.atomic_max(self.max_v_sq[None], u_sq)

                for k in ti.static(range(9)):
                    eu = u_vec.dot(ti.Vector([self.ex[k], self.ey[k]]))
                    feq = self.w[k] * rho * \
                        (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
                    self.f[i, j][k] = f_vec[k] + self.omega * (feq - f_vec[k])

                self.rho[i, j] = rho
                self.u[i, j] = u_vec

    @ti.kernel
    def render_visuals(self, mode: int):
        for i, j in self.rgb_buf:
            if self.cylinder[i, j] == 1:
                self.rgb_buf[i, j] = ti.Vector([100, 100, 100]).cast(ti.u8)
            else:
                if mode == 0:  # CURL
                    ip = min(i+1, self.width-1)
                    im = max(i-1, 0)
                    jp = min(j+1, self.height-1)
                    jm = max(j-1, 0)
                    uy_dx = (self.u[ip, j][1] - self.u[im, j][1]) * 0.5
                    ux_dy = (self.u[i, jp][0] - self.u[i, jm][0]) * 0.5
                    curl = uy_dx - ux_dy
                    val = int((curl + 0.1) * 1200)
                    val = max(0, min(255, val))
                    self.rgb_buf[i, j] = ti.Vector(
                        [val, 0, 255 - val]).cast(ti.u8)

                elif mode == 1:  # SPEED
                    spd = self.u[i, j].norm()
                    val = int(spd * 1500)
                    val = max(0, min(255, val))
                    self.rgb_buf[i, j] = ti.Vector([0, val, val]).cast(ti.u8)

                elif mode == 3:  # PRESSURE
                    rho = self.rho[i, j]
                    delta = (rho - 1.0) * 4000.0

                    r, g, b = 0, 0, 0

                    if delta > 0:
                        val = int(min(255, delta))
                        r = val
                        g = int(val * 0.4)
                    else:
                        val = int(min(255, -delta))
                        b = val
                        g = int(val * 0.4)

                    self.rgb_buf[i, j] = ti.Vector([r, g, b]).cast(ti.u8)

    def step(self):
        self.step_kernel()
        return self.drag_val[None], self.lift_val[None], np.sqrt(self.max_v_sq[None])

    def export_visuals(self, out_arr):
        out_arr[:] = self.rgb_buf.to_numpy()

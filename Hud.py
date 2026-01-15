import pygame
import numpy as np


class HUD:
    def __init__(self, display_w, display_h, width, height, cell_size):
        self.dw = display_w
        self.dh = display_h
        self.sim_w = width
        self.sim_h = height
        self.cell_size = cell_size

        self.font = pygame.font.SysFont("consolas", 18)
        self.btn_font = pygame.font.SysFont("consolas", 16, bold=True)

        self.c_text = (200, 200, 200)
        self.c_bg = (20, 20, 40)
        self.c_green = (50, 255, 50)
        self.c_yellow = (255, 255, 0)
        self.c_red = (255, 50, 50)
        self.c_orange = (255, 100, 50)
        self.c_safe_line = (255, 50, 50)

        # Menu Buttons
        box_w, box_h = 400, 160
        self.box_rect = pygame.Rect(
            (display_w - box_w)//2, (display_h - box_h)//2, box_w, box_h)
        self.btn_gen = pygame.Rect(
            self.box_rect.x + 20, self.box_rect.y + 100, 170, 40)
        self.btn_swp = pygame.Rect(
            self.box_rect.x + 210, self.box_rect.y + 100, 170, 40)

        # Lightbox Overlay
        self.overlay_surf = pygame.Surface((display_w, display_h))
        self.overlay_surf.fill((0, 0, 0))

    def render(self, screen, fluid, stats):
        if stats['show_hud']:
            self._draw_dashboard(screen, fluid, stats)
            self._draw_status(screen, stats)
            self._draw_controls(screen)
            self._draw_safety_box(screen, stats['margin_x'], stats['margin_y'])

        if stats['sweep_active']:
            self._draw_sweep_status(screen, stats)

        expansion = stats.get('graph_expansion', 0.0)

        if expansion > 0.01:
            alpha = int(expansion * 160)
            self.overlay_surf.set_alpha(alpha)
            screen.blit(self.overlay_surf, (0, 0))

        if stats['sweep_data']:
            self._draw_graph(
                screen, stats['sweep_data'], expansion, stats['sweep_active'])

        if stats['input_active']:
            self._draw_menu(screen, stats['user_text'])

    def _draw_sweep_status(self, screen, stats):
        cx = self.dw // 2
        cy = 40

        txt_1 = f"COLLECTING DATA | {stats['name']} | {stats['swp_rem_angle']:.0f}s remaining"
        txt_2 = f"Full Sweep: {stats['swp_rem_total']:.0f}s remaining"
        txt_3 = "Press 'X' to Cancel"

        s1 = self.btn_font.render(txt_1, True, self.c_red)
        s2 = self.btn_font.render(txt_2, True, self.c_orange)
        s3 = self.btn_font.render(txt_3, True, (200, 200, 200))

        w = max(s1.get_width(), s2.get_width()) + 40
        h = 85
        bg_rect = pygame.Rect(cx - w//2, cy - 10, w, h)

        s = pygame.Surface((w, h))
        s.set_alpha(220)
        s.fill((10, 10, 20))
        screen.blit(s, bg_rect)
        pygame.draw.rect(screen, self.c_red, bg_rect, 2)

        screen.blit(s1, (cx - s1.get_width()//2, cy))
        screen.blit(s2, (cx - s2.get_width()//2, cy + 25))
        screen.blit(s3, (cx - s3.get_width()//2, cy + 50))

    def _draw_dashed_line(self, surf, color, p1, p2):
        if p1[0] == p2[0]:  # Vertical
            for y in range(p1[1], p2[1], 10):
                if (y // 10) % 2 == 0:
                    pygame.draw.line(surf, color, (p1[0], y), (p2[0], y+10), 2)
        else:  # Horizontal
            for x in range(p1[0], p2[0], 10):
                if (x // 10) % 2 == 0:
                    pygame.draw.line(surf, color, (x, p1[1]), (x+10, p2[1]), 2)

    def _draw_safety_box(self, screen, mx, my):
        xl, xr = mx * self.cell_size, (self.sim_w - mx) * self.cell_size
        yt, yb = my * self.cell_size, (self.sim_h - my) * self.cell_size
        self._draw_dashed_line(screen, self.c_safe_line, (xl, yt), (xl, yb))
        self._draw_dashed_line(screen, self.c_safe_line, (xr, yt), (xr, yb))
        self._draw_dashed_line(screen, self.c_safe_line, (xl, yt), (xr, yt))
        self._draw_dashed_line(screen, self.c_safe_line, (xl, yb), (xr, yb))

    def _draw_dashboard(self, screen, fluid, stats):
        dur = (pygame.time.get_ticks() - stats['start_tick']) / 1000.0
        header = f"AIRFOIL: {stats['name']} | TIME: {dur:.1f}s"
        screen.blit(self.font.render(header, True, (255, 255, 255)), (10, 10))

        peak = stats.get('peak_speed', 0.0)

        s_pct = max(0, min(100, (1.0 - peak/stats['max_speed'])*100))
        c = self.c_green if s_pct > 70 else self.c_yellow if s_pct > 40 else self.c_red

        screen.blit(self.font.render(
            f"SIM STABILITY: {s_pct:.1f}%", True, c), (10, 35))
        pygame.draw.rect(screen, (50, 50, 50), (10, 55, 150, 6))
        pygame.draw.rect(screen, c, (10, 55, int(s_pct * 1.5), 6))

        d_pct = min(100, (pygame.time.get_ticks() -
                    stats['start_tick']) / stats['conv_time'] * 100)
        c = self.c_green if d_pct >= 100 else self.c_yellow if d_pct > 50 else self.c_orange

        screen.blit(self.font.render(
            f"DATA STABILITY: {d_pct:.1f}%", True, c), (10, 70))
        pygame.draw.rect(screen, (50, 50, 50), (10, 90, 150, 6))
        pygame.draw.rect(screen, c, (10, 90, int(d_pct * 1.5), 6))

        y_off = 110
        rd, rl, rw = stats['drag'], stats['lift'], stats['wind']
        screen.blit(self.font.render(
            f"DRAG: {rd:.2f} N", True, (255, 100, 100)), (10, y_off))
        screen.blit(self.font.render(
            f"LIFT: {rl:.2f} N", True, (100, 255, 100)), (10, y_off + 20))
        screen.blit(self.font.render(
            f"WIND: {rw:.1f} m/s", True, (100, 200, 255)), (10, y_off + 40))

    def _draw_status(self, screen, stats):
        scale_txt = f"Time Scale: {stats['time_scale']}"
        screen.blit(self.font.render(scale_txt, True,
                    (150, 255, 255)), (10, self.dh - 45))

        status_state = "PAUSED" if stats['paused'] else "RUNNING"
        final_str = f"FPS: {stats['fps']} | AVG FPS: {stats['avg_fps']} | {status_state} | Mode: {stats['mode_str']}"

        screen.blit(self.font.render(final_str, True,
                    self.c_yellow), (10, self.dh - 25))

    def _draw_controls(self, screen):
        lines = ["CONTROLS", "SPACE: Pause", "R: Reset Airflow", "C: Clear Obstacles",
                 "A: Airfoil Menu", "D: Sweep Data", "1-4: View Modes", "H: HUD"]
        w, h = 200, len(lines)*20 + 10
        x, y = self.dw - w - 10, 10

        s = pygame.Surface((w, h))
        s.set_alpha(180)
        s.fill(self.c_bg)
        screen.blit(s, (x, y))
        pygame.draw.rect(screen, (100, 100, 100), (x, y, w, h), 1)

        for i, t in enumerate(lines):
            c = self.c_yellow if i == 0 else self.c_text
            screen.blit(self.font.render(t, True, c), (x+10, y+5+i*20))

    def get_graph_rect(self, expansion):
        # Small: Bottom Right
        gw_s, gh_s = 300, 150
        rect_s = pygame.Rect(self.dw - gw_s - 10,
                             self.dh - gh_s - 10, gw_s, gh_s)

        # Large: Centered Square-ish
        m_y = 50
        m_x = 100

        rect_l = pygame.Rect(m_x, m_y, self.dw - 2*m_x, self.dh - 2*m_y)

        curr_x = rect_s.x + (rect_l.x - rect_s.x) * expansion
        curr_y = rect_s.y + (rect_l.y - rect_s.y) * expansion
        curr_w = rect_s.width + (rect_l.width - rect_s.width) * expansion
        curr_h = rect_s.height + (rect_l.height - rect_s.height) * expansion

        return pygame.Rect(int(curr_x), int(curr_y), int(curr_w), int(curr_h))

    def _draw_graph(self, screen, data, expansion, sweep_active):
        rect = self.get_graph_rect(expansion)
        gx, gy, gw, gh = rect.x, rect.y, rect.width, rect.height

        s = pygame.Surface((gw, gh))
        s.set_alpha(200 if expansion < 0.5 else 230)
        s.fill(self.c_bg)
        screen.blit(s, (gx, gy))
        pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        vals = [d[1] for d in data] + [d[2] for d in data]
        if not vals:
            vals = [0]

        min_y, max_y = min(0, min(vals)-2), max(vals)+2
        dy = max_y - min_y or 1

        min_a, max_a = data[0][0], data[-1][0]
        da = max_a - min_a or 1

        def pts(angle, val):
            px = gx + int(((angle - min_a) / da) * gw)
            py = gy + gh - int(((val - min_y) / dy) * gh)
            return (px, py)

        # Zero Line
        zy = gy + gh - int((0-min_y)/dy*gh)
        if gy <= zy <= gy+gh:
            pygame.draw.line(screen, (80, 80, 80), (gx, zy), (gx+gw, zy))

        # Plot Lines
        for i in range(len(data)):
            a, l, d = data[i]
            pl, pd = pts(a, l), pts(a, d)

            pygame.draw.circle(screen, self.c_green, pl,
                               3 if expansion > 0.5 else 2)
            pygame.draw.circle(screen, self.c_red, pd,
                               3 if expansion > 0.5 else 2)

            if i > 0:
                pa, pl0, pd0 = data[i-1]
                pygame.draw.line(screen, self.c_green, pts(pa, pl0), pl, 2)
                pygame.draw.line(screen, self.c_red, pts(pa, pd0), pd, 2)

        # Expanded Content
        if expansion > 0.8:
            # Axes
            for i in range(9):
                t = i / 8.0
                val = min_y + (dy * t)
                py = gy + gh - int((t) * gh)

                pygame.draw.line(screen, (150, 150, 150),
                                 (gx, py), (gx - 8, py))
                lbl = self.font.render(f"{val:.1f}", True, (180, 180, 180))
                screen.blit(lbl, (gx - lbl.get_width() -
                            12, py - lbl.get_height()//2))

            start_tick = int(np.ceil(min_a / 5.0)) * 5
            end_tick = int(np.floor(max_a / 5.0)) * 5

            curr_tick = start_tick
            while curr_tick <= end_tick:
                px = gx + int(((curr_tick - min_a) / da) * gw)
                pygame.draw.line(screen, (150, 150, 150),
                                 (px, gy + gh), (px, gy + gh + 8))
                lbl = self.font.render(f"{curr_tick}°", True, (180, 180, 180))
                screen.blit(lbl, (px - lbl.get_width()//2, gy + gh + 10))
                curr_tick += 5

            lbl_y = self.font.render("Force (N)", True, (200, 200, 200))
            lbl_y = pygame.transform.rotate(lbl_y, 90)
            screen.blit(lbl_y, (gx - 60, gy + gh//2 - lbl_y.get_height()//2))

            screen.blit(self.font.render("Angle of Attack (°)", True, (200, 200, 200)),
                        (gx + gw//2 - 60, gy + gh + 35))

            # Stats Box
            max_l = max(data, key=lambda x: x[1])
            max_d = max(data, key=lambda x: x[2])
            min_d = min(data, key=lambda x: x[2])
            best_ld = max(
                data, key=lambda x: x[1]/(x[2] if abs(x[2]) > 0.001 else 0.001))

            stats_txt = [
                f"Highest Lift: {max_l[1]:.2f}N @ {max_l[0]}°",
                f"Highest Drag: {max_d[2]:.2f}N @ {max_d[0]}°",
                f"Lowest Drag:  {min_d[2]:.2f}N @ {min_d[0]}°",
                f"Best L/D Ratio: {(best_ld[1]/best_ld[2]):.2f} @ {best_ld[0]}°"
            ]

            screen.blit(self.btn_font.render(
                "LIFT (Green) vs DRAG (Red) POLAR", True, (255, 255, 255)), (gx+20, gy+20))

            sy = gy + 60
            for line in stats_txt:
                screen.blit(self.font.render(
                    line, True, (200, 200, 200)), (gx+30, sy))
                sy += 25

        # Small Mode Content
        elif not sweep_active and len(data) > 0:
            txt = self.btn_font.render("CLICK TO VIEW", True, self.c_yellow)
            screen.blit(txt, (gx + gw//2 - txt.get_width()//2, gy + 10))

    def _draw_menu(self, screen, user_text):
        s = pygame.Surface((self.box_rect.width, self.box_rect.height))
        s.set_alpha(240)
        s.fill((30, 30, 30))
        screen.blit(s, self.box_rect)
        pygame.draw.rect(screen, (200, 200, 200), self.box_rect, 2)

        screen.blit(self.font.render("NACA Code:", True, self.c_text),
                    (self.box_rect.x+20, self.box_rect.y+20))
        screen.blit(self.font.render(user_text + "_", True,
                    self.c_yellow), (self.box_rect.x+20, self.box_rect.y+50))

        mx, my = pygame.mouse.get_pos()
        c1 = (60, 60, 60) if self.btn_gen.collidepoint(
            mx, my) else (40, 40, 40)
        c2 = (60, 60, 60) if self.btn_swp.collidepoint(
            mx, my) else (40, 40, 40)

        pygame.draw.rect(screen, c1, self.btn_gen)
        pygame.draw.rect(screen, (200, 200, 200), self.btn_gen, 1)
        screen.blit(self.btn_font.render("GENERATE", True,
                    (255, 255, 255)), (self.btn_gen.x+45, self.btn_gen.y+10))

        pygame.draw.rect(screen, c2, self.btn_swp)
        pygame.draw.rect(screen, (200, 200, 200), self.btn_swp, 1)
        screen.blit(self.btn_font.render("SWEEP", True, self.c_green),
                    (self.btn_swp.x+60, self.btn_swp.y+10))

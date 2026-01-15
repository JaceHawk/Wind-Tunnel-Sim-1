import pygame
import numpy as np
import taichi as ti
from FluidTaichi import FluidTaichi
from ParticlesTaichi import ParticlesTaichi
from AirfoilGenerator import stamp_airfoil
from Hud import HUD

# Initialize GPU
try:
    ti.init(arch=ti.cuda)
except:
    ti.init(arch=ti.vulkan)

# Configuration
WIDTH, HEIGHT = 600, 250
CELL_SIZE = 2
TARGET_FPS = 60
STEPS_PER_FRAME = 4
DISPLAY_W, DISPLAY_H = WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE

# Physics Constants
TUNNEL_HEIGHT_M = 1.25
REAL_AIR_SPEED = 30.0
AIR_DENSITY = 1.225
LATTICE_SPEED = 0.1
MAX_LATTICE_SPEED = 0.577

# Derived Math
dx = TUNNEL_HEIGHT_M / HEIGHT
dt = (LATTICE_SPEED * dx) / REAL_AIR_SPEED
FORCE_SCALE = AIR_DENSITY * (dx**3) / (dt**2)
MARGIN_X, MARGIN_Y = WIDTH // 5, HEIGHT // 3

# Data Sweep
SWEEP_TIME_FIRST = 150 * TARGET_FPS * STEPS_PER_FRAME
SWEEP_ANGLES = list(range(-5, 16, 1))
CONVERGENCE_TIME_MS = 180000

# Setup
pygame.init()
screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
clock = pygame.time.Clock()

fluid = FluidTaichi(WIDTH, HEIGHT, viscosity=0.015)
particles = ParticlesTaichi(200000, WIDTH, HEIGHT, CELL_SIZE)
hud = HUD(DISPLAY_W, DISPLAY_H, WIDTH, HEIGHT, CELL_SIZE)

fluid.init_flow()

# Memory Pre-allocation
fluid_arr = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
fluid_surf = pygame.Surface((WIDTH, HEIGHT))
part_arr = np.zeros((DISPLAY_W, DISPLAY_H, 3), dtype=np.uint8)
part_surf = pygame.Surface((DISPLAY_W, DISPLAY_H))

# State
view_mode = 2
show_hud = True
paused = False
input_active = False
user_text = ""
current_naca = "0012"
sim_start_tick = pygame.time.get_ticks()
current_airfoil_name = "None"
sweep_buffer = []

# Graph Animation State
graph_expansion = 0.0
graph_target_state = 0.0
GRAPH_ANIM_SPEED = 1.5

# Performance Tracking
app_start_time = pygame.time.get_ticks()
total_frames = 0

# Smoothing
smooth_drag, smooth_lift = 0.0, 0.0
SMOOTHING_ALPHA = 0.005
current_lb_speed, target_lb_speed = 0.0, LATTICE_SPEED
spool_rate = 0.001

# Sweep State
sweep_active = False
sweep_index = 0
sweep_timer = 0
sweep_data = []
sweep_buffer = []


def reset_simulation(hard=False):
    global current_lb_speed, smooth_drag, smooth_lift, sim_start_tick
    fluid.init_flow()
    if hard:
        fluid.cylinder.fill(0)
    current_lb_speed = 0.0
    smooth_drag, smooth_lift = 0.0, 0.0
    sim_start_tick = pygame.time.get_ticks()


def action_generate(text):
    global input_active, user_text, current_naca, current_airfoil_name, sim_start_tick
    try:
        parts = text.split()
        code = parts[0]
        angle = float(parts[1]) if len(parts) > 1 else 0.0
        current_naca = code
        current_airfoil_name = f"NACA {code}"

        temp_cyl = np.zeros((WIDTH, HEIGHT), dtype=bool)
        stamp_airfoil(temp_cyl, code, WIDTH//2, HEIGHT//2, WIDTH//3, angle)
        fluid.cylinder.from_numpy(temp_cyl.astype(int))

        sim_start_tick = pygame.time.get_ticks()
        print(f"Generated {code} at {angle}°")
    except:
        pass
    input_active = False
    user_text = ""


def action_sweep(text):
    global input_active, user_text, current_naca, sweep_active, sweep_index, sweep_timer, sweep_data, sweep_buffer, current_airfoil_name
    try:
        code = text.split()[0]
        current_naca = code
        current_airfoil_name = f"Sweep {code}"
        reset_simulation(hard=False)
        sweep_active, sweep_index, sweep_timer = True, 0, 0
        sweep_data, sweep_buffer = [], []

        temp_cyl = np.zeros((WIDTH, HEIGHT), dtype=bool)
        stamp_airfoil(temp_cyl, current_naca, WIDTH//2,
                      HEIGHT//2, WIDTH//3, SWEEP_ANGLES[0])
        fluid.cylinder.from_numpy(temp_cyl.astype(int))
    except:
        pass
    input_active = False
    user_text = ""


running = True
while running:
    # Event Loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Menu Input
        if input_active:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    action_generate(user_text)
                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]
                elif event.key == pygame.K_ESCAPE:
                    input_active = False
                else:
                    user_text += event.unicode

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if hud.btn_gen.collidepoint(mx, my):
                    action_generate(user_text)
                if hud.btn_swp.collidepoint(mx, my):
                    action_sweep(user_text)

        # Simulation Controls
        else:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    view_mode = 0
                if event.key == pygame.K_2:
                    view_mode = 1
                if event.key == pygame.K_3:
                    view_mode = 2
                if event.key == pygame.K_4:
                    view_mode = 3

                if event.key == pygame.K_h:
                    show_hud = not show_hud
                if event.key == pygame.K_a:
                    input_active = True
                    user_text = ""
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_c:
                    reset_simulation(hard=True)
                    current_airfoil_name = "None"
                if event.key == pygame.K_r:
                    reset_simulation(hard=False)
                if event.key == pygame.K_d:
                    action_sweep(current_naca)
                if event.key == pygame.K_x:
                    if sweep_active:
                        sweep_active = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                curr_rect = hud.get_graph_rect(graph_expansion)
                if curr_rect.collidepoint(mx, my) and len(sweep_data) > 0:
                    graph_target_state = 1.0 if graph_target_state == 0.0 else 0.0

    # Physics
    current_peak_speed = 0.0
    if not paused:
        for _ in range(STEPS_PER_FRAME):
            if current_lb_speed < target_lb_speed:
                current_lb_speed += spool_rate

            fluid.set_inlet(current_lb_speed)
            d, l, spd = fluid.step()
            current_peak_speed = spd

            smooth_drag = (d * SMOOTHING_ALPHA) + \
                (smooth_drag * (1-SMOOTHING_ALPHA))
            smooth_lift = (l * SMOOTHING_ALPHA) + \
                (smooth_lift * (1-SMOOTHING_ALPHA))

        if sweep_active:
            sweep_timer += STEPS_PER_FRAME
            target = SWEEP_TIME_FIRST

            if sweep_timer > target // 2:
                sweep_buffer.append(
                    (-smooth_lift*FORCE_SCALE, smooth_drag*FORCE_SCALE))

            if sweep_timer > target:
                if sweep_buffer:
                    avg_l = sum(d[0] for d in sweep_buffer)/len(sweep_buffer)
                    avg_d = sum(d[1] for d in sweep_buffer)/len(sweep_buffer)
                else:
                    avg_l, avg_d = -smooth_lift*FORCE_SCALE, smooth_drag*FORCE_SCALE

                sweep_data.append((SWEEP_ANGLES[sweep_index], avg_l, avg_d))
                print(
                    f"Recorded {SWEEP_ANGLES[sweep_index]}°: L={avg_l:.2f} D={avg_d:.2f}")

                sweep_index += 1
                sweep_timer = 0
                sweep_buffer = []

                if sweep_index >= len(SWEEP_ANGLES):
                    sweep_active = False
                    print("--- Sweep Complete ---")
                else:
                    reset_simulation(hard=False)
                    temp_cyl = np.zeros((WIDTH, HEIGHT), dtype=bool)
                    stamp_airfoil(temp_cyl, current_naca, WIDTH//2,
                                  HEIGHT//2, WIDTH//3, SWEEP_ANGLES[sweep_index])
                    fluid.cylinder.from_numpy(temp_cyl.astype(int))

    # Render
    ti.sync()

    if graph_expansion < graph_target_state:
        graph_expansion += GRAPH_ANIM_SPEED * (1.0 / TARGET_FPS)
        if graph_expansion > graph_target_state:
            graph_expansion = graph_target_state
    elif graph_expansion > graph_target_state:
        graph_expansion -= GRAPH_ANIM_SPEED * (1.0 / TARGET_FPS)
        if graph_expansion < graph_target_state:
            graph_expansion = graph_target_state

    if view_mode == 2:
        if not paused:
            particles.update(fluid.u, fluid.cylinder)
        particles.render(fluid.u, fluid.cylinder, 0.1)
        particles.export_visuals(part_arr)
        pygame.surfarray.blit_array(part_surf, part_arr)
        screen.blit(part_surf, (0, 0))

    else:
        fluid.render_visuals(view_mode)
        fluid.export_visuals(fluid_arr)
        pygame.surfarray.blit_array(fluid_surf, fluid_arr)
        scaled_surf = pygame.transform.scale(
            fluid_surf, (DISPLAY_W, DISPLAY_H))
        screen.blit(scaled_surf, (0, 0))

    # Stats & HUD
    total_frames += 1
    current_time = pygame.time.get_ticks()
    session_duration = (current_time - app_start_time) / 1000.0
    if session_duration > 0:
        avg_fps = total_frames / session_duration
    else:
        avg_fps = 0.0

    mode_names = {0: "Curl", 1: "Speed", 2: "Particles", 3: "Pressure"}
    mode_str = mode_names.get(view_mode, "Unknown")

    sim_time_ratio = TARGET_FPS * dt
    time_scale_str = f"1/{int(round((1/sim_time_ratio)/10.0)*10/STEPS_PER_FRAME)} Speed" if sim_time_ratio > 0 else "--"

    swp_rem_angle = 0
    swp_rem_total = 0
    if sweep_active:
        steps_left = (SWEEP_TIME_FIRST - sweep_timer)
        swp_rem_angle = steps_left / (TARGET_FPS * STEPS_PER_FRAME)
        angles_left = len(SWEEP_ANGLES) - sweep_index - 1
        time_per_angle = SWEEP_TIME_FIRST / (TARGET_FPS * STEPS_PER_FRAME)
        swp_rem_total = swp_rem_angle + (angles_left * time_per_angle)

    stats = {
        'show_hud': show_hud,
        'paused': paused,
        'start_tick': sim_start_tick,
        'name': current_airfoil_name,
        'max_speed': MAX_LATTICE_SPEED,
        'conv_time': CONVERGENCE_TIME_MS,
        'peak_speed': current_peak_speed,
        'drag': smooth_drag * FORCE_SCALE,
        'lift': -smooth_lift * FORCE_SCALE,
        'wind': (current_lb_speed / LATTICE_SPEED) * REAL_AIR_SPEED,
        'fps': int(clock.get_fps()),
        'time_scale': time_scale_str,
        'sweep_data': sweep_data,
        'input_active': input_active,
        'user_text': user_text,
        'margin_x': MARGIN_X,
        'margin_y': MARGIN_Y,
        'sweep_active': sweep_active,
        'graph_expansion': graph_expansion,
        'swp_rem_angle': swp_rem_angle,
        'avg_fps': int(avg_fps),
        'mode_str': mode_str,
        'swp_rem_total': swp_rem_total
    }
    hud.render(screen, fluid, stats)

    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()

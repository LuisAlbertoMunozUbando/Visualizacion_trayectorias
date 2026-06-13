#!/usr/bin/env python
# coding: utf-8
"""
Mario Kart Obstacle Avoidance — Live Video Tracking (script puro, sin Jupyter)

Modos de visualización:
  MODE 1: Paths de tracking visibles + sprite navegando esquivando obstáculos
  MODE 2: Igual que Modo 1 pero sin guardar los paths (solo obstáculos en tiempo real)
  MODE 3: Paths visibles + el sprite NO puede atravesar los paths (son muros)

Coloca MC.png en el mismo directorio antes de ejecutar.
Requiere: opencv-python, numpy, torch, yt-dlp, ffmpeg en PATH.
Se abre una ventana con cv2.imshow; presiona 'q' para salir.
"""

import subprocess, numpy as np, cv2, time, math, os
from collections import deque
import torch, torch.nn.functional as F

# =====================================================================
#  SELECCIONA EL MODO ANTES DE EJECUTAR
# =====================================================================
MODE = 2   # <-- CAMBIA AQUI: 1, 2 o 3

descriptions = {
    1: "Paths de tracking + sprite esquivando obstaculos",
    2: "Sin paths (tiempo real) + sprite esquivando obstaculos",
    3: "Paths de tracking son MUROS — el sprite no puede atravesarlos",
}

# --------------------- Parameters ---------------------
WIDTH, HEIGHT = 1280, 720

YOUTUBE_URL = "https://youtube.com/shorts/cGVcse5bUGo?si=mE7A9ZYFzD_zLDUT"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_AREA        = 800
MAX_TRACK_DIST  = 60
MINI_SCALE      = 0.25
MAX_TRAJ        = 200
N_FRAMES        = 2000
ALPHA_BG        = 0.02
EDGE_THRESH     = 0.2
MOTION_THRESH   = 0.15
SPRITE_SCALE    = 1.0 / 20.0

# Navegacion
AVOIDANCE_RADIUS    = 120
AVOIDANCE_STRENGTH  = 8.0
HOME_STRENGTH       = 0.5
WANDER_STRENGTH     = 2.0
DAMPING             = 0.92
MAX_VELOCITY        = 12.0

# Modo 3: paths como muros
PATH_WALL_RADIUS    = 18     # grosor de la "pared" de cada punto del path
PATH_WALL_STRENGTH  = 14.0   # fuerza de repulsion del muro
PATH_SAMPLE_STEP    = 3      # cada cuantos puntos del path se evaluan (performance)


def main():
    print(f"Modo seleccionado: {MODE}")
    print(f"  -> {descriptions[MODE]}")
    print("PyTorch device:", DEVICE)

    # --------------------- Load sprite ---------------------
    sprite_paths = ["MC.png", "./MC.png", os.path.expanduser("~/MC.png")]
    sprite_raw = None
    for sp in sprite_paths:
        if os.path.exists(sp):
            sprite_raw = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
            print(f"Imagen cargada desde: {sp}")
            break
    if sprite_raw is None:
        print("AVISO: MC.png no encontrada, usando placeholder.")
        sz = int(min(WIDTH, HEIGHT) * SPRITE_SCALE)
        sprite_raw = np.zeros((sz, sz, 4), dtype=np.uint8)
        cv2.circle(sprite_raw, (sz // 2, sz // 2), sz // 2 - 2, (0, 0, 255, 255), -1)

    sprite_target_h = int(HEIGHT * SPRITE_SCALE)
    sprite_target_w = int(sprite_raw.shape[1] * sprite_target_h / sprite_raw.shape[0])
    sprite_resized = cv2.resize(sprite_raw, (sprite_target_w, sprite_target_h),
                                 interpolation=cv2.INTER_AREA)
    if sprite_resized.shape[2] == 4:
        sprite_bgr   = sprite_resized[:, :, :3]
        sprite_alpha = sprite_resized[:, :, 3].astype(np.float32) / 255.0
    else:
        sprite_bgr   = sprite_resized
        sprite_alpha = np.ones(sprite_resized.shape[:2], dtype=np.float32)

    print(f"Sprite: {sprite_bgr.shape[1]}x{sprite_bgr.shape[0]} px "
          f"(1/{int(1/SPRITE_SCALE)} del escenario)")

    # --------------------- Sprite state ---------------------
    state = {
        "x": float(WIDTH // 2),
        "y": float(HEIGHT // 2),
        "vx": 0.0,
        "vy": 0.0,
        "wander": np.random.uniform(0, 2 * math.pi),
    }
    home_x = float(WIDTH // 2)
    home_y = float(HEIGHT // 2)

    def overlay_sprite(canvas, sx, sy, bgr, alpha):
        sh, sw = bgr.shape[:2]
        x1, y1 = int(sx - sw // 2), int(sy - sh // 2)
        x2, y2 = x1 + sw, y1 + sh
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
        if cx1 >= cx2 or cy1 >= cy2:
            return
        sx1, sy1 = cx1 - x1, cy1 - y1
        sx2, sy2 = sx1 + (cx2 - cx1), sy1 + (cy2 - cy1)
        roi = canvas[cy1:cy2, cx1:cx2]
        a   = alpha[sy1:sy2, sx1:sx2, np.newaxis]
        canvas[cy1:cy2, cx1:cx2] = (bgr[sy1:sy2, sx1:sx2] * a + roi * (1 - a)).astype(np.uint8)

    def update_sprite(obstacles, all_path_points=None):
        """
        Steering behaviors:
          1. Repulsion de obstaculos en vivo
          2. Atraccion al centro
          3. Wander aleatorio
          4. Repulsion de bordes
          5. (Modo 3) Repulsion de puntos de path ya dibujados
        """
        fx, fy = 0.0, 0.0

        # 1. Repulsion de obstaculos en vivo
        for (ox, oy) in obstacles:
            dx = state["x"] - ox
            dy = state["y"] - oy
            dist = math.hypot(dx, dy) + 1e-6
            if dist < AVOIDANCE_RADIUS:
                s = AVOIDANCE_STRENGTH * (1.0 - dist / AVOIDANCE_RADIUS) ** 2
                fx += (dx / dist) * s
                fy += (dy / dist) * s

        # 2. Atraccion al centro
        dx_h = home_x - state["x"]
        dy_h = home_y - state["y"]
        d_h  = math.hypot(dx_h, dy_h) + 1e-6
        fx += (dx_h / d_h) * HOME_STRENGTH
        fy += (dy_h / d_h) * HOME_STRENGTH

        # 3. Wander
        state["wander"] += np.random.uniform(-0.5, 0.5)
        fx += math.cos(state["wander"]) * WANDER_STRENGTH
        fy += math.sin(state["wander"]) * WANDER_STRENGTH

        # 4. Bordes
        margin = 60
        if state["x"] < margin:           fx += 3.0 * (margin - state["x"]) / margin
        if state["x"] > WIDTH - margin:   fx -= 3.0 * (state["x"] - (WIDTH - margin)) / margin
        if state["y"] < margin:           fy += 3.0 * (margin - state["y"]) / margin
        if state["y"] > HEIGHT - margin:  fy -= 3.0 * (state["y"] - (HEIGHT - margin)) / margin

        # 5. (Modo 3) Paths como muros
        if all_path_points is not None and len(all_path_points) > 0:
            for idx in range(0, len(all_path_points), PATH_SAMPLE_STEP):
                px, py = all_path_points[idx]
                dx = state["x"] - px
                dy = state["y"] - py
                dist = math.hypot(dx, dy) + 1e-6
                if dist < PATH_WALL_RADIUS:
                    s = PATH_WALL_STRENGTH * (1.0 - dist / PATH_WALL_RADIUS) ** 2
                    fx += (dx / dist) * s
                    fy += (dy / dist) * s

        # Aplicar fuerzas
        state["vx"] = state["vx"] * DAMPING + fx
        state["vy"] = state["vy"] * DAMPING + fy
        speed = math.hypot(state["vx"], state["vy"])
        if speed > MAX_VELOCITY:
            state["vx"] = (state["vx"] / speed) * MAX_VELOCITY
            state["vy"] = (state["vy"] / speed) * MAX_VELOCITY

        # Posicion candidata
        new_x = state["x"] + state["vx"]
        new_y = state["y"] + state["vy"]

        # (Modo 3) Hard collision: si la posicion nueva esta dentro de un muro,
        #           revertir y deslizar
        if all_path_points is not None and len(all_path_points) > 0:
            blocked = False
            for idx in range(0, len(all_path_points), PATH_SAMPLE_STEP):
                px, py = all_path_points[idx]
                if math.hypot(new_x - px, new_y - py) < PATH_WALL_RADIUS * 0.5:
                    blocked = True
                    break
            if blocked:
                # Intentar solo movimiento horizontal
                test_x = state["x"] + state["vx"]
                ok_x = True
                for idx in range(0, len(all_path_points), PATH_SAMPLE_STEP):
                    px, py = all_path_points[idx]
                    if math.hypot(test_x - px, state["y"] - py) < PATH_WALL_RADIUS * 0.5:
                        ok_x = False
                        break
                # Intentar solo movimiento vertical
                test_y = state["y"] + state["vy"]
                ok_y = True
                for idx in range(0, len(all_path_points), PATH_SAMPLE_STEP):
                    px, py = all_path_points[idx]
                    if math.hypot(state["x"] - px, test_y - py) < PATH_WALL_RADIUS * 0.5:
                        ok_y = False
                        break
                if ok_x:
                    new_x = test_x
                    new_y = state["y"]
                    state["vy"] *= -0.5
                elif ok_y:
                    new_x = state["x"]
                    new_y = test_y
                    state["vx"] *= -0.5
                else:
                    new_x = state["x"]
                    new_y = state["y"]
                    state["vx"] *= -0.7
                    state["vy"] *= -0.7

        state["x"] = max(-30, min(WIDTH + 30, new_x))
        state["y"] = max(-30, min(HEIGHT + 30, new_y))

    # --------------------- Stream setup ---------------------
    ytdlp_cmd  = ["yt-dlp", "-f", "b", "-o", "-", YOUTUBE_URL]
    ffmpeg_cmd = ["ffmpeg", "-hwaccel", "cuda", "-i", "pipe:0",
                  "-vf", f"scale={WIDTH}:{HEIGHT}",
                  "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    ytdlp_proc  = subprocess.Popen(ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=ytdlp_proc.stdout,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    bufsize=10 ** 8)
    frame_size = WIDTH * HEIGHT * 3

    def read_frame():
        raw = ffmpeg_proc.stdout.read(frame_size)
        return np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3)) if len(raw) == frame_size else None

    # --------------------- Tracking state ---------------------
    next_id      = 0
    tracks       = {}
    trajectories = {}
    colors_map   = {}
    sprite_trail = deque(maxlen=400)

    def random_color():
        return tuple(np.random.randint(80, 240, 3).tolist())

    def euclidean(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # --------------------- GPU kernels ---------------------
    sobel_x  = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=DEVICE).float().view(1, 1, 3, 3)
    sobel_y  = sobel_x.transpose(2, 3)
    bg_model = None

    # --------------------- Main loop ---------------------
    mode_names = {
        1: "MODO 1: Paths + sprite esquiva obstaculos",
        2: "MODO 2: Sin paths + sprite esquiva obstaculos",
        3: "MODO 3: Paths son MUROS — sprite no los atraviesa",
    }
    print(f"\nIniciando {mode_names[MODE]}...")
    print("=" * 60)
    print("Presiona 'q' en la ventana de video para salir.")

    cv2.namedWindow("Mario Kart Obstacle Avoidance", cv2.WINDOW_NORMAL)

    it = 0
    try:
        for it in range(N_FRAMES):

            frame = read_frame()
            if frame is None:
                break

            # ---- GPU processing ----
            frame_t = torch.from_numpy(frame).to(DEVICE).float() / 255.0
            gray = frame_t.mean(dim=2, keepdim=True).permute(2, 0, 1).unsqueeze(0)

            if bg_model is None:
                bg_model = gray.clone()
            else:
                bg_model = (1 - ALPHA_BG) * bg_model + ALPHA_BG * gray

            motion = torch.abs(gray - bg_model)
            gx = F.conv2d(gray, sobel_x, padding=1)
            gy = F.conv2d(gray, sobel_y, padding=1)
            edges = torch.sqrt(gx ** 2 + gy ** 2)

            mask = ((motion > MOTION_THRESH) | (edges > EDGE_THRESH)).squeeze().float()
            mask_cpu = (mask.detach().cpu().numpy() * 255).astype(np.uint8)

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_cpu = cv2.morphologyEx(mask_cpu, cv2.MORPH_OPEN, k)
            mask_cpu = cv2.morphologyEx(mask_cpu, cv2.MORPH_DILATE, k)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cpu)

            # ---- Obstacles & tracking ----
            obstacle_centers = []
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                    continue
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                obstacle_centers.append((cx, cy))

                assigned, best = None, MAX_TRACK_DIST
                for tid, (tx, ty) in tracks.items():
                    d = euclidean((cx, cy), (tx, ty))
                    if d < best:
                        best, assigned = d, tid
                if assigned is None:
                    assigned = next_id; next_id += 1
                    tracks[assigned] = (cx, cy)
                    trajectories[assigned] = deque(maxlen=MAX_TRAJ)
                    colors_map[assigned] = random_color()
                tracks[assigned] = (cx, cy)
                trajectories[assigned].append((cx, cy))

            # ---- Collect all path points for Mode 3 ----
            all_path_pts = None
            if MODE == 3:
                all_path_pts = []
                for pts in trajectories.values():
                    all_path_pts.extend(pts)

            # ---- Update sprite ----
            update_sprite(obstacle_centers, all_path_pts)
            sprite_trail.append((int(state["x"]), int(state["y"])))

            # ================================================================
            #  RENDER
            # ================================================================

            if MODE == 1:
                # --- MODO 1: canvas blanco, paths visibles, sprite esquiva obstaculos ---
                canvas = np.ones_like(frame) * 255

                # Paths de objetos
                for tid, pts in trajectories.items():
                    col = colors_map[tid]
                    for j in range(1, len(pts)):
                        cv2.line(canvas, pts[j - 1], pts[j], col, 2)
                    x, y = tracks[tid]
                    cv2.circle(canvas, (x, y), 6, col, -1)
                    cv2.putText(canvas, str(tid), (x + 8, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                # Radios de avoidance
                for (ox, oy) in obstacle_centers:
                    cv2.circle(canvas, (ox, oy), AVOIDANCE_RADIUS, (210, 210, 210), 1)

                # Trail del sprite (verde degradado)
                for j in range(1, len(sprite_trail)):
                    a = j / len(sprite_trail)
                    cv2.line(canvas, sprite_trail[j - 1], sprite_trail[j],
                             (0, int(100 + 155 * a), 0), max(1, int(3 * a)))

                cv2.putText(canvas, f"MODO 1: Tracking + Navegacion | Obj: {len(tracks)} | F: {it}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                mini = cv2.resize(frame, None, fx=MINI_SCALE, fy=MINI_SCALE)
                h, w = mini.shape[:2]
                canvas[10:10 + h, WIDTH - w - 10:WIDTH - 10] = mini

            elif MODE == 2:
                # --- MODO 2: canvas blanco, SIN paths, solo obstaculos en tiempo real ---
                canvas = np.ones_like(frame) * 255

                # Obstaculos actuales como puntos naranjas
                for (ox, oy) in obstacle_centers:
                    cv2.circle(canvas, (ox, oy), 10, (0, 140, 255), -1)
                    cv2.circle(canvas, (ox, oy), AVOIDANCE_RADIUS, (230, 230, 230), 1)

                # Trail corto del sprite (ultimas 30 posiciones, se desvanece)
                recent = list(sprite_trail)[-30:]
                for j in range(1, len(recent)):
                    a = j / len(recent)
                    cv2.line(canvas, recent[j - 1], recent[j],
                             (0, int(80 + 175 * a), 0), max(1, int(2 * a)))

                cv2.putText(canvas, f"MODO 2: Sin paths + Navegacion | Obj: {len(obstacle_centers)} | F: {it}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                mini = cv2.resize(frame, None, fx=MINI_SCALE, fy=MINI_SCALE)
                h, w = mini.shape[:2]
                canvas[10:10 + h, WIDTH - w - 10:WIDTH - 10] = mini

            else:
                # --- MODO 3: canvas blanco, paths son MUROS, sprite no los atraviesa ---
                canvas = np.ones_like(frame) * 255

                # Paths como muros gruesos (rojo oscuro)
                for tid, pts in trajectories.items():
                    col = colors_map[tid]
                    lpts = list(pts)
                    for j in range(1, len(lpts)):
                        # Muro exterior (rojo oscuro grueso)
                        cv2.line(canvas, lpts[j - 1], lpts[j], (40, 40, 140), PATH_WALL_RADIUS)
                    for j in range(1, len(lpts)):
                        # Linea interior coloreada
                        cv2.line(canvas, lpts[j - 1], lpts[j], col, 2)
                    x, y = tracks[tid]
                    cv2.circle(canvas, (x, y), 6, col, -1)
                    cv2.putText(canvas, str(tid), (x + 8, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                # Trail del sprite (verde)
                for j in range(1, len(sprite_trail)):
                    a = j / len(sprite_trail)
                    cv2.line(canvas, sprite_trail[j - 1], sprite_trail[j],
                             (0, int(100 + 155 * a), 0), max(1, int(3 * a)))

                cv2.putText(canvas, f"MODO 3: Paths = MUROS | Obj: {len(tracks)} | F: {it}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                mini = cv2.resize(frame, None, fx=MINI_SCALE, fy=MINI_SCALE)
                h, w = mini.shape[:2]
                canvas[10:10 + h, WIDTH - w - 10:WIDTH - 10] = mini

            # ---- Sprite overlay (todos los modos) ----
            overlay_sprite(canvas, int(state["x"]), int(state["y"]), sprite_bgr, sprite_alpha)

            cv2.imshow("Mario Kart Obstacle Avoidance", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --------------------- Cleanup ---------------------
        try:
            ffmpeg_proc.terminate(); ytdlp_proc.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()

        print(f"\nFinalizado. {it + 1} frames en {mode_names[MODE]}.")
        print(f"Objetos rastreados: {len(tracks)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full-screen Pomodoro timer with animated backgrounds."""

import argparse
import math
import random
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import pygame
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This program requires pygame. Install it with `pip install pygame`."
    ) from exc


FrameDims = Tuple[int, int]

BACKGROUND_DIR = Path(__file__).resolve().parent / "backgrounds"
IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tga",
    ".webp",
    ".ppm",
    ".pgm",
    ".pbm",
)

@dataclass
class ModeDefinition:
    description: str
    initializer: Callable[[int, int], dict]
    renderer: Callable[[pygame.Surface, dict, float, FrameDims], dict]


def parse_pattern_pair(text: str) -> Tuple[int, int]:
    try:
        focus_raw, break_raw = text.split("/")
    except ValueError as exc:
        raise ValueError("Use the format <focus>/<break>, for example 50/10.") from exc
    try:
        focus = int(focus_raw)
        rest = int(break_raw)
    except ValueError as exc:
        raise ValueError("Focus and break values must be integers.") from exc
    if focus <= 0:
        raise ValueError("Focus minutes must be greater than zero.")
    if rest < 0:
        raise ValueError("Break minutes cannot be negative.")
    return focus * 60, rest * 60


def format_timer(seconds: float) -> str:
    total = max(0, int(math.ceil(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def build_fonts(width: int, height: int) -> Dict[str, pygame.font.Font]:
    base = min(width, height)
    timer_size = max(72, int(base * 0.2))
    label_size = max(36, int(base * 0.08))
    info_size = max(24, int(base * 0.045))
    small_size = max(18, int(base * 0.03))
    return {
        "timer": pygame.font.Font(None, timer_size),
        "label": pygame.font.Font(None, label_size),
        "info": pygame.font.Font(None, info_size),
        "small": pygame.font.Font(None, small_size),
    }


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: Tuple[int, int, int],
    center: Tuple[int, int],
) -> None:
    if not text:
        return
    text_surface = font.render(text, True, color)
    shadow_surface = font.render(text, True, (0, 0, 0))
    rect = text_surface.get_rect(center=center)
    shadow_rect = shadow_surface.get_rect(center=(center[0] + 2, center[1] + 2))
    surface.blit(shadow_surface, shadow_rect)
    surface.blit(text_surface, rect)


def draw_timer_overlay(
    surface: pygame.Surface,
    dims: FrameDims,
    fonts: Dict[str, pygame.font.Font],
    remaining: float,
) -> None:
    width, height = dims
    timer_text = format_timer(remaining)
    font = fonts["timer"]

    text_surface = font.render(timer_text, True, (255, 255, 255))
    shadow_surface = font.render(timer_text, True, (0, 0, 0))

    mirrored_text = pygame.transform.flip(text_surface, True, False)
    mirrored_shadow = pygame.transform.flip(shadow_surface, True, False)

    text_rect = mirrored_text.get_rect(center=(width // 2, height // 2))
    shadow_rect = mirrored_shadow.get_rect(center=(width // 2 + 3, height // 2 + 3))

    surface.blit(mirrored_shadow, shadow_rect)
    surface.blit(mirrored_text, text_rect)


def ensure_state(
    state: Optional[dict],
    dims: FrameDims,
    initializer: Callable[[int, int], dict],
) -> dict:
    width, height = dims
    if state is None or state.get("dims") != dims:
        state = initializer(width, height)
        state["dims"] = dims
    return state


def list_background_images() -> List[Path]:
    if not BACKGROUND_DIR.exists():
        return []
    files: List[Path] = []
    for path in BACKGROUND_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def find_background_image(name: str) -> Optional[Path]:
    if not name:
        return None
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    candidate = BACKGROUND_DIR / name
    if candidate.is_file():
        return candidate
    base = Path(name).stem if Path(name).suffix else name
    for path in list_background_images():
        if path.stem == base:
            return path
    return None


def make_image_mode(image_path: Path) -> ModeDefinition:
    def initializer(width: int, height: int) -> dict:
        image = pygame.image.load(str(image_path))
        if image.get_alpha():
            image = image.convert_alpha()
        else:
            image = image.convert()
        return {"image": image, "scaled_size": (0, 0), "scaled": None}

    def renderer(
        surface: pygame.Surface,
        state: dict,
        _: float,
        dims: FrameDims,
    ) -> dict:
        width, height = dims
        if state.get("scaled") is None or state.get("scaled_size") != dims:
            state["scaled"] = pygame.transform.smoothscale(state["image"], (width, height))
            state["scaled_size"] = dims
        surface.blit(state["scaled"], (0, 0))
        return state

    return ModeDefinition(
        description=f"Background image: {image_path.name}",
        initializer=initializer,
        renderer=renderer,
    )


def build_notification_sound() -> Optional[pygame.mixer.Sound]:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=1)
    except pygame.error:
        return None

    sample_rate = 22050
    duration = 0.4
    frequency = 880.0
    samples = int(sample_rate * duration)
    waveform = array("h")
    volume = 6000
    for i in range(samples):
        sample = int(math.sin(2 * math.pi * frequency * (i / sample_rate)) * volume)
        waveform.append(sample)
    try:
        return pygame.mixer.Sound(buffer=waveform.tobytes())
    except pygame.error:
        return None


def play_notification(sound: Optional[pygame.mixer.Sound]) -> None:
    if sound is None:
        return
    try:
        sound.play()
    except pygame.error:
        pass


def init_matrix(width: int, height: int) -> dict:
    spacing = max(12, width // 160)
    columns = []
    for x in range(0, width + spacing, spacing):
        columns.append(
            {
                "x": x + random.randint(-2, 2),
                "y": random.uniform(-height, 0),
                "speed": random.uniform(height * 0.4, height * 1.1),
                "length": random.randint(8, 22),
            }
        )
    return {"columns": columns, "spacing": spacing}


def render_matrix(
    surface: pygame.Surface,
    state: dict,
    dt: float,
    dims: FrameDims,
) -> dict:
    width, height = dims
    segment_height = max(16, int(height * 0.02))
    surface.fill((2, 8, 12))
    spacing = state["spacing"]
    for column in state["columns"]:
        column["y"] += column["speed"] * dt
        if column["y"] - column["length"] * segment_height > height + segment_height:
            column["y"] = random.uniform(-height, 0)
            column["speed"] = random.uniform(height * 0.4, height * 1.1)
            column["length"] = random.randint(8, 24)
            column["x"] = (column["x"] + random.randint(-spacing, spacing)) % max(1, width)
        x = column["x"] % max(1, width)
        for idx in range(column["length"]):
            y = column["y"] - idx * segment_height
            if y < -segment_height:
                break
            if y > height:
                continue
            fade = 1 - (idx / column["length"])
            color = (
                int(20 + 120 * fade),
                int(150 + 90 * fade),
                int(30 + 50 * fade),
            )
            if idx == 0:
                color = (200, 255, 200)
            rect = pygame.Rect(x, int(y), spacing - 2, segment_height - 1)
            surface.fill(color, rect)
    return state


def init_fireflies(width: int, height: int) -> dict:
    count = max(24, (width * height) // 20000)
    particles = []
    for _ in range(count):
        particles.append(
            {
                "x": random.random() * width,
                "y": random.random() * height,
                "vx": random.uniform(-30, 30),
                "vy": random.uniform(-20, 20),
                "pulse": random.random() * math.tau,
                "size": random.uniform(2.0, 4.5),
            }
        )
    return {"particles": particles}


def render_fireflies(
    surface: pygame.Surface,
    state: dict,
    dt: float,
    dims: FrameDims,
) -> dict:
    width, height = dims
    surface.fill((3, 5, 24))
    for particle in state["particles"]:
        particle["x"] = (particle["x"] + particle["vx"] * dt) % width
        particle["y"] = (particle["y"] + particle["vy"] * dt) % height
        particle["pulse"] = (particle["pulse"] + dt) % math.tau
        glow_strength = (math.sin(particle["pulse"]) + 1) / 2
        radius = particle["size"] + glow_strength * 3
        color = (
            int(90 + 130 * glow_strength),
            int(170 + 70 * glow_strength),
            int(80 + 60 * glow_strength),
        )
        glow_radius = max(4, int(radius * 3))
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*color, 45), (glow_radius, glow_radius), glow_radius)
        surface.blit(
            glow_surface,
            (particle["x"] - glow_radius, particle["y"] - glow_radius),
            special_flags=pygame.BLEND_ADD,
        )
        pygame.draw.circle(
            surface,
            color,
            (int(particle["x"]), int(particle["y"])),
            max(1, int(radius)),
        )
    return state


def init_aurora(width: int, height: int) -> dict:
    stars = [
        (random.randrange(width), random.randrange(height), random.randint(40, 140))
        for _ in range(max(80, width // 6))
    ]
    bands = [random.random() * math.tau for _ in range(3)]
    return {"stars": stars, "bands": bands, "phase": random.random() * math.tau}


def render_aurora(
    surface: pygame.Surface,
    state: dict,
    dt: float,
    dims: FrameDims,
) -> dict:
    width, height = dims
    surface.fill((4, 6, 20))
    for y in range(0, height, 3):
        shade = int(20 + 40 * (y / max(1, height)))
        pygame.draw.line(surface, (shade, shade, shade + 25), (0, y), (width, y))
    for x, y, brightness in state["stars"]:
        color = (brightness // 2, brightness // 2, brightness)
        surface.fill(color, pygame.Rect(x, y, 2, 2))
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 0))
    phase = state["phase"]
    for idx, offset in enumerate(state["bands"]):
        amplitude = height * (0.08 + idx * 0.04)
        baseline = height * (0.35 + idx * 0.15)
        step = max(4, width // 180)
        points = []
        for x in range(-20, width + 20, step):
            angle = offset + phase + x * 0.004 * (idx + 1)
            y = baseline + math.sin(angle) * amplitude
            points.append((x, y))
        color = (
            int(40 + idx * 35),
            int(150 + idx * 25),
            200,
        )
        pygame.draw.polygon(overlay, (*color, 30), points + [(width, height), (0, height)])
        pygame.draw.aalines(surface, color, False, points)
    surface.blit(overlay, (0, 0))
    state["phase"] = (phase + dt * 0.5) % math.tau
    return state


MODES: Dict[str, ModeDefinition] = {
    "matrix": ModeDefinition(
        "Matrix inspired digital rain.",
        init_matrix,
        render_matrix,
    ),
    "fireflies": ModeDefinition(
        "Forest fireflies floating across the night.",
        init_fireflies,
        render_fireflies,
    ),
    "aurora": ModeDefinition(
        "Soothing aurora ribbons over a star field.",
        init_aurora,
        render_aurora,
    ),
}

DEFAULT_MODE = "matrix"


def list_modes() -> None:
    width = max(len(name) for name in MODES)
    for name, mode in MODES.items():
        print(f"{name.ljust(width)}  - {mode.description}")
    backgrounds = list_background_images()
    if backgrounds:
        print("\nBackground images (use the name to select one):")
        for path in backgrounds:
            print(f"  {path.stem}  ->  backgrounds/{path.name}")
    else:
        print("\nAdd PNG/JPG/BMP/PPM files to the 'backgrounds' folder to use them as modes.")


def show_end_card(
    screen: pygame.Surface,
    fonts: Dict[str, pygame.font.Font],
    message: str,
) -> None:
    width, height = screen.get_size()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))
    draw_text(screen, fonts["label"], message, (255, 255, 255), (width // 2, height // 2))
    draw_text(
        screen,
        fonts["small"],
        "Press ESC, Q, or click to close",
        (200, 200, 200),
        (width // 2, int(height * 0.6)),
    )
    pygame.display.flip()
    wait_until = time.perf_counter() + 3.0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q, pygame.K_RETURN):
                return
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.QUIT):
                return
        if time.perf_counter() >= wait_until:
            return
        time.sleep(0.05)


def run_pomodoro(
    focus_seconds: int,
    break_seconds: int,
    mode: ModeDefinition,
    cycles: int,
    windowed: bool,
) -> None:
    pygame.init()
    alert_sound = build_notification_sound()
    flags = pygame.RESIZABLE if windowed else pygame.FULLSCREEN
    size = (1280, 720) if windowed else (0, 0)
    screen = pygame.display.set_mode(size, flags)
    pygame.display.set_caption("Pomodoro Focus Timer")
    pygame.mouse.set_visible(windowed)

    clock = pygame.time.Clock()
    fonts = build_fonts(*screen.get_size())
    font_dims = screen.get_size()
    state: Optional[dict] = None
    phase = "focus"
    remaining = float(focus_seconds)
    completed_focus = 0
    aborted = False
    goal_reached = False

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                aborted = True
                running = False
                break
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                aborted = True
                running = False
                break
            if windowed and event.type == pygame.VIDEORESIZE:
                new_size = (max(400, event.w), max(300, event.h))
                screen = pygame.display.set_mode(new_size, pygame.RESIZABLE)
                fonts = build_fonts(*new_size)
                font_dims = new_size
                state = None
        if not running:
            break

        dims = screen.get_size()
        if dims != font_dims:
            fonts = build_fonts(*dims)
            font_dims = dims
            state = None
        state = ensure_state(state, dims, mode.initializer)

        remaining -= dt
        if remaining <= 0:
            if phase == "focus":
                completed_focus += 1
                play_notification(alert_sound)
                if cycles and completed_focus >= cycles:
                    goal_reached = True
                    break
                if break_seconds > 0:
                    phase = "break"
                    remaining = float(break_seconds)
                else:
                    remaining = float(focus_seconds)
            else:
                play_notification(alert_sound)
                phase = "focus"
                remaining = float(focus_seconds)

        state = mode.renderer(screen, state, dt, dims)
        draw_timer_overlay(
            screen,
            dims,
            fonts,
            remaining,
        )
        pygame.display.flip()

    pygame.mouse.set_visible(True)
    if aborted and not completed_focus:
        final_text = "Timer stopped."
    elif aborted:
        final_text = f"Stopped after {completed_focus} cycle(s)."
    elif goal_reached and cycles:
        final_text = f"Completed {completed_focus} of {cycles} cycle(s)."
    else:
        final_text = f"Finished {completed_focus} cycle(s)."

    show_end_card(screen, fonts, final_text)
    if pygame.mixer.get_init():
        pygame.mixer.quit()
    pygame.quit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fullscreen Pomodoro timer with animated backgrounds."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Start the Pomodoro timer.")
    run_parser.add_argument("pattern", help="Focus/break minutes, for example 50/10.")
    run_parser.add_argument(
        "style",
        nargs="?",
        default=DEFAULT_MODE,
        help=(
            "Animation mode or background image name "
            f"(default: {DEFAULT_MODE})."
        ),
    )
    run_parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of focus rounds to run (0 = endless).",
    )
    run_parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run in a window instead of fullscreen.",
    )

    subparsers.add_parser("modes", help="List available animation modes.")

    args = parser.parse_args()

    if args.command == "modes":
        list_modes()
        return 0

    if args.command == "run":
        if args.cycles < 0:
            run_parser.error("Cycles must be zero or greater.")
        try:
            focus_seconds, break_seconds = parse_pattern_pair(args.pattern)
        except ValueError as exc:
            run_parser.error(str(exc))
        style_name = args.style or DEFAULT_MODE
        mode = MODES.get(style_name)
        if mode is None:
            image_path = find_background_image(style_name)
            if image_path is None:
                run_parser.error(
                    f"Unknown mode or background '{style_name}'. "
                    "Use `modes` to list animations or add an image inside the "
                    f"'{BACKGROUND_DIR.name}' folder."
                )
            mode = make_image_mode(image_path)
        run_pomodoro(
            focus_seconds,
            break_seconds,
            mode,
            args.cycles,
            args.windowed,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

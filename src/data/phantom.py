"""
A fake endoscope view, drawn from scratch.

I had no surgical footage I could share, so I drew stand-in frames instead: a round
window, pink tissue with vessels running over it, some glare, and one or two metal
instruments reaching in from the edge. Every frame comes with its own perfect mask
and labels, which real footage never does.

I did two things to keep it a fair test rather than a giveaway. I cut the mask from
the same shapes used to draw the instruments, so it is not painted into the pixels for
a model to find. And I made sure brightness alone does not separate metal from tissue,
so the best score any single brightness cutoff reaches is about 0.28 out of 1. A model
has to learn shape and shading to beat that.

None of this is surgery. Any number measured on it says something about the code,
not about a patient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

N_STEPS = 14
N_INSTRUMENT_KINDS = 18

# Which instruments each step is allowed to show. Every step gets its own pair, so
# the tools in view are a real clue about the step, the way they are in surgery.
STEP_INSTRUMENTS: tuple[tuple[int, ...], ...] = tuple(
    (s % N_INSTRUMENT_KINDS, (s + 5) % N_INSTRUMENT_KINDS) for s in range(N_STEPS)
)

# Tissue shifts from pink toward deep red as the operation goes on, giving the step
# a second, weaker clue. Each video is nudged off the exact colour so a model cannot
# just read the step off one pixel.
_MUCOSA_BRIGHT = np.array([0.94, 0.64, 0.56], dtype=np.float32)
_MUCOSA_DEEP = np.array([0.62, 0.15, 0.19], dtype=np.float32)
_TINT_JITTER = 0.35


@dataclass(frozen=True)
class Frame:
    """One rendered frame with everything known about it."""

    image: np.ndarray  # (size, size, 3), values 0 to 255
    mask: np.ndarray  # (size, size), 0 where tissue, 255 where instrument
    step: int  # which surgical step, from 0 up to N_STEPS
    instruments: tuple[int, ...]  # which kinds are actually visible

    def presence(self, num_kinds: int = N_INSTRUMENT_KINDS) -> list[int]:
        """One slot per instrument kind, 1 if it is in view. This is the label."""
        vec = [0] * num_kinds
        for k in self.instruments:
            vec[k] = 1
        return vec


def render_labelled_frame(rng: np.random.Generator, size: int = 128) -> Frame:
    """
    One standalone frame, unrelated to any other.

    I added a few retries, because an instrument can be aimed so that it misses the
    round window entirely and leaves an empty mask.
    """
    step = int(rng.integers(0, N_STEPS))
    optics = _sample_optics(rng)
    vessels = _sample_vessels(rng)
    tint = float(rng.uniform(-_TINT_JITTER, _TINT_JITTER))
    for _ in range(8):
        tracks = _sample_tracks(rng, step, n_frames=1)
        poses = tuple(p for t in tracks for p in (_pose_at(t, 0),) if p is not None)
        frame = _render(_scene(rng, step, tint, optics, vessels, poses), size)
        if frame.instruments:
            return frame
    raise RuntimeError(f"no instrument landed inside the field of view at size={size}")


def render_video(rng: np.random.Generator, n_frames: int = 40,
                 size: int = 96) -> list[Frame]:
    """
    A run of frames that hang together as one clip.

    Unlike `render_labelled_frame`, the window shape, the vessels and the tissue colour
    stay fixed across the clip, and the instruments drift a little from frame to frame.
    The clip moves through a few steps in ascending order, each lasting a while, which
    is what gives the phase model something to learn from time rather than from one image.
    """
    if n_frames < _MIN_SEGMENT:
        raise ValueError(f"n_frames must be >= {_MIN_SEGMENT}, got {n_frames}")
    optics = _sample_optics(rng)
    vessels = _sample_vessels(rng)
    tint = float(rng.uniform(-_TINT_JITTER, _TINT_JITTER))

    frames: list[Frame] = []
    for step, length in _sample_segments(rng, n_frames):
        tracks = _sample_tracks(rng, step, n_frames=length)
        for t in range(length):
            poses = tuple(p for tr in tracks for p in (_pose_at(tr, t),) if p is not None)
            frames.append(_render(_scene(rng, step, tint, optics, vessels, poses), size))
    return frames


def best_threshold_dice(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    How well plain brightness alone finds the instrument. The bar a model has to clear.

    I tried many brightness cutoffs, in both directions, and kept the best score any of
    them managed. Across the phantom that came out around 0.28 out of 1, so a model
    scoring well above that has learned something beyond "metal is bright".
    """
    target = mask.astype(bool)
    if not target.any():
        return 0.0
    best = 0.0
    for q in np.linspace(1, 99, 33):
        thr = np.percentile(gray, q)
        for pred in ((gray > thr), (gray < thr)):
            denom = pred.sum() + target.sum()
            if denom:
                best = max(best, 2.0 * np.logical_and(pred, target).sum() / denom)
    return float(best)


# Choosing what is in the scene, before anything is drawn.
_MIN_SEGMENT = 5
_MAX_SEGMENTS = 8


@dataclass(frozen=True)
class _Optics:
    centre: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class _Pose:
    kind: int
    entry: tuple[float, float]
    tip: tuple[float, float]
    half_width: float


@dataclass(frozen=True)
class _Track:
    """
    How one instrument moves across a step, described once and evaluated per frame.

    Coordinates run from -1 to 1 across the frame, so the same track works at any size.
    """

    kind: int
    border: float  # where it enters, as a fraction around the frame edge
    border_drift: float  # how fast that entry point slides, per frame
    aim: tuple[float, float]  # the point the shaft is pushed toward
    reach: float  # how far in it reaches
    reach_drift: float
    wobble: float
    wobble_freq: float
    wobble_phase: float
    half_width: float
    withdraw: tuple[int, int]


@dataclass(frozen=True)
class _Scene:
    step: int
    mucosa: np.ndarray
    poses: tuple[_Pose, ...]
    optics: _Optics
    vessels: tuple[tuple[tuple[float, float], ...], ...]
    glare: tuple[tuple[float, float, float, float], ...]
    gain: float
    noise_sigma: float
    blur_radius: float
    texture_seed: int


def _sample_optics(rng: np.random.Generator) -> _Optics:
    return _Optics(centre=(float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.05, 0.05))),
                   radius=float(rng.uniform(0.88, 0.96)))


def _sample_vessels(rng: np.random.Generator) -> tuple[tuple[tuple[float, float], ...], ...]:
    out = []
    for _ in range(int(rng.integers(3, 6))):
        p0 = _border_point(float(rng.random()))
        p2 = _border_point(float(rng.random()))
        p1 = (float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.8, 0.8)))
        out.append(_bezier(p0, p1, p2))
    return tuple(out)


def _sample_segments(rng: np.random.Generator, n_frames: int) -> list[tuple[int, int]]:
    max_segments = min(_MAX_SEGMENTS, n_frames // _MIN_SEGMENT)
    n_seg = int(rng.integers(min(3, max_segments), max_segments + 1))
    steps = sorted(int(s) for s in rng.choice(N_STEPS, size=n_seg, replace=False))

    spare = n_frames - n_seg * _MIN_SEGMENT
    extra = rng.multinomial(spare, [1 / n_seg] * n_seg) if spare > 0 else np.zeros(n_seg, int)
    return [(s, _MIN_SEGMENT + int(e)) for s, e in zip(steps, extra)]


def _sample_tracks(rng: np.random.Generator, step: int, n_frames: int) -> list[_Track]:
    kinds = list(STEP_INSTRUMENTS[step])
    if rng.random() < 0.35:  # sometimes only one tool is in view, not both
        kinds = [kinds[int(rng.integers(0, len(kinds)))]]
    tracks = []
    for kind in kinds:
        withdraw = (0, 0)
        if n_frames > _MIN_SEGMENT and rng.random() < 0.3:
            a = int(rng.integers(0, n_frames - 3))
            withdraw = (a, a + int(rng.integers(2, 5)))
        tracks.append(_Track(
            kind=kind,
            border=float(rng.random()),
            border_drift=float(rng.normal(0.0, 0.004)),
            aim=(float(rng.uniform(-0.55, 0.55)), float(rng.uniform(-0.55, 0.55))),
            reach=float(rng.uniform(0.75, 1.5)),
            reach_drift=float(rng.normal(0.0, 0.030)),
            wobble=float(rng.uniform(0.05, 0.20)),
            wobble_freq=float(rng.uniform(0.04, 0.14)),
            wobble_phase=float(rng.uniform(0.0, 2 * math.pi)),
            half_width=float(rng.uniform(0.055, 0.10)),
            withdraw=withdraw,
        ))
    return tracks


def _pose_at(track: _Track, t: int) -> _Pose | None:
    lo, hi = track.withdraw
    if lo <= t < hi:
        return None
    entry = _border_point(track.border + track.border_drift * t)
    wob = track.wobble * math.sin(2 * math.pi * track.wobble_freq * t + track.wobble_phase)
    aim = (track.aim[0] + wob, track.aim[1] + wob * 0.6)
    dx, dy = aim[0] - entry[0], aim[1] - entry[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    reach = track.reach + track.reach_drift * t
    if reach <= 0.12:
        return None
    tip = (entry[0] + dx / norm * reach, entry[1] + dy / norm * reach)
    return _Pose(track.kind, entry, tip, track.half_width)


def _scene(rng: np.random.Generator, step: int, tint: float, optics: _Optics,
           vessels: tuple[tuple[tuple[float, float], ...], ...],
           poses: tuple[_Pose, ...]) -> _Scene:
    frac = float(np.clip((step + tint) / (N_STEPS - 1), 0.0, 1.0))
    mucosa = _MUCOSA_BRIGHT * (1.0 - frac) + _MUCOSA_DEEP * frac
    glare = tuple(
        (float(rng.uniform(-0.7, 0.7)), float(rng.uniform(-0.7, 0.7)),
         float(rng.uniform(0.04, 0.13)), float(rng.uniform(0.25, 0.65)))
        for _ in range(int(rng.integers(2, 6)))
    )
    return _Scene(
        step=step, mucosa=mucosa, poses=poses, optics=optics, vessels=vessels, glare=glare,
        gain=float(rng.uniform(0.92, 1.06)),
        noise_sigma=float(rng.uniform(0.015, 0.040)),
        blur_radius=float(rng.uniform(0.3, 0.9)),
        texture_seed=int(rng.integers(0, 2 ** 31)),
    )


# Turning a chosen scene into actual pixels.
def _render(scene: _Scene, size: int) -> Frame:
    """
    Draw one frame, and cut its mask from the same shapes.

    Everything is drawn by maths over a grid of coordinates rather than with a drawing
    library, which is what lets the mask come from the instrument shapes themselves.
    """
    if size < 32:
        raise ValueError(f"size must be at least 32 for the shapes to survive, got {size}")
    lin = ((np.arange(size, dtype=np.float32) + 0.5) / size) * 2.0 - 1.0
    u = lin[None, :]  # x of every pixel, as one row repeated down the frame
    v = lin[:, None]  # y of every pixel, as one column repeated across
    px = 2.0 / size  # how wide one pixel is in these coordinates
    rng = np.random.default_rng(scene.texture_seed)

    r = np.hypot(u - scene.optics.centre[0], v - scene.optics.centre[1])
    inside = 1.0 - _smoothstep(scene.optics.radius - 2 * px, scene.optics.radius, r)
    vignette = 1.0 - 0.55 * _smoothstep(0.20, scene.optics.radius, r)

    img = np.broadcast_to(scene.mucosa, (size, size, 3)) * _mucosa_shading(rng, size)[..., None]
    img = img.astype(np.float32)
    for poly in scene.vessels:
        a = 1.0 - _smoothstep(0.008, 0.008 + px, _polyline_dist(u, v, poly))
        img *= 1.0 - 0.55 * a[..., None] * np.array([0.45, 0.75, 0.70], np.float32)

    min_area = max(6, size * size // 2000)
    visible = np.zeros((size, size), np.float32)
    kinds: list[int] = []
    for pose in scene.poses:
        alpha, colour = _instrument(u, v, pose, px)
        clipped = alpha * inside
        if int((clipped > 0.5).sum()) < min_area:
            continue
        img = img * (1.0 - alpha[..., None]) + colour * alpha[..., None]
        visible = np.maximum(visible, clipped)
        kinds.append(pose.kind)

    img = img * (vignette * inside)[..., None] + 0.015 * (1.0 - inside)[..., None]

    for gx, gy, sigma, strength in scene.glare:
        blob = strength * np.exp(-((u - gx) ** 2 + (v - gy) ** 2) / (2 * sigma * sigma))
        img = img + (blob * inside)[..., None]

    img = np.clip(img * scene.gain, 0.0, 1.0)
    rgb = (img * 255.0 + 0.5).astype(np.uint8)
    if scene.blur_radius > 0:
        rgb = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(scene.blur_radius)))
    noisy = rgb.astype(np.float32) / 255.0 + rng.normal(0.0, scene.noise_sigma, rgb.shape)
    rgb = (np.clip(noisy, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    mask = (visible > 0.5).astype(np.uint8) * 255
    return Frame(image=rgb, mask=mask, step=scene.step, instruments=tuple(sorted(set(kinds))))


def _mucosa_shading(rng: np.random.Generator, size: int) -> np.ndarray:
    return 0.74 + 0.34 * _lowfreq(rng, size, 5) + 0.14 * _lowfreq(rng, size, 11)


def _lowfreq(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    small = rng.random((cells, cells)).astype(np.float32)
    return np.asarray(Image.fromarray(small, mode="F").resize((size, size), Image.BICUBIC))


def _instrument(u: np.ndarray, v: np.ndarray, pose: _Pose,
                px: float) -> tuple[np.ndarray, np.ndarray]:
    """One instrument, as how solid it is at each pixel plus what colour it is there."""
    (x0, y0), (x1, y1) = pose.entry, pose.tip
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    t = np.clip(((u - x0) * dx + (v - y0) * dy) / (length * length), 0.0, 1.0)
    qx, qy = x0 + t * dx, y0 + t * dy
    dist = np.hypot(u - qx, v - qy)

    width = pose.half_width * (1.0 - 0.30 * t)
    alpha = 1.0 - _smoothstep(width - px, width + px, dist)
    alpha = np.maximum(alpha, _tip_alpha(u, v, pose, (dx / length, dy / length), px))

    off = (u - qx) * (-dy / length) + (v - qy) * (dx / length)
    n = np.clip(off / pose.half_width, -1.0, 1.0)
    base = 0.40 + 0.055 * (pose.kind % 4)
    body = base * (0.70 + 0.50 * np.sqrt(np.clip(1.0 - n * n, 0.0, 1.0)))
    spec = 0.55 * np.exp(-((n + 0.35) / 0.22) ** 2)
    grey = np.clip((body + spec) * (1.0 - 0.18 * t), 0.0, 1.0)
    colour = np.stack([grey * 0.97, grey, grey * 1.05], axis=-1)
    return alpha, colour


def _tip_alpha(u: np.ndarray, v: np.ndarray, pose: _Pose,
               direction: tuple[float, float], px: float) -> np.ndarray:
    tip = pose.tip
    w = pose.half_width * 0.70
    style = pose.kind % 3
    if style == 0:
        ang = math.atan2(direction[1], direction[0])
        out = np.zeros_like(u + v)
        for sign in (-1.0, 1.0):
            a = ang + sign * 0.55
            end = (tip[0] + 3.4 * w * math.cos(a), tip[1] + 3.4 * w * math.sin(a))
            out = np.maximum(out, _capsule(u, v, tip, end, 0.60 * w, px))
        return out
    if style == 1:
        return _disc(u, v, tip, 1.90 * w, px)
    outer = _disc(u, v, tip, 2.40 * w, px)
    inner = _disc(u, v, tip, 1.30 * w, px)
    return np.clip(outer - inner, 0.0, 1.0)


def _capsule(u: np.ndarray, v: np.ndarray, p0: tuple[float, float],
             p1: tuple[float, float], w: float, px: float) -> np.ndarray:
    return 1.0 - _smoothstep(w - px, w + px, _segment_dist(u, v, p0, p1))


def _disc(u: np.ndarray, v: np.ndarray, c: tuple[float, float],
          r: float, px: float) -> np.ndarray:
    return 1.0 - _smoothstep(r - px, r + px, np.hypot(u - c[0], v - c[1]))


def _segment_dist(u: np.ndarray, v: np.ndarray, p0: tuple[float, float],
                  p1: tuple[float, float]) -> np.ndarray:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return np.hypot(u - p0[0], v - p0[1])
    t = np.clip(((u - p0[0]) * dx + (v - p0[1]) * dy) / denom, 0.0, 1.0)
    return np.hypot(u - (p0[0] + t * dx), v - (p0[1] + t * dy))


def _polyline_dist(u: np.ndarray, v: np.ndarray,
                   poly: tuple[tuple[float, float], ...]) -> np.ndarray:
    out = _segment_dist(u, v, poly[0], poly[1])
    for a, b in zip(poly[1:-1], poly[2:]):
        np.minimum(out, _segment_dist(u, v, a, b), out=out)
    return out


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _border_point(p: float) -> tuple[float, float]:
    q = (p % 1.0) * 4.0
    side, f = int(q), q - int(q)
    if side == 0:
        return (-1.0 + 2.0 * f, -1.0)
    if side == 1:
        return (1.0, -1.0 + 2.0 * f)
    if side == 2:
        return (1.0 - 2.0 * f, 1.0)
    return (-1.0, 1.0 - 2.0 * f)


def _bezier(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float],
            n: int = 24) -> tuple[tuple[float, float], ...]:
    ts = np.linspace(0.0, 1.0, n)
    x = (1 - ts) ** 2 * p0[0] + 2 * (1 - ts) * ts * p1[0] + ts ** 2 * p2[0]
    y = (1 - ts) ** 2 * p0[1] + 2 * (1 - ts) * ts * p1[1] + ts ** 2 * p2[1]
    return tuple((float(a), float(b)) for a, b in zip(x, y))


__all__ = ["N_STEPS", "N_INSTRUMENT_KINDS", "STEP_INSTRUMENTS", "Frame",
           "best_threshold_dice", "render_labelled_frame", "render_video"]

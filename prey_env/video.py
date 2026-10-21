"""MP4 recording of episodes with Mice-look frames (render.py), one frame per physics tick.

Like cellworld_game.video.save_video_output: frames at fps = 1 / tick (40), one file per episode.
Frames are streamed to ffmpeg (moviepy's writer), nothing is kept in memory.
"""
import os
import shutil
import tempfile

from render import Renderer, WIDTH, HEIGHT


def concatenate_videos(paths, out_path: str) -> None:
    """Joins per-episode MP4s into one file, e.g. all_episodes.mp4."""
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    clips = [VideoFileClip(str(p)) for p in paths]
    joined = concatenate_videoclips(clips)
    joined.write_videofile(str(out_path), fps=clips[0].fps, codec="libx264", audio=False, logger=None)
    for clip in clips:
        clip.close()


class EpisodeRecorder:
    """Records every `ticks_per_frame`-th physics tick of `env` while `active`; `save(path)` ends the episode."""

    def __init__(self, env, pov: str = "prey", sprites: bool = True, ticks_per_frame: int = 1):
        self.env = env
        self.renderer = Renderer(env, pov, sprites)
        self.fps = env.freq / ticks_per_frame
        self.every = ticks_per_frame
        self.active = True
        self.tick = self.frames = 0
        self.writer = self.tmp = None
        env.tick_callback = self.on_tick

    def on_tick(self) -> None:
        self.tick += 1
        if self.active and self.tick % self.every == 0:
            self.grab()

    def grab(self) -> None:
        if self.writer is None:
            from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
            self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            self.writer = FFMPEG_VideoWriter(self.tmp, (WIDTH, HEIGHT), self.fps, codec="libx264")
        self.writer.write_frame(self.renderer.frame())
        self.frames += 1

    def save(self, path: str) -> int:
        n = self.frames
        if self.writer is not None:
            self.writer.close()
            shutil.move(self.tmp, path)
            os.chmod(path, 0o644)   # the temp file is private by default
        self.writer = self.tmp = None
        self.tick = self.frames = 0
        return n


def record_episode(env, act, path: str, pov: str = "top", sprites: bool = True) -> dict:
    """Runs one episode with `act(obs) -> action`, saves it to `path` (.mp4), returns a summary."""
    recorder = EpisodeRecorder(env, pov, sprites)
    obs = env.reset()
    recorder.grab()
    done, info, total = False, {}, 0.0
    while not done:
        obs, reward, done, info = env.step(act(obs))
        total += reward
    frames = recorder.save(path)
    return dict(path=path,
                frames=frames,
                length=env.current_step - 1,
                zone=env.ambush_zone,
                surges=getattr(env.predator, "surges", 0),
                captures=info.get("captures", 0),
                goal=bool(info.get("is success", False)),
                timeout=bool(info.get("is truncated", False)),
                ret=total)

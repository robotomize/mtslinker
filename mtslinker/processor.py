import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mtslinker.downloader import download_video_chunk

# Source recordings are mostly a few frames per second. Encoding a steady
# 30 fps picture repeats the same frame and is what made the merge take hours.
OUTPUT_FPS_MIN = 8
OUTPUT_FPS_MAX = 15
TRACK_OVERLAP_TOLERANCE = 1.0
MULTI_TRACK_SIZE = (1920, 1080)
AUDIO_RATE = 48000


@dataclass
class MediaChunk:
    path: str
    start: float
    duration: float
    width: int = 0
    height: int = 0
    fps: Optional[float] = None
    has_audio: bool = False
    has_video: bool = False


def process_video_clips(directory: str, json_data: Dict) -> Tuple[float, List[MediaChunk], List[MediaChunk]]:
    total_duration = float(json_data.get('duration', 0))
    if not total_duration:
        raise ValueError('Duration not found in JSON data.')

    video_clips = []
    audio_clips = []

    for event in json_data.get('eventLogs', []):
        if not isinstance(event, dict):
            continue
        data = event.get('data', {})
        if not isinstance(data, dict) or 'url' not in data:
            continue
        start_time = float(event.get('relativeTime') or 0)
        downloaded_file_path = download_video_chunk(data['url'], directory)
        chunk = inspect_media(downloaded_file_path, start_time)
        if chunk is None:
            logging.error('Skipping unreadable media %s', os.path.basename(downloaded_file_path))
            continue
        if chunk.has_video:
            video_clips.append(chunk)
        elif chunk.has_audio:
            audio_clips.append(chunk)
        else:
            logging.error('No audio or video stream in %s', os.path.basename(downloaded_file_path))

    logging.info(f'Total duration of clips: {total_duration}')
    return total_duration, video_clips, audio_clips


def inspect_media(path: str, start: float) -> Optional[MediaChunk]:
    """Read stream lengths without decoding.

    The container length follows whichever stream is longer. For these
    recordings that is often the audio, which previously stretched the picture
    until the sound ended.
    """
    streams = _ffprobe_streams(path)
    if not streams:
        return None

    video = next((stream for stream in streams if stream.get('codec_type') == 'video' and stream.get('width')), None)
    audio = next((stream for stream in streams if stream.get('codec_type') == 'audio'), None)
    audio_duration = _stream_duration(audio) if audio else None

    if video:
        duration = _stream_duration(video)
        frames = _positive_float(video.get('nb_frames'))
        average = _parse_rate(video.get('avg_frame_rate'))
        if duration is None and frames and average:
            duration = frames / average
        if not duration:
            return None
        fps = frames / duration if frames else average
        if fps is not None and not 0.05 <= fps <= 240:
            fps = None
        if audio_duration and audio_duration - duration > 0.02:
            logging.info(
                'Using video stream length %.3fs instead of audio %.3fs for %s',
                duration,
                audio_duration,
                os.path.basename(path),
            )
        return MediaChunk(
            path=path,
            start=start,
            duration=duration,
            width=int(video['width']),
            height=int(video['height']),
            fps=fps,
            has_audio=audio is not None,
            has_video=True,
        )

    if audio_duration:
        return MediaChunk(
            path=path,
            start=start,
            duration=audio_duration,
            has_audio=True,
        )
    return None


def compile_final_video(total_duration: float, video_clips: Sequence[MediaChunk], audio_clips: Sequence[MediaChunk],
                        output_path: str, max_duration: Union[int, None]):
    duration = float(total_duration)
    if max_duration and duration > max_duration:
        logging.info('Duration limit! Crop!')
        duration = float(max_duration)
    render_with_ffmpeg(duration, list(video_clips), list(audio_clips), output_path)


def render_with_ffmpeg(total_duration: float, video_clips: List[MediaChunk], audio_clips: List[MediaChunk],
                       output_path: str):
    placements, width, height = _layout(video_clips, total_duration)
    if audio_clips:
        audio_pieces = [
            (chunk, start, end)
            for track in _assign_tracks(audio_clips, total_duration)
            for chunk, start, end in track
        ]
    else:
        audio_pieces = [
            (placement[0], placement[1], placement[2])
            for placement in placements
            if placement[0].has_audio
        ]

    fps = _output_fps(video_clips)
    encoder_name, encoder_args = _video_encoder()
    logging.info(
        'Encoding with ffmpeg: %s video, %s audio, %sx%s, %s fps, %s track(s), encoder %s',
        len(placements),
        len(audio_pieces),
        width,
        height,
        fps,
        _track_count(video_clips, total_duration),
        encoder_name,
    )

    inputs: List[str] = []
    input_of: Dict[Tuple[str, float], int] = {}

    def add_input(chunk: MediaChunk) -> int:
        key = (os.path.abspath(chunk.path), round(float(chunk.start), 3))
        if key not in input_of:
            input_of[key] = len(inputs)
            inputs.append(os.path.abspath(chunk.path))
        return input_of[key]

    filters: List[str] = []
    cursor = f'color=c=black:s={width}x{height}:r={fps}:d={total_duration:.3f},format=yuv420p[base]'
    filters.append(cursor)
    current = 'base'
    for index, (chunk, start, end, cell_w, cell_h, x, y, scale_up) in enumerate(placements):
        source = add_input(chunk)
        fitted_w, fitted_h = _fitted_size(chunk.width, chunk.height, cell_w, cell_h, scale_up)
        label = f'v{index}'
        visible = end - start
        filters.append(
            f'[{source}:v]trim=duration={visible:.3f},setpts=PTS-STARTPTS,'
            f'scale={fitted_w}:{fitted_h}:flags=fast_bilinear,'
            f'pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black,'
            f'fps={fps},setpts=PTS-STARTPTS+{start:.3f}/TB,format=yuv420p[{label}]'
        )
        output = 'vout' if index == len(placements) - 1 else f't{index}'
        filters.append(
            f'[{current}][{label}]overlay={x}:{y}:eof_action=pass:repeatlast=0[{output}]'
        )
        current = output

    if not placements:
        filters.append(f'[base]format=yuv420p[vout]')

    audio_labels = []
    if audio_pieces or placements:
        filters.append(
            f'anullsrc=channel_layout=stereo:sample_rate={AUDIO_RATE},'
            f'atrim=duration={total_duration:.3f}[sil]'
        )
        audio_labels.append('[sil]')
    for index, (chunk, start, end) in enumerate(audio_pieces):
        source = add_input(chunk)
        delay_ms = max(0, int(round(start * 1000)))
        visible = end - start
        label = f'a{index}'
        filters.append(
            f'[{source}:a]atrim=duration={visible:.3f},asetpts=PTS-STARTPTS,'
            f'aformat=sample_rates={AUDIO_RATE}:channel_layouts=stereo,'
            f'adelay={delay_ms}|{delay_ms}[{label}]'
        )
        audio_labels.append(f'[{label}]')
    if audio_labels:
        filters.append(
            f'{"".join(audio_labels)}amix=inputs={len(audio_labels)}:'
            f'duration=first:dropout_transition=0:normalize=0[aout]'
        )

    command = [_ffmpeg_binary(), '-y', '-hide_banner']
    for path in inputs:
        command.extend(['-i', path])
    command.extend(['-filter_complex', ';'.join(filters), '-map', '[vout]'])
    if audio_labels:
        command.extend(['-map', '[aout]', '-c:a', 'aac', '-b:a', '128k'])
    else:
        command.append('-an')
    command.extend(encoder_args)
    # Two runs used to open the same mp4 at once. The file then had no valid
    # header and players refused it. Encode into a private part and publish
    # it only after ffmpeg has finished the faststart rewrite.
    partial_path = f'{output_path}.{os.getpid()}.part.mp4'
    lock_path = _acquire_render_lock(output_path)
    command.extend([
        '-r', str(fps),
        '-t', f'{total_duration:.3f}',
        '-movflags', '+faststart',
        partial_path,
    ])
    logging.info('Running ffmpeg (%s inputs)', len(inputs))
    try:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f'ffmpeg exited with status {completed.returncode}')
        os.replace(partial_path, output_path)
        logging.info('Final video written to %s', output_path)
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        _release_render_lock(lock_path)


def _layout(video_clips: Sequence[MediaChunk], total_duration: float):
    """Put the slide or screen on the full frame and the camera in a corner.

    Splitting every stream into an equal grid shrinks the screen share into
    one cell and leaves the rest of the picture black.
    """
    tracks = _assign_tracks(video_clips, total_duration)
    if not tracks:
        width, height = MULTI_TRACK_SIZE
        return [], width, height

    max_width = max(chunk.width for track in tracks for chunk, _, _ in track)
    main_tracks = []
    camera_tracks = []
    for track in tracks:
        widest = max(chunk.width for chunk, _, _ in track)
        if widest < max_width * 0.6:
            camera_tracks.append(track)
        else:
            main_tracks.append(track)
    if not main_tracks:
        main_tracks = camera_tracks
        camera_tracks = []

    width = _even(max(chunk.width for track in main_tracks for chunk, _, _ in track))
    height = _even(max(chunk.height for track in main_tracks for chunk, _, _ in track))

    placements = []
    for track in main_tracks:
        for chunk, start, end in track:
            placements.append((chunk, start, end, width, height, 0, 0, False))

    pip_w = max(2, _even(int(width * 0.22)))
    pip_h = max(2, _even(int(height * 0.22)))
    margin = max(8, _even(int(min(width, height) * 0.03)))
    for index, track in enumerate(camera_tracks):
        for chunk, start, end in track:
            fitted_w, fitted_h = _fitted_size(chunk.width, chunk.height, pip_w, pip_h, True)
            x = max(0, width - fitted_w - margin)
            y = height - fitted_h - margin - index * (fitted_h + margin)
            y = max(0, y)
            placements.append((chunk, start, end, fitted_w, fitted_h, x, y, True))
    return placements, width, height


def _output_fps(video_clips: Sequence[MediaChunk]) -> int:
    rates = [chunk.fps for chunk in video_clips if chunk.fps and chunk.fps >= 1]
    if not rates:
        return OUTPUT_FPS_MIN
    return int(min(OUTPUT_FPS_MAX, max(OUTPUT_FPS_MIN, math.ceil(max(rates)))))


def _track_count(video_clips: Sequence[MediaChunk], total_duration: float) -> int:
    return len(_assign_tracks(video_clips, total_duration))


def _fitted_size(src_w: int, src_h: int, cell_w: int, cell_h: int, scale_up: bool) -> Tuple[int, int]:
    scale = min(cell_w / src_w, cell_h / src_h)
    if not scale_up:
        scale = min(scale, 1.0)
    new_w = max(2, int(round(src_w * scale)))
    new_h = max(2, int(round(src_h * scale)))
    new_w -= new_w % 2
    new_h -= new_h % 2
    return max(2, min(new_w, cell_w)), max(2, min(new_h, cell_h))


def _assign_tracks(clips: Sequence[MediaChunk], total_duration: float) -> List[List[Tuple]]:
    items = []
    for clip in clips:
        start = max(0.0, float(getattr(clip, 'start', 0) or 0))
        duration = float(clip.duration or 0)
        if duration <= 0 or start >= total_duration:
            continue
        end = min(start + duration, total_duration)
        if end - start > 0.02:
            items.append([clip, start, end])

    items.sort(key=lambda item: (item[1], item[2]))
    tracks: List[List[list]] = []
    for item in items:
        placed = False
        for track in tracks:
            previous_end = track[-1][2]
            if item[1] >= previous_end - TRACK_OVERLAP_TOLERANCE:
                if item[1] < previous_end:
                    track[-1][2] = item[1]
                track.append(item)
                placed = True
                break
        if not placed:
            tracks.append([item])

    laid_out = []
    for track in tracks:
        kept = [
            (clip, start, end)
            for clip, start, end in track
            if end - start > 0.02
        ]
        if kept:
            laid_out.append(kept)
    return laid_out


def _even(value: int) -> int:
    value = max(2, int(value))
    return value if value % 2 == 0 else value + 1


def _acquire_render_lock(output_path: str) -> str:
    lock_path = f'{output_path}.lock'
    if os.path.exists(lock_path):
        pid = _lock_pid(lock_path)
        if pid and _pid_alive(pid):
            raise RuntimeError(
                f'Another encode is already writing {output_path} (pid {pid}). Wait until it finishes.'
            )
        os.remove(lock_path)
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, f'{os.getpid()}\n'.encode())
    finally:
        os.close(fd)
    return lock_path


def _release_render_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _lock_pid(lock_path: str) -> Optional[int]:
    try:
        text = open(lock_path, encoding='utf-8').read().strip()
        return int(text)
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _ffmpeg_binary() -> str:
    found = shutil.which('ffmpeg')
    if found:
        return found
    raise RuntimeError('ffmpeg was not found on PATH. Install it and run the merge again.')


def _ffprobe_streams(path: str) -> List[dict]:
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        raise RuntimeError('ffprobe was not found on PATH. It is installed together with ffmpeg.')
    try:
        raw = subprocess.check_output(
            [
                ffprobe,
                '-v', 'error',
                '-show_entries', 'stream=codec_type,width,height,duration,nb_frames,avg_frame_rate',
                '-of', 'json',
                path,
            ],
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as error:
        logging.error('ffprobe failed for %s: %s', path, error)
        return []
    try:
        return json.loads(raw.decode('utf-8')).get('streams') or []
    except (ValueError, UnicodeError):
        return []


def _stream_duration(stream: Optional[dict]) -> Optional[float]:
    if not stream:
        return None
    duration = _positive_float(stream.get('duration'))
    if duration:
        return duration
    frames = _positive_float(stream.get('nb_frames'))
    average = _parse_rate(stream.get('avg_frame_rate'))
    if frames and average:
        return frames / average
    return None


def _video_encoder() -> Tuple[str, List[str]]:
    if _ffmpeg_has_encoder('h264_videotoolbox'):
        return 'h264_videotoolbox', [
            '-c:v', 'h264_videotoolbox',
            '-b:v', '2M',
            '-allow_sw', '1',
            '-pix_fmt', 'yuv420p',
        ]
    return 'libx264', [
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '26',
        '-pix_fmt', 'yuv420p',
        '-threads', str(os.cpu_count() or 0),
    ]


def _ffmpeg_has_encoder(name: str) -> bool:
    try:
        output = subprocess.check_output(
            [_ffmpeg_binary(), '-hide_banner', '-encoders'],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return any(len(line.split()) >= 2 and line.split()[1] == name for line in output.splitlines())


def _parse_rate(value) -> Optional[float]:
    if not value or value in {'0/0', 'N/A'}:
        return None
    if isinstance(value, str) and '/' in value:
        numerator, denominator = value.split('/', 1)
        den = _positive_float(denominator)
        num = _positive_float(numerator)
        if not den or num is None:
            return None
        return num / den
    return _positive_float(value)


def _positive_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number

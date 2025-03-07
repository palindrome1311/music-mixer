import librosa
import numpy as np
from pydub import AudioSegment
import os
import logging
import subprocess
import scipy.ndimage

logger = logging.getLogger(__name__)

class AutoMixer:
    """
    A class for automatically mixing two songs together based on beat strength and vocal presence.
    Creates transitions at points with strong beats and minimal vocal content.
    """
    
    def __init__(self, ffmpeg_path, ffprobe_path):
        """
        Initialize the AutoMixer with FFmpeg paths.
        
        Args:
            ffmpeg_path (str): Path to FFmpeg executable
            ffprobe_path (str): Path to FFprobe executable
        """
        self.supported_formats = ['.mp3', '.wav']
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        
        # Configure pydub to use FFmpeg paths
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path
        AudioSegment.ffprobe = ffprobe_path
        
        # Verify FFmpeg installation
        if not os.path.exists(ffmpeg_path) or not os.path.exists(ffprobe_path):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and update the paths.")

    def _run_ffmpeg(self, args):
        """Run an FFmpeg command with the given arguments."""
        try:
            process = subprocess.Popen(
                [self.ffmpeg_path] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr.decode()}")
            return stdout
        except Exception as e:
            logger.error(f"Error running FFmpeg: {str(e)}")
            raise

    def load_audio(self, audio_path):
        """
        Load an audio file and convert it to WAV format.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            AudioSegment: The loaded audio
        """
        try:
            file_ext = os.path.splitext(audio_path)[1].lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported audio format. Supported formats: {self.supported_formats}")
            
            # Convert to WAV first
            temp_wav = os.path.splitext(audio_path)[0] + "_temp.wav"
            self._run_ffmpeg([
                '-i', audio_path,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-y',
                temp_wav
            ])
            
            audio = AudioSegment.from_wav(temp_wav)
            os.remove(temp_wav)
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            raise

    def get_vocal_activity(self, y, sr, start_time, duration):
        """
        Get detailed vocal activity information for a segment of audio.
        Returns vocal activity score and whether we're in the middle of a phrase.
        """
        start_sample = int(start_time * sr)
        end_sample = start_sample + int(duration * sr)
        segment = y[start_sample:end_sample]
        
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(segment)
        
        # Get mel spectrograms
        mel_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=sr, 
                                                     n_mels=128, fmin=200, fmax=3000)
        mel_percussive = librosa.feature.melspectrogram(y=y_percussive, sr=sr,
                                                       n_mels=128, fmin=200, fmax=3000)
        
        # Convert to dB scale
        mel_harmonic_db = librosa.power_to_db(mel_harmonic, ref=np.max)
        mel_percussive_db = librosa.power_to_db(mel_percussive, ref=np.max)
        
        # Calculate vocal range energy
        harmonic_energy = np.mean(mel_harmonic_db[20:80, :])  # Focus on vocal frequency range
        percussive_energy = np.mean(mel_percussive_db[20:80, :])
        
        # Calculate vocal activity score
        vocal_score = (harmonic_energy - percussive_energy) / 80.0
        vocal_score = np.clip((vocal_score + 1.0) / 2.0, 0.0, 1.0)
        
        # Detect if we're in the middle of a phrase
        # Look at the energy pattern in the last second
        frame_length = sr // 10  # 100ms frames
        energy_frames = librosa.feature.rms(y=y_harmonic, frame_length=frame_length)[0]
        
        # Check if energy is decreasing (phrase ending) or increasing (phrase beginning)
        if len(energy_frames) > 2:
            energy_trend = np.mean(np.diff(energy_frames[-5:]))  # Look at last 500ms
            in_phrase = energy_trend > -0.001  # Negative trend indicates phrase ending
        else:
            in_phrase = True
            
        return vocal_score, in_phrase

    def analyze_segments(self, audio_path, min_segment_duration=4, max_segment_duration=16):
        """
        Analyze audio file in segments based on musical structure and beat patterns.
        
        Args:
            audio_path (str): Path to the audio file
            min_segment_duration (int): Minimum duration of a segment in seconds
            max_segment_duration (int): Maximum duration of a segment in seconds
            
        Returns:
            list: List of segment information dictionaries
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # Load audio
            y, sr = librosa.load(audio_path)
            total_duration = len(y) / sr
            
            # Get beat frames and onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            if len(beat_times) < 2:
                logger.warning("Not enough beats detected, using fixed-duration segments")
                # Fall back to fixed-duration segments
                segment_duration = 8  # 8 seconds per segment
                boundary_times = np.arange(0, total_duration, segment_duration)
                if boundary_times[-1] < total_duration:
                    boundary_times = np.append(boundary_times, total_duration)
            else:
                # Compute mel spectrogram for segmentation
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
                
                # Compute recurrence matrix
                R = librosa.segment.recurrence_matrix(mfcc, mode='affinity')
                
                try:
                    # Find segment boundaries using agglomerative clustering
                    segment_frames = librosa.segment.agglomerative(R, len(beat_frames))
                    boundary_times = librosa.frames_to_time(segment_frames, sr=sr)
                except Exception as e:
                    logger.warning(f"Agglomerative clustering failed: {str(e)}, falling back to beat-based segmentation")
                    # Fall back to beat-based segmentation
                    boundary_times = [0]
                    current_time = 0
                    for beat_time in beat_times:
                        if beat_time - current_time >= min_segment_duration:
                            boundary_times.append(beat_time)
                            current_time = beat_time
            
            # Ensure we have at least one valid segment
            if len(boundary_times) < 2:
                logger.warning("No valid boundaries found, creating default segments")
                boundary_times = [0, min(total_duration, 30)]
            
            # Ensure minimum segment length and merge short segments
            filtered_boundaries = [0]  # Start with beginning of song
            for time in boundary_times:
                if time > 0 and (time - filtered_boundaries[-1]) >= min_segment_duration:
                    filtered_boundaries.append(time)
            
            # Add end of song if needed
            if total_duration - filtered_boundaries[-1] >= min_segment_duration:
                filtered_boundaries.append(total_duration)
            
            segments_info = []
            
            # Analyze each segment
            for i in range(len(filtered_boundaries) - 1):
                try:
                    start_time = filtered_boundaries[i]
                    end_time = filtered_boundaries[i + 1]
                    
                    # Split long segments at beat positions
                    if end_time - start_time > max_segment_duration:
                        segment_beats = [b for b in beat_times if start_time <= b <= end_time]
                        sub_segments = []
                        current_start = start_time
                        
                        for beat in segment_beats:
                            if beat - current_start >= min_segment_duration:
                                sub_segments.append((current_start, beat))
                                current_start = beat
                        
                        if end_time - current_start >= min_segment_duration:
                            sub_segments.append((current_start, end_time))
                    else:
                        sub_segments = [(start_time, end_time)]
                    
                    # Process each sub-segment
                    for sub_start, sub_end in sub_segments:
                        # Get segment data
                        start_sample = int(sub_start * sr)
                        end_sample = int(sub_end * sr)
                        segment = y[start_sample:end_sample]
                        
                        # Calculate beat strength using onset envelope
                        segment_onset = librosa.onset.onset_strength(y=segment, sr=sr)
                        beat_strength = np.mean(segment_onset) if len(segment_onset) > 0 else 0
                        
                        # Get vocal strength
                        vocal_strength = self.get_vocal_activity(y, sr, sub_start, sub_end - sub_start)
                        
                        # Calculate segment energy (volume)
                        energy = float(np.mean(librosa.feature.rms(y=segment)))
                        
                        segments_info.append({
                            'start_time': sub_start,
                            'end_time': sub_end,
                            'beat_strength': float(beat_strength),
                            'vocal_strength': float(vocal_strength[0]),
                            'energy': energy,
                            'duration': sub_end - sub_start
                        })
                    
                except Exception as e:
                    logger.error(f"Error analyzing segment at {start_time:.1f}s: {str(e)}")
                    continue
            
            if not segments_info:
                logger.warning("No valid segments found, creating a default segment")
                segments_info.append({
                    'start_time': 0,
                    'end_time': min(total_duration, 30),
                    'beat_strength': 0.5,
                    'vocal_strength': 0.5,
                    'energy': 0.5,
                    'duration': min(total_duration, 30)
                })
            
            # Calculate max energy safely
            max_energy = max(s['energy'] for s in segments_info)
            if max_energy == 0:
                max_energy = 1.0
            
            # Sort segments by musical quality (combination of beat strength and energy)
            segments_info.sort(
                key=lambda x: (float(x['beat_strength']) * 0.7 + (float(x['energy']) / max_energy) * 0.3),
                reverse=True
            )
            
            logger.info(f"Found {len(segments_info)} segments with durations between "
                       f"{min([s['duration'] for s in segments_info]):.1f}s and "
                       f"{max([s['duration'] for s in segments_info]):.1f}s")
                
            return segments_info
            
        except Exception as e:
            logger.error(f"Error analyzing segments: {str(e)}")
            raise

    def find_best_matching_segments(self, segments1, segments2, num_matches=5):
        """
        Find the best matching segments between two songs based on beat strength and vocal presence.
        
        Args:
            segments1 (list): Segments from first song
            segments2 (list): Segments from second song
            num_matches (int): Number of top matches to return
            
        Returns:
            list: List of tuples containing top matching segments and their scores
        """
        matches = []
        min_beat_strength = 0.4
        max_vocal_strength = 0.7
        
        # Calculate normalization factors
        avg_beat_strength1 = np.mean([seg['beat_strength'] for seg in segments1])
        avg_beat_strength2 = np.mean([seg['beat_strength'] for seg in segments2])
        max_beat_strength = max(avg_beat_strength1, avg_beat_strength2) * 2
        
        # Sort by beat strength
        segments1.sort(key=lambda x: x['beat_strength'], reverse=True)
        segments2.sort(key=lambda x: x['beat_strength'], reverse=True)
        
        # Find matches
        for seg1 in segments1:
            for seg2 in segments2:
                if seg1['beat_strength'] < min_beat_strength or seg2['beat_strength'] < min_beat_strength:
                    continue
                
                # Calculate scores
                strength1 = min(seg1['beat_strength'] / max_beat_strength, 1.0)
                strength2 = min(seg2['beat_strength'] / max_beat_strength, 1.0)
                strength_score = (strength1 + strength2) / 2
                
                vocal1 = seg1.get('vocal_strength', 0.5)
                vocal2 = seg2.get('vocal_strength', 0.5)
                vocal_score = 1 - ((vocal1 + vocal2) / 2)
                
                if (vocal1 + vocal2) / 2 > max_vocal_strength:
                    continue
                
                # Combined score (80% beat strength, 20% vocal avoidance)
                score = (strength_score * 0.8) + (vocal_score * 0.2)
                
                matches.append((seg1, seg2, score))
                logger.debug(f"Match found: Score={score:.2f}, Beat Strength={strength_score:.2f}, "
                          f"Song1 at {seg1['start_time']:.1f}s, Song2 at {seg2['start_time']:.1f}s")
        
        if not matches:
            raise ValueError("Could not find suitable matching segments")
        
        # Sort matches by score and take top N
        matches.sort(key=lambda x: x[2], reverse=True)
        top_matches = matches[:num_matches]
        
        for i, (seg1, seg2, score) in enumerate(top_matches, 1):
            logger.info(f"Match #{i}: Score={score:.2f}, "
                       f"Song1 at {seg1['start_time']:.1f}s, Song2 at {seg2['start_time']:.1f}s")
        
        return top_matches

    def find_natural_transition(self, y, sr, start_time, min_duration=8, max_duration=40):
        """Find a natural transition point based on multiple musical features."""
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)
        
        valid_points = []
        look_ahead = 2.0
        
        for beat_time in beat_times:
            if min_duration <= (beat_time - start_time) <= max_duration:
                vocal_score, in_phrase = self.get_vocal_activity(y, sr, beat_time - 1, 3)
                
                if in_phrase and vocal_score > 0.4:
                    continue
                
                idx = np.argmin(np.abs(onset_times - beat_time))
                window_size = int(sr * 1.0)
                beat_sample = int(beat_time * sr)
                window_start = max(0, beat_sample - window_size)
                window_end = min(len(y), beat_sample + window_size)
                
                onset_strength = onset_env[max(0, idx - 2):min(len(onset_env), idx + 3)].mean()
                window_energy = np.mean(np.abs(y[window_start:window_end]))
                
                future_vocal_score = 0
                for t in np.arange(0.1, look_ahead, 0.1):
                    if beat_time + t >= len(y)/sr:
                        break
                    v_score, _ = self.get_vocal_activity(y, sr, beat_time + t, 0.2)
                    future_vocal_score = max(future_vocal_score, v_score)
                
                beat_in_bar = (np.where(beat_times == beat_time)[0][0] % 4) == 0
                
                score = (
                    onset_strength * 0.2 +
                    window_energy * 0.2 +
                    vocal_score * 0.3 +
                    future_vocal_score * 0.2 +
                    (0.0 if beat_in_bar else 0.1)
                )
                
                valid_points.append((beat_time, score))
        
        if not valid_points:
            return start_time + min_duration
        
        valid_points.sort(key=lambda x: x[1])
        return valid_points[0][0]

    def find_next_natural_break(self, y, sr, start_time, min_duration=8, max_duration=60):
        """Find the next natural break in the music using multiple features."""
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)
        
        valid_breaks = []
        look_ahead = 3.0
        
        for beat_time in beat_times:
            if beat_time - start_time >= min_duration:
                vocal_score, in_phrase = self.get_vocal_activity(y, sr, beat_time - 1, 3)
                
                if in_phrase and vocal_score > 0.4:
                    continue
                
                window_size = int(sr * 1.0)
                beat_sample = int(beat_time * sr)
                pre_window = y[max(0, beat_sample - window_size):beat_sample]
                post_window = y[beat_sample:min(len(y), beat_sample + window_size)]
                
                pre_energy = np.mean(np.abs(pre_window))
                post_energy = np.mean(np.abs(post_window))
                energy_drop = pre_energy - post_energy
                
                future_vocal_activity = False
                for t in np.arange(0.1, look_ahead, 0.1):
                    if beat_time + t >= len(y)/sr:
                        break
                    v_score, in_p = self.get_vocal_activity(y, sr, beat_time + t, 0.2)
                    if v_score > 0.4 and in_p:
                        future_vocal_activity = True
                        break
                
                if future_vocal_activity:
                    continue
                
                beat_in_bar = (np.where(beat_times == beat_time)[0][0] % 4) == 0
                
                score = (
                    (energy_drop * 0.3 if energy_drop > 0 else 0) +
                    (0.4 if not in_phrase else 0) +
                    (0.3 if beat_in_bar else 0)
                )
                
                valid_breaks.append((beat_time, score))
                
                if beat_time - start_time > max_duration and score > 0.5:
                    break
        
        if not valid_breaks:
            return start_time + min_duration
        
        valid_breaks.sort(key=lambda x: x[1], reverse=True)
        return valid_breaks[0][0]

    def detect_chorus(self, y, sr):
        """
        Detect chorus sections in a song using repetition and energy analysis.
        Returns a list of chorus segments with their start times and durations.
        """
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, fmin=30, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc = librosa.util.normalize(mfcc, axis=1)
        
        R = librosa.segment.recurrence_matrix(
            mfcc, 
            mode='affinity',
            width=3,
            k=2*mfcc.shape[0],
            sym=True
        )
        
        sigma = 2
        diagonal_gaussian = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + 1)**2) / sigma**2)
        R = scipy.ndimage.convolve1d(R, diagonal_gaussian, axis=0)
        R = scipy.ndimage.convolve1d(R, diagonal_gaussian, axis=1)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        novelty = np.mean([
            np.gradient(np.mean(contrast, axis=0)),
            onset_env,
            np.gradient(np.mean(S_db, axis=0))
        ], axis=0)
        
        novelty = scipy.ndimage.gaussian_filter1d(novelty, sigma=2)
        
        threshold = np.mean(novelty) + 0.5 * np.std(novelty)
        peaks = librosa.util.peak_pick(
            novelty,
            pre_max=30,
            post_max=30,
            pre_avg=30,
            post_avg=30,
            delta=threshold,
            wait=20
        )
        
        boundary_times = librosa.frames_to_time(peaks, sr=sr)
        
        if len(boundary_times) == 0 or boundary_times[0] > 0:
            boundary_times = np.concatenate([[0], boundary_times])
        if boundary_times[-1] < len(y)/sr:
            boundary_times = np.concatenate([boundary_times, [len(y)/sr]])
        
        segments = []
        min_duration = 6
        max_duration = 35
        
        for i in range(len(boundary_times)-1):
            start_time = boundary_times[i]
            end_time = boundary_times[i+1]
            duration = end_time - start_time
            
            if min_duration <= duration <= max_duration:
                seg_start = int(start_time * sr)
                seg_end = int(end_time * sr)
                segment = y[seg_start:seg_end]
                
                rms = librosa.feature.rms(y=segment, frame_length=2048, hop_length=512)[0]
                energy = np.mean(rms)
                energy_std = np.std(rms)
                
                vocal_score, _ = self.get_vocal_activity(y, sr, start_time, duration)
                
                seg_frames = np.arange(int(start_time * sr / 512), int(end_time * sr / 512))
                if len(seg_frames) > 0:
                    repetition_score = np.mean([R[i, j] for i in seg_frames for j in seg_frames])
                    if repetition_score < 0.1:
                        repetition_score = 0
                else:
                    repetition_score = 0
                
                onset_env_segment = librosa.onset.onset_strength(y=segment, sr=sr)
                tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env_segment, sr=sr)
                
                rhythm_strength = 0.0
                if len(onset_env_segment) > 0:
                    peaks = librosa.util.peak_pick(onset_env_segment,
                                               pre_max=7, post_max=7,
                                               pre_avg=7, post_avg=7,
                                               delta=0.1, wait=10)
                    
                    if len(peaks) > 0:
                        peak_heights = onset_env_segment[peaks]
                        peak_intervals = np.diff(peaks)
                        avg_peak_height = np.mean(peak_heights)
                        peak_regularity = 1.0 / (np.std(peak_intervals) + 1e-6)
                        rhythm_strength = np.clip((avg_peak_height * 0.6 + peak_regularity * 0.4) / 2.0, 0, 1)
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'energy': float(energy),
                    'energy_stability': float(energy_std),
                    'vocal_score': vocal_score,
                    'repetition_score': repetition_score,
                    'rhythm_strength': float(rhythm_strength)
                })
        
        if not segments:
            segment_duration = 20
            for start_time in np.arange(0, len(y)/sr - segment_duration, segment_duration/2):
                end_time = min(start_time + segment_duration, len(y)/sr)
                segment = y[int(start_time * sr):int(end_time * sr)]
                
                energy = float(librosa.feature.rms(y=segment)[0].mean())
                vocal_score, _ = self.get_vocal_activity(y, sr, start_time, end_time - start_time)
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'energy': energy,
                    'energy_stability': 0.5,
                    'vocal_score': vocal_score,
                    'repetition_score': 0.5,
                    'rhythm_strength': 0.5
                })
        
        def normalize_scores(segments):
            for feature in ['energy', 'repetition_score', 'rhythm_strength']:
                values = [s[feature] for s in segments]
                if values:
                    min_v, max_v = min(values), max(values)
                    if max_v > min_v:
                        for s in segments:
                            s[feature] = max(0.2, (s[feature] - min_v) / (max_v - min_v))
                    else:
                        for s in segments:
                            s[feature] = 0.5
            return segments
        
        segments = normalize_scores(segments)
        
        for segment in segments:
            segment['chorus_score'] = (
                segment['energy'] * 0.25 +
                segment['vocal_score'] * 0.25 +
                segment['repetition_score'] * 0.25 +
                segment['rhythm_strength'] * 0.25
            )
        
        segments.sort(key=lambda x: x['chorus_score'], reverse=True)
        chorus_segments = [s for s in segments if s['chorus_score'] > 0.4]
        
        if not chorus_segments and segments:
            chorus_segments = segments[:3]
        
        logger.info(f"Found {len(chorus_segments)} chorus segments")
        return chorus_segments

    def find_best_matching_choruses(self, chorus_segments1, chorus_segments2, y1, sr1, y2, sr2, num_matches=3):
        """
        Find the best matching chorus segments between two songs based on beat strength and energy.
        """
        matches = []
        
        for seg1 in chorus_segments1:
            for seg2 in chorus_segments2:
                # Get beat patterns for both segments
                start1, end1 = int(seg1['start_time'] * sr1), int(seg1['end_time'] * sr1)
                start2, end2 = int(seg2['start_time'] * sr2), int(seg2['end_time'] * sr2)
                
                y1_seg = y1[start1:end1]
                y2_seg = y2[start2:end2]
                
                # Get onset envelopes
                onset_env1 = librosa.onset.onset_strength(y=y1_seg, sr=sr1)
                onset_env2 = librosa.onset.onset_strength(y=y2_seg, sr=sr2)
                
                # Calculate beat similarity
                # Resample to same length for comparison
                target_len = min(len(onset_env1), len(onset_env2))
                
                # Use the new resampling API and ensure exact lengths
                onset_env1_resampled = librosa.resample(
                    y=onset_env1,
                    orig_sr=len(onset_env1),
                    target_sr=target_len
                )[:target_len]  # Ensure exact length
                onset_env2_resampled = librosa.resample(
                    y=onset_env2,
                    orig_sr=len(onset_env2),
                    target_sr=target_len
                )[:target_len]  # Ensure exact length
                
                # Normalize the onset envelopes
                onset_env1_resampled = (onset_env1_resampled - onset_env1_resampled.mean()) / (onset_env1_resampled.std() + 1e-8)
                onset_env2_resampled = (onset_env2_resampled - onset_env2_resampled.mean()) / (onset_env2_resampled.std() + 1e-8)
                
                # Calculate energy similarity
                energy_similarity = 1 - abs(seg1['energy'] - seg2['energy'])
                
                # Calculate overall match score
                match_score = (
                    np.corrcoef(onset_env1_resampled, onset_env2_resampled)[0, 1] * 0.6 +          # Beat pattern similarity
                    energy_similarity * 0.2 +        # Energy level similarity
                    min(seg1['chorus_score'], seg2['chorus_score']) * 0.2  # Chorus confidence
                )
                
                matches.append((seg1, seg2, match_score))
        
        if not matches:
            raise ValueError("Could not find suitable matching chorus segments")
        
        # Sort matches by score and take top N
        matches.sort(key=lambda x: x[2], reverse=True)
        top_matches = matches[:num_matches]
        
        for i, (seg1, seg2, score) in enumerate(top_matches, 1):
            logger.info(f"Chorus match #{i}: Score={score:.2f}, "
                       f"Song1 at {seg1['start_time']:.1f}s, Song2 at {seg2['start_time']:.1f}s")
        
        return top_matches

    def mix_songs(self, song1_path, song2_path, output_path=None, crossfade_duration=2000):
        """
        Mix two songs together based on chorus detection and beat matching.
        Creates a mix focusing on chorus sections with clean transitions.
        Ensures vocals aren't cut off and mix is at least 30 seconds.
        Returns a list of successful mixes with their details.
        
        Args:
            song1_path (str): Path to first song
            song2_path (str): Path to second song
            output_path (str, optional): Output path for mixed file
            crossfade_duration (int): Maximum duration of crossfade in milliseconds
            
        Returns:
            list: List of dictionaries containing mix details and paths
        """
        if output_path is None:
            song1_name = os.path.splitext(os.path.basename(song1_path))[0]
            song2_name = os.path.splitext(os.path.basename(song2_path))[0]
            output_base = f"{song1_name}_{song2_name}"
        else:
            output_base = os.path.splitext(output_path)[0]
        
        y1, sr1 = librosa.load(song1_path)
        y2, sr2 = librosa.load(song2_path)
        
        logger.info("Analyzing songs...")
        chorus_segments1 = self.detect_chorus(y1, sr1)
        chorus_segments2 = self.detect_chorus(y2, sr2)
        
        if not chorus_segments1 or not chorus_segments2:
            raise ValueError("Could not detect chorus sections")
        
        matches = self.find_best_matching_choruses(
            chorus_segments1, chorus_segments2, y1, sr1, y2, sr2
        )
        
        successful_mixes = []
        
        for mix_index, (seg1, seg2, score) in enumerate(matches, 1):
            try:
                song1 = self.load_audio(song1_path).set_channels(1) - 3
                song2 = self.load_audio(song2_path).set_channels(1) - 3
                
                lead_in = 2000
                start1 = max(0, int((seg1['start_time'] * 1000) - lead_in))
                
                seg1_end = self.find_natural_transition(y1, sr1, seg1['start_time'], 
                                                      min_duration=15, max_duration=25)
                
                seg2_end = self.find_next_natural_break(y2, sr2, seg2['start_time'],
                                                      min_duration=15, max_duration=25)
                
                final_end = self.find_next_natural_break(y2, sr2, seg2_end,
                                                       min_duration=2, max_duration=5)
                
                dur1 = (seg1_end - seg1['start_time']) * 1000
                dur2 = (final_end - seg2['start_time']) * 1000
                
                song1_segment = song1[start1:int(start1 + dur1 + 2000)]
                
                song2_start = int(seg2['start_time'] * 1000)
                song2_end = int(final_end * 1000) + 3000
                song2_segment = song2[song2_start:song2_end]
                
                fade_duration = min(3000, len(song2_segment))
                song2_segment = song2_segment.fade_out(fade_duration)
                
                tempo1, beats1 = librosa.beat.beat_track(y=y1[int(seg1_end*sr1-sr1):int(seg1_end*sr1+sr1)], sr=sr1)
                tempo2, beats2 = librosa.beat.beat_track(y=y2[int(seg2['start_time']*sr2):int(seg2['start_time']*sr2+sr2)], sr=sr2)
                
                avg_beat_time = (60.0 / tempo1 + 60.0 / tempo2) / 2
                optimal_crossfade = int(min(4 * avg_beat_time * 1000, crossfade_duration))
                optimal_crossfade = max(optimal_crossfade, 2000)
                
                min_segment_duration = min(len(song1_segment), len(song2_segment))
                actual_crossfade = min(optimal_crossfade, min_segment_duration - 1000)
                actual_crossfade = max(actual_crossfade, 2000)
                
                mixed = song1_segment.append(song2_segment, crossfade=actual_crossfade)
                
                if len(mixed) < 30000:
                    continue
                
                current_output_path = f"{output_base}_mix{mix_index}.mp3"
                mixed.export(current_output_path, format="mp3")
                
                successful_mixes.append({
                    'path': current_output_path,
                    'score': score,
                    'duration': len(mixed) / 1000,
                    'song1_start': seg1['start_time'],
                    'song1_duration': dur1/1000,
                    'song2_start': seg2['start_time'],
                    'song2_duration': dur2/1000,
                    'crossfade': actual_crossfade/1000
                })
                
            except Exception as e:
                logger.warning(f"Failed to create mix {mix_index}")
                continue
        
        if not successful_mixes:
            raise ValueError("Could not create any satisfactory mixes")
        
        return successful_mixes

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    FFMPEG_PATH = "C:\\Users\\mitta\\Downloads\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"
    FFPROBE_PATH = "C:\\Users\\mitta\\Downloads\\ffmpeg-master-latest-win64-gpl\\bin\\ffprobe.exe"
    
    try:
        mixer = AutoMixer(FFMPEG_PATH, FFPROBE_PATH)
        song2_path = "C:\\Users\\mitta\\Downloads\\ishq.mp3"
        song1_path = "C:\\Users\\mitta\\Downloads\\kaise.mp3"
        
        logger.info("Creating mixes...")
        mixes = mixer.mix_songs(song1_path, song2_path)
        logger.info(f"Created {len(mixes)} mixes")
        
        for i, mix in enumerate(mixes, 1):
            logger.info(f"Mix {i}: {mix['duration']:.1f}s, Score: {mix['score']:.2f}, Path: {mix['path']}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise 
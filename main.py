import pyaudio
import numpy as np
from scipy import signal
import logging
import argparse
import wave

CHUNK = 2048
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

logging.basicConfig(level=logging.INFO)

# Default pitch range for bass drum (in Hz)
default_min_pitch = 60
default_max_pitch = 100

def ensure_array(data):
    """Ensure the input is a numpy array of the correct size."""
    if np.isscalar(data):
        return np.full(CHUNK, data, dtype=np.float32)
    data = np.array(data, dtype=np.float32)
    if len(data) != CHUNK:
        return resize_to_chunk(data)
    return data

def resize_to_chunk(data):
    """Resize the data to match CHUNK size."""
    if len(data) > CHUNK:
        return data[:CHUNK]
    elif len(data) < CHUNK:
        return np.pad(data, (0, CHUNK - len(data)), 'constant')
    return data

def find_loudest_segment(data, segment_length):
    """Find the loudest segment of the given length in the data."""
    if len(data) <= segment_length:
        return data
    
    segment_energy = np.array([
        np.sum(data[i:i+segment_length]**2)
        for i in range(0, len(data) - segment_length + 1)
    ])
    loudest_start = np.argmax(segment_energy)
    return data[loudest_start:loudest_start+segment_length]

def time_stretch(data, stretch_factor):
    """Time stretch the audio data and return the loudest chunk."""
    data = ensure_array(data)
    stretched = signal.resample(data, int(len(data) * stretch_factor))
    
    # Find the loudest CHUNK-sized segment
    loudest_segment = find_loudest_segment(stretched, CHUNK)
    
    return resize_to_chunk(loudest_segment)

def transpose_to_low_frequency(data, min_freq, max_freq):
    """Transpose the pitch of the audio data to fit within the specified low frequency range."""
    try:
        data = ensure_array(data)
        f, t, Zxx = signal.stft(data, fs=RATE, nperseg=256)
        mag, phase = np.abs(Zxx), np.angle(Zxx)
        new_mag = np.zeros_like(mag)
        max_freq_index = np.argmax(f > max_freq)
        
        for i in range(mag.shape[1]):
            for j in range(max_freq_index):
                freq = f[j]
                if freq < min_freq:
                    new_index = int((freq / min_freq) * (max_freq_index * min_freq / max_freq))
                else:
                    new_index = j
                new_mag[new_index, i] += mag[j, i]
        
        new_mag[max_freq_index:, :] = 0
        new_Zxx = new_mag * np.exp(1j * phase)
        _, transposed = signal.istft(new_Zxx, fs=RATE)
        return resize_to_chunk(transposed).astype(np.float32)
    except Exception as e:
        logging.error(f"Error in transpose_to_low_frequency: {str(e)}")
        return data

def low_pass_filter(data, cutoff, order=6):
    """Apply a low-pass filter to the audio data."""
    data = ensure_array(data)
    nyq = 0.5 * RATE
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, data)

def process_audio(input_data, stretch_factor, min_pitch, max_pitch):
    try:
        # Ensure input is an array
        input_data = ensure_array(input_data)
        logging.info(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")

        # Time stretch and select loudest segment
        stretched = time_stretch(input_data, stretch_factor)
        logging.info(f"Stretched (loudest segment) shape: {stretched.shape}, dtype: {stretched.dtype}")

        # Transpose to low frequency
        transposed = transpose_to_low_frequency(stretched, min_pitch, max_pitch)
        logging.info(f"Transposed shape: {transposed.shape}, dtype: {transposed.dtype}")

        # Apply low-pass filter
        filtered = low_pass_filter(transposed, max_pitch)
        filtered = ensure_array(filtered).astype(np.float32)
        logging.info(f"Filtered shape: {filtered.shape}, dtype: {filtered.dtype}")

        return filtered
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return input_data

def main(bypass_processing, record_output, stretch_factor, min_pitch, max_pitch):
    p = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    logging.info(f"Number of devices: {numdevices}")

    logging.info("Available input devices:")
    for i in range(numdevices):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            logging.info(f"Device {i}: {device_info['name']}")

    mic_device_index = int(input("Select microphone device index: "))

    selected_device_info = p.get_device_info_by_index(mic_device_index)
    logging.info(f"Selected microphone: {selected_device_info['name']}")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=mic_device_index)

    print("* recording")
    print("Press Ctrl+C to stop the recording")

    # Prepare for recording if enabled
    if record_output:
        recorded_frames = []

    try:
        while True:
            input_data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
            
            logging.info(f"Processing audio from {selected_device_info['name']}")

            if bypass_processing:
                output_data = input_data
            else:
                output_data = process_audio(input_data, stretch_factor, min_pitch, max_pitch)
            
            stream.write(output_data.astype(np.float32).tobytes())
            
            if record_output:
                recorded_frames.append(output_data)

    except KeyboardInterrupt:
        print("* done recording")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recording if enabled
    if record_output and recorded_frames:
        output_filename = "output_recording.wav"
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join([frame.astype(np.float32).tobytes() for frame in recorded_frames]))
        wf.close()
        print(f"Recording saved as {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio processing script with loudest segment selection")
    parser.add_argument("--bypass", action="store_true", help="Bypass all processing")
    parser.add_argument("--record", action="store_true", help="Record the output audio")
    parser.add_argument("--stretch", type=float, default=2.0, help="Time stretch factor (default: 2.0)")
    parser.add_argument("--min_pitch", type=float, default=default_min_pitch, help=f"Minimum output pitch in Hz (default: {default_min_pitch} Hz)")
    parser.add_argument("--max_pitch", type=float, default=default_max_pitch, help=f"Maximum output pitch in Hz (default: {default_max_pitch} Hz)")
    args = parser.parse_args()

    main(args.bypass, args.record, args.stretch, args.min_pitch, args.max_pitch)
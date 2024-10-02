import pyaudio
import numpy as np
from scipy import signal
import logging
import argparse
import wave

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
BASE_RATE = 44100  # Standard sample rate

logging.basicConfig(level=logging.INFO)

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

def calculate_rate(stretch_factor):
    """Calculate the needed rate based on the stretch factor."""
    return int(BASE_RATE * stretch_factor)

def change_pitch(data, input_rate, output_rate):
    """Change pitch by resampling the audio data."""
    return signal.resample(data, int(len(data) * output_rate / input_rate))

def apply_low_pass_filter(data, cutoff_freq, sample_rate):
    """Apply a low-pass filter to the audio data."""
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(6, normal_cutoff, btype='low', analog=False)
    return signal.lfilter(b, a, data)

def extract_envelope(data, smoothing=0.995):
    """Extract the envelope of the input data."""
    return np.abs(signal.hilbert(data))

def enhance_bass(data, boost_factor=2.0, boost_freq=100, sample_rate=44100):
    """Enhance bass frequencies."""
    nyquist = 0.5 * sample_rate
    normal_freq = boost_freq / nyquist
    b, a = signal.butter(2, normal_freq, btype='lowpass')
    bass = signal.lfilter(b, a, data)
    return data + (bass * (boost_factor - 1))

def add_distortion(data, amount=0.1):
    """Add subtle distortion to the audio."""
    return np.tanh(amount * data) / np.tanh(amount)

def process_audio(input_data, input_rate, output_rate, cutoff_freq):
    try:
        # Ensure input is an array
        input_data = ensure_array(input_data)

        # Extract the envelope of the input data
        input_envelope = extract_envelope(input_data)

        # Change pitch by resampling
        pitched = change_pitch(input_data, input_rate, output_rate)
        pitched = ensure_array(pitched)

        # Apply low-pass filter
        filtered = apply_low_pass_filter(pitched, cutoff_freq, output_rate)

        # Enhance bass
        bass_boosted = enhance_bass(filtered, boost_factor=1.5, boost_freq=40, sample_rate=output_rate)

        # Apply the input envelope to the processed audio
        result = bass_boosted * (input_envelope / np.max(input_envelope))

        return result.astype(np.float32)
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return input_data

def main(bypass_processing, record_output, stretch_factor, cutoff_freq):
    output_rate = calculate_rate(stretch_factor)
    logging.info(f"Using output rate: {output_rate} Hz (stretch factor: {stretch_factor})")
    logging.info(f"Low-pass filter cutoff frequency: {cutoff_freq} Hz")

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
                    rate=BASE_RATE,
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
                output_data = process_audio(input_data, BASE_RATE, output_rate, cutoff_freq)
            
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
        wf.setframerate(output_rate)
        wf.writeframes(b''.join([frame.astype(np.float32).tobytes() for frame in recorded_frames]))
        wf.close()
        print(f"Recording saved as {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio processing script with input-matched envelope")
    parser.add_argument("--bypass", action="store_true", help="Bypass all processing")
    parser.add_argument("--record", action="store_true", help="Record the output audio")
    parser.add_argument("--stretch", type=float, default=0.5, help="Stretch factor for pitch change (default: 0.5)")
    parser.add_argument("--cutoff", type=float, default=200, help="Cutoff frequency for low-pass filter in Hz (default: 200)")
    args = parser.parse_args()

    main(args.bypass, args.record, args.stretch, args.cutoff)
import pyaudio
import numpy as np
from scipy import signal
import logging
import argparse
import wave

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

logging.basicConfig(level=logging.INFO)

def time_stretch(data, stretch_factor):
    """Time stretch the audio data."""
    return signal.resample(data, int(len(data) * stretch_factor))

def detect_envelope(data, frame_length=128, hop_length=32, threshold=0.1):
    """Detect the amplitude envelope of the audio data, focusing on significant volume increases."""
    if len(data) < frame_length:
        return np.ones_like(data)
    
    frames = np.array([np.sqrt(np.mean(data[i:i+frame_length]**2)) 
                       for i in range(0, len(data)-frame_length, hop_length)])
    frames = frames / np.max(frames)
    frames[frames < threshold] = 0
    envelope = np.interp(np.linspace(0, len(frames), len(data)), np.arange(len(frames)), frames)
    
    return envelope

def apply_envelope(data, envelope):
    """Apply the detected envelope to the audio data."""
    return data * envelope

def normalize_volume(data, target_dB=-10):
    """Normalize the volume of the audio data to a target dB level."""
    rms = np.sqrt(np.mean(data**2))
    if rms > 0:
        current_dB = 20 * np.log10(rms)
        gain = 10**((target_dB - current_dB) / 20)
        return data * gain
    return data

def spectral_subtraction(input_data, output_data, alpha=2, beta=0.1):
    """Perform spectral subtraction for noise reduction."""
    # Compute FFT of input and output
    input_fft = np.fft.rfft(input_data)
    output_fft = np.fft.rfft(output_data)
    
    # Estimate noise spectrum
    noise_spectrum = np.abs(input_fft)
    
    # Compute power spectrum of output
    output_power = np.abs(output_fft) ** 2
    
    # Subtract scaled noise spectrum from output power spectrum
    clean_power = np.maximum(output_power - alpha * noise_spectrum ** 2, beta * output_power)
    
    # Compute new magnitude spectrum
    clean_mag = np.sqrt(clean_power)
    
    # Apply new magnitude to output phase
    clean_fft = clean_mag * np.exp(1j * np.angle(output_fft))
    
    # Inverse FFT to get clean signal
    clean_signal = np.fft.irfft(clean_fft)
    
    return clean_signal

def process_audio(input_data, stretch_factor, target_dB, envelope_threshold):
    # Detect original envelope
    original_envelope = detect_envelope(input_data, threshold=envelope_threshold)
    
    # Time stretch
    stretched = time_stretch(input_data, stretch_factor)
    
    # Resize to original length
    if len(stretched) < CHUNK:
        stretched = np.pad(stretched, (0, CHUNK - len(stretched)))
    else:
        stretched = stretched[:CHUNK]
    
    # Apply original envelope to stretched audio
    output = apply_envelope(stretched, original_envelope)
    
    # Perform spectral subtraction for noise reduction
    output = spectral_subtraction(input_data, output)
    
    # Normalize volume
    output = normalize_volume(output, target_dB)
    
    return output

def main(bypass_processing, record_output, stretch_factor, target_dB, envelope_threshold):
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
                output_data = process_audio(input_data, stretch_factor, target_dB, envelope_threshold)
            
            stream.write(output_data.astype(np.float32).tobytes())
            
            if record_output:
                recorded_frames.append(output_data)

    except KeyboardInterrupt:
        print("* done recording")

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
    parser = argparse.ArgumentParser(description="Audio processing script with noise reduction")
    parser.add_argument("--bypass", action="store_true", help="Bypass all processing")
    parser.add_argument("--record", action="store_true", help="Record the output audio")
    parser.add_argument("--stretch", type=float, default=2.0, help="Time stretch factor (default: 2.0)")
    parser.add_argument("--target_db", type=float, default=-10, help="Target dB level for normalization (default: -10)")
    parser.add_argument("--envelope_threshold", type=float, default=0.1, help="Threshold for envelope detection (default: 0.1)")
    args = parser.parse_args()

    main(args.bypass, args.record, args.stretch, args.target_db, args.envelope_threshold)
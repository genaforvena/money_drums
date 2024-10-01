import pyaudio
import numpy as np
from scipy import signal
import logging
import argparse
import wave
import threading
import time

CHUNK = 2048
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

logging.basicConfig(level=logging.INFO)

# Global variables for runtime control
feedback_threshold = 0.7
notch_frequency = 1000
notch_q = 30
echo_decay = 0.5

def time_stretch(data, stretch_factor):
    """Time stretch the audio data."""
    return signal.resample(data, int(len(data) * stretch_factor))

def onset_detection(data, threshold=0.1):
    """Detect onsets in the audio data."""
    if len(data) < 2:
        return []

    spectrum = np.abs(np.fft.rfft(data))
    flux = np.sum(np.diff(spectrum, axis=0), axis=-1)

    if len(flux) == 0:
        return []

    max_flux = np.max(flux)
    if max_flux > 0:
        flux = flux / max_flux
    else:
        return []

    detection_function = (flux > threshold).astype(int)

    peaks = []
    for i in range(1, len(detection_function) - 1):
        if detection_function[i] == 1:
            if detection_function[i] >= detection_function[i - 1] and detection_function[i] >= detection_function[i + 1]:
                peaks.append(i)

    return peaks

def bandpass_filter(data, lowcut=100, highcut=8000, order=6):
    """Apply a bandpass filter to the audio data."""
    nyq = 0.5 * RATE
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.lfilter(b, a, data)

def transient_shaper(data, attack=5, sustain=50):
    """Enhance transients in the audio data."""
    envelope = np.abs(signal.hilbert(data))
    attack_env = signal.lfilter([1], [1, -np.exp(-1/(RATE * attack/1000))], envelope)
    sustain_env = signal.lfilter([1], [1, -np.exp(-1/(RATE * sustain/1000))], envelope)
    transient = np.maximum(0, attack_env - sustain_env)
    return data + transient * data

def noise_gate(data, threshold=-30, attack=5, release=50):
    """Apply a noise gate to the audio data."""
    db = 20 * np.log10(np.abs(data) + 1e-6)
    mask = np.zeros_like(data)
    mask[db > threshold] = 1
    
    attack_coef = np.exp(-1/(RATE * attack/1000))
    release_coef = np.exp(-1/(RATE * release/1000))
    for i in range(1, len(mask)):
        if mask[i] > mask[i-1]:
            mask[i] = attack_coef * mask[i-1] + (1-attack_coef) * mask[i]
        else:
            mask[i] = release_coef * mask[i-1] + (1-release_coef) * mask[i]
    
    return data * mask

def detect_feedback(data, threshold):
    """Detect potential feedback in the audio data."""
    rms = np.sqrt(np.mean(data**2))
    return rms > threshold

def adaptive_notch_filter(data, freq, q):
    """Apply an adaptive notch filter to reduce a specific frequency."""
    nyq = 0.5 * RATE
    freq_normalized = freq / nyq
    b, a = signal.iirnotch(freq_normalized, q)
    filtered_data = signal.lfilter(b, a, data)
    
    # Adapt the notch frequency based on the energy in the filtered signal
    fft = np.fft.fft(filtered_data)
    freqs = np.fft.fftfreq(len(fft), 1/RATE)
    max_freq = freqs[np.argmax(np.abs(fft))]
    
    return filtered_data, max_freq

def limiter(data, threshold=0.95):
    """Apply a simple limiter to prevent volume spikes."""
    return np.clip(data, -threshold, threshold)

def echo_cancellation(input_data, output_data, decay):
    """Apply echo cancellation."""
    return input_data - decay * output_data

def feedback_suppressor(data, threshold=0.9):
    """Apply a feedback suppressor."""
    fft = np.fft.rfft(data)
    magnitude = np.abs(fft)
    phase = np.angle(fft)
    
    # Reduce magnitude of frequency components above threshold
    magnitude[magnitude > threshold] *= 0.5
    
    suppressed = np.fft.irfft(magnitude * np.exp(1j * phase))
    return suppressed

last_output = np.zeros(CHUNK)  # Global variable to store last output

def process_audio(input_data, onset_threshold, bandpass_low, bandpass_high, stretch_factor):
    global last_output, feedback_threshold, notch_frequency, notch_q, echo_decay
    
    try:
        # Echo cancellation
        echo_cancelled = echo_cancellation(input_data, last_output, echo_decay)
        
        # Feedback suppression
        suppressed = feedback_suppressor(echo_cancelled)
        
        # Detect feedback
        if detect_feedback(suppressed, feedback_threshold):
            logging.warning("Feedback detected!")
            # Apply adaptive notch filter
            suppressed, peak_freq = adaptive_notch_filter(suppressed, notch_frequency, notch_q)
            notch_frequency = peak_freq  # Update notch frequency
        
        # Time stretch
        stretched = time_stretch(suppressed, stretch_factor)
        
        # Resize to original length
        if len(stretched) < CHUNK:
            stretched = np.pad(stretched, (0, CHUNK - len(stretched)))
        else:
            stretched = stretched[:CHUNK]
        
        # Detect onsets
        onsets = onset_detection(stretched, threshold=onset_threshold)
        
        # Apply bandpass filter
        filtered = bandpass_filter(stretched, lowcut=bandpass_low, highcut=bandpass_high)
        
        # Enhance transients
        enhanced = transient_shaper(filtered)
        
        # Apply noise gate
        gated = noise_gate(enhanced)
        
        # Apply limiter
        limited = limiter(gated)
        
        last_output = limited  # Store this output for next echo cancellation
        return limited
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return input_data  # Return original data if processing fails

def user_input_thread():
    global feedback_threshold, notch_frequency, echo_decay
    while True:
        command = input("Enter command (f: feedback, n: notch, e: echo, q: quit): ")
        if command == 'f':
            feedback_threshold = float(input("Enter new feedback threshold (0-1): "))
        elif command == 'n':
            notch_frequency = float(input("Enter new notch frequency (Hz): "))
        elif command == 'e':
            echo_decay = float(input("Enter new echo decay (0-1): "))
        elif command == 'q':
            print("Quitting...")
            break

def main(bypass_processing, record_output, onset_threshold, bandpass_low, bandpass_high, stretch_factor):
    global feedback_threshold, notch_frequency, notch_q, echo_decay
    
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
    print("Enter 'f' to adjust feedback threshold, 'n' to adjust notch frequency, 'e' to adjust echo decay, or 'q' to quit")

    # Start user input thread
    input_thread = threading.Thread(target=user_input_thread)
    input_thread.start()

    # Prepare for recording if enabled
    if record_output:
        recorded_frames = []

    try:
        while input_thread.is_alive():
            input_data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
            
            logging.info(f"Processing audio from {selected_device_info['name']}")

            if bypass_processing:
                output_data = input_data
            else:
                output_data = process_audio(input_data, onset_threshold, bandpass_low, bandpass_high, stretch_factor)
            
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
    parser = argparse.ArgumentParser(description="Audio processing script with enhanced feedback prevention")
    parser.add_argument("--bypass", action="store_true", help="Bypass all processing")
    parser.add_argument("--record", action="store_true", help="Record the output audio")
    parser.add_argument("--onset_threshold", type=float, default=0.1, help="Threshold for onset detection (default: 0.1)")
    parser.add_argument("--bandpass_low", type=float, default=100, help="Lower frequency for bandpass filter (default: 100 Hz)")
    parser.add_argument("--bandpass_high", type=float, default=8000, help="Higher frequency for bandpass filter (default: 8000 Hz)")
    parser.add_argument("--stretch", type=float, default=2.0, help="Time stretch factor (default: 2.0)")
    args = parser.parse_args()

    main(args.bypass, args.record, args.onset_threshold, args.bandpass_low, args.bandpass_high, args.stretch)
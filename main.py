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

def process_audio(data, stretch_factor):
    # Time stretch
    stretched = time_stretch(data, stretch_factor)
    
    # Resize to original length
    if len(stretched) < CHUNK:
        stretched = np.pad(stretched, (0, CHUNK - len(stretched)))
    else:
        stretched = stretched[:CHUNK]
    
    return stretched

def main(bypass_processing, record_output, stretch_factor):
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
                output_data = process_audio(input_data, stretch_factor)
            
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
    parser = argparse.ArgumentParser(description="Audio processing script with time stretching")
    parser.add_argument("--bypass", action="store_true", help="Bypass all processing")
    parser.add_argument("--record", action="store_true", help="Record the output audio")
    parser.add_argument("--stretch", type=float, default=10.0, help="Time stretch factor (default: 10.0)")
    args = parser.parse_args()

    main(args.bypass, args.record, args.stretch)
import pyaudio
import numpy as np
import logging
import argparse

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

logging.basicConfig(level=logging.INFO)

def simple_pitch_shift(data, pitch_factor):
    """
    Simple pitch shifting using resampling and interpolation.
    """
    indices = np.arange(0, len(data), pitch_factor)
    return np.interp(indices, np.arange(len(data)), data)

def process_audio(input_data, pitch_factor):
    try:
        # Convert input to numpy array
        audio = np.frombuffer(input_data, dtype=np.float32)
        
        # Apply pitch shift
        pitched = simple_pitch_shift(audio, pitch_factor)
        
        # Ensure output is the same length as input
        if len(pitched) > CHUNK:
            pitched = pitched[:CHUNK]
        elif len(pitched) < CHUNK:
            pitched = np.pad(pitched, (0, CHUNK - len(pitched)), 'constant')
        
        return pitched.astype(np.float32)
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return input_data

def main(pitch_factor):
    p = pyaudio.PyAudio()

    # List available input devices
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(f"Input Device id {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}")

    # Get user input for device selection
    device_id = int(input("Enter input device ID: "))

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_id)

    print("* recording")

    try:
        while True:
            input_data = stream.read(CHUNK)
            output_data = process_audio(input_data, pitch_factor)
            stream.write(output_data.tobytes())
    except KeyboardInterrupt:
        print("* done recording")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple pitch shift audio processor")
    parser.add_argument("--pitch", type=float, default=0.5, help="Pitch factor (default: 0.5, lower values = lower pitch)")
    args = parser.parse_args()

    main(args.pitch)
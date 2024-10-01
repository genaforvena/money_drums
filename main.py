import pyaudio
import numpy as np
from scipy import signal

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

print("* recording")

def process_audio(data, lowcut, highcut):
    # Apply bandpass filter
    nyq = 0.5 * RATE
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(6, [low, high], btype='band')
    filtered = signal.lfilter(b, a, data)
    
    # Apply envelope
    envelope = np.exp(-np.linspace(0, 5, len(filtered)))
    shaped = filtered * envelope
    
    return shaped

try:
    while True:
        input_data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
        
        # Process the audio to shape it into a kick drum sound
        processed_data = process_audio(input_data, lowcut=50, highcut=150)
        
        # Amplify the output
        processed_data *= 5  # Adjust this value to change the output volume
        
        stream.write(processed_data.astype(np.float32).tobytes())

except KeyboardInterrupt:
    print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()
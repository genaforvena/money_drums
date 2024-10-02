import pyaudio
import numpy as np
import pyrubberband as pyrb
import scipy.signal as signal
import logging
import argparse
import time

logging.basicConfig(level=logging.INFO)

def process_audio(input_data, semitones, input_rate, output_rate):
    try:
        audio = np.frombuffer(input_data, dtype=np.float32)
        
        # Downsample
        if input_rate != output_rate:
            audio = signal.resample(audio, int(len(audio) * output_rate / input_rate))
        
        # Apply anti-aliasing filter
        nyquist = output_rate / 2
        cutoff = min(nyquist * 0.9, 20000)  # Prevent cutting off all audible frequencies
        b, a = signal.butter(5, cutoff, fs=output_rate, btype='low')
        audio = signal.lfilter(b, a, audio)
        
        # Pitch shift
        pitched = pyrb.pitch_shift(audio, output_rate, n_steps=semitones,
                                   rbargs={'-F':''})       
        # Upsample back to input rate if necessary
        if input_rate != output_rate:
            pitched = signal.resample(pitched, len(input_data) // 4)  # Divide by 4 because input is byte stream
        
        return pitched.astype(np.float32)
    except Exception as e:
        logging.error(f"Error in process_audio: {str(e)}")
        return input_data

def main(semitones, chunk_size, input_rate, output_rate):
    p = pyaudio.PyAudio()

    # List available input devices
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(f"Input Device id {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}")

    device_id = int(input("Enter input device ID: "))

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=input_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk_size,
                    input_device_index=device_id)

    print(f"* Recording with CHUNK size: {chunk_size}")
    print(f"* Input sample rate: {input_rate} Hz, Processing sample rate: {output_rate} Hz")
    print(f"* Pitch shift: {semitones} semitones")
    print("Press Ctrl+C to stop the recording")

    try:
        latencies = []
        while True:
            start_time = time.time()
            
            input_data = stream.read(chunk_size)
            output_data = process_audio(input_data, semitones, input_rate, output_rate)
            stream.write(output_data)
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            # Print average latency every 100 iterations
            if len(latencies) % 100 == 0:
                avg_latency = sum(latencies) / len(latencies)
                print(f"Average latency: {avg_latency:.2f} ms")
                latencies = []  # Reset the list

    except KeyboardInterrupt:
        print("* Done recording")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low sample rate pitch shift audio processor")
    parser.add_argument("--pitch", type=float, default=-12, help="Pitch shift in semitones (default: -12)")
    parser.add_argument("--chunk", type=int, default=1024, help="CHUNK size (default: 1024)")
    parser.add_argument("--input-rate", type=int, default=44100, help="Input sample rate in Hz (default: 44100)")
    parser.add_argument("--output-rate", type=int, default=1000, help="Processing sample rate in Hz (default: 1000)")
    args = parser.parse_args()

    main(args.pitch, args.chunk, args.input_rate, args.output_rate)
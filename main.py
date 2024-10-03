import pyaudio
import numpy as np
import pyrubberband as pyrb
import scipy.signal as signal
import logging
import argparse
import time

logging.basicConfig(level=logging.INFO)

def simple_pitch_shift(data, pitch_factor):
    """
    Simple pitch shifting using resampling and interpolation.
    """
    indices = np.arange(0, len(data), pitch_factor)
    return np.interp(indices, np.arange(len(data)), data)

def simple_slow_down(data, slow_factor):
    """
    Simple audio slowing by repeating samples.
    """
    return np.repeat(data, slow_factor)

def process_audio(input_data, pitch_factor, slow_factor=None):
        # Apply slowing if slow_factor is provided
        if slow_factor is not None:
            pitched = simple_slow_down(pitched, slow_factor)
        
        # Ensure output is the same length as input
        if len(pitched) > CHUNK:
            pitched = pitched[:CHUNK]
        elif len(pitched) < CHUNK:
            pitched = np.pad(pitched, (0, CHUNK - len(pitched)), 'constant')
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

def main(pitch_factor, slow_factor):
    input_data = stream.read(CHUNK)
    output_data = process_audio(input_data, pitch_factor, slow_factor)
    stream.write(output_data.tobytes())
    except KeyboardInterrupt:
        print("* Done recording")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple pitch shift and slow down audio processor")
    parser.add_argument("--pitch", type=float, default=0.5, help="Pitch factor (default: 0.5, lower values = lower pitch)")
    parser.add_argument("--slow", type=int, help="Slow factor (optional, integer value for sample repetition)")
    args = parser.parse_args()

    main(args.pitch, args.slow)

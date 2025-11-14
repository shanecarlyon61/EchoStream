# EchoStream Python Project

Python implementation of the EchoStream audio streaming system.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The application reads configuration from `/home/will/.an/config.json`. This file should contain:
- Channel IDs
- Tone detection settings
- Passthrough configuration

## Usage

Run the main application:
```bash
python main.py
```

## Features

- Multi-channel audio streaming
- GPIO monitoring (using lgpio library)
- WebSocket communication
- UDP audio transmission
- Opus audio encoding/decoding
- Tone detection
- MQTT publishing
- S3 audio upload

## GPIO Pins

- GPIO 20 (Physical pin 38) - Channel 1
- GPIO 21 (Physical pin 40) - Channel 2
- GPIO 23 (Physical pin 16) - Channel 3
- GPIO 24 (Physical pin 18) - Channel 4

## Notes

- Uses lgpio library for GPIO control (instead of gpiod in C version)
- Requires Python 3.7+
- Some features may require additional system libraries (ALSA, PortAudio)


# Vehiclebot

This is an ML-powered Vehicle detection framework designed to capture vehicles detected in a live video feed, and store the results for further processing.

## Installing with Docker

1. Clone the repository to a directory and run:

```bash
docker build -t vehicle-entry-exit .
```

2. Run the image

```bash
docker run -p 8080:8080 vehicle-entry-exit
```

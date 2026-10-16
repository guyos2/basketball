# Web interface in a container: works on Hugging Face Spaces (Docker), Google Cloud Run, Render, etc.
FROM python:3.12-slim

# OpenCV (pulled in by ultralytics) needs these system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small. For a GPU machine use e.g. --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user torch torchvision --index-url ${TORCH_INDEX}
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the detection models into the image so the first analysis does not wait for a download.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11s.pt')"

COPY --chown=user courtvision courtvision

ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "python -m courtvision serve --host 0.0.0.0 --port ${PORT} --workspace /home/user/app/workspace"]

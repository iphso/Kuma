FROM python:3.11-slim

WORKDIR /workspace

# CPU-only torch keeps the image small; swap index URL for CUDA builds.
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    numpy \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
COPY src/ src/
# --no-deps skips torch/numpy (already installed above); pytest comes from [dev]
RUN pip install --no-cache-dir pytest onnx && \
    pip install --no-cache-dir -e . --no-deps

COPY examples/ examples/
COPY tests/ tests/

VOLUME ["/workspace/artifacts"]

ENTRYPOINT ["python", "-m", "examples.export_simple"]

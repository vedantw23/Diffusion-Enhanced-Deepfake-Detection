FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
# Use CPU-only PyTorch. The default Linux package can pull multi-gigabyte CUDA
# dependencies, which are not usable on the free Render CPU service.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=10000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]

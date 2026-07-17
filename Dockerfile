FROM python:3.12-slim

WORKDIR /srv/xray-vision

# CPU-only torch keeps the image small; everything else from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model
COPY docs/example_input.png ./docs/example_input.png

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

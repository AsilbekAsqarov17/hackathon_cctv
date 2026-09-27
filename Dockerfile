FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The application and model weights are copied into the image. Inputs and output
# are intentionally mounted at runtime so predictions survive container exit.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY solution.py run_submission.py evaluate.py ./
COPY src ./src
COPY configs ./configs
COPY weights ./weights

RUN mkdir -p /data/in /data/out

# Override the command when processing a different input/output location, e.g.
#   docker run --rm -v "$PWD/in:/data/in:ro" -v "$PWD/out:/data/out" image
CMD ["python", "run_submission.py", "--videos", "/data/in", "--out", "/data/out/predictions.json", "--no-risk"]
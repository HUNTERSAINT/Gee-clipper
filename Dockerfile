FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp uses Deno to solve YouTube's JavaScript challenges
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
CMD ["python", "bot.py"]

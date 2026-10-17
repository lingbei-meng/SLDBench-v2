FROM sldbench-runtime:20261016-three-harnesses

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux asciinema \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /app/data

COPY data/train.parquet /app/data/train.parquet


FROM python:3.11-slim-bookworm

# ── System bioinformatics tools ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        bedtools \
        bcftools \
        tabix \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── UCSC bigWigAverageOverBed ─────────────────────────────────────────────────
# Detects linux/amd64 (x86_64) or linux/arm64 (aarch64) automatically.
RUN ARCH=$(uname -m) && \
    wget -q -O /usr/local/bin/bigWigAverageOverBed \
        "https://hgdownload.soe.ucsc.edu/admin/exe/linux.${ARCH}/bigWigAverageOverBed" && \
    chmod +x /usr/local/bin/bigWigAverageOverBed

# ── uv ────────────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ── MuTopia ───────────────────────────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/mutopia
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install --no-cache \
    "git+https://github.com/AllenWLynch/Mutopia.git@main"

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="MuTopia" \
      org.opencontainers.image.description="Mutational Topography Modeling Toolkit" \
      org.opencontainers.image.source="https://github.com/AllenWLynch/Mutopia"

WORKDIR /workspace
CMD ["bash"]

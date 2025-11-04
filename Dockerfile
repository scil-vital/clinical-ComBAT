FROM ubuntu:noble-20251001

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    git \
    wget \
    && apt-get clean

# Create venv and install dependencies
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install python 3.12 via uv
WORKDIR /
RUN uv python install 3.12

# Create env
RUN uv venv --python 3.12 /.venvs/combat
ENV PATH="/.venvs/combat/bin:$PATH"
RUN uv cache clean
RUN uv pip install --upgrade pip setuptools==75.1.0 wheel setuptools_scm kiwisolver fonttools

# Install combat
RUN git clone https://github.com/arnaudbore/clinical-ComBAT.git
WORKDIR /clinical-ComBAT
RUN git checkout update_organization_and_add_docker
RUN uv pip install -e . --no-build-isolation

WORKDIR /
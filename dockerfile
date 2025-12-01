# Use a Python 3.5-slim image as the base
FROM python:3.5-slim

# Set environment variable for libfreenect2 installation prefix
ENV LIBFREENECT2_INSTALL_PREFIX=/usr/local

# Fix: Adjust sources.list to point to the Debian archive since 'buster' is old
# And install required C++ and development dependencies
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i '/stretch-updates/d' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    libturbojpeg0-dev \
    libglfw3-dev \
    pkg-config \
    udev \
    python3-dev \
    python3-numpy \
    # We install cython with pip later, but keep udev for device access \
    && rm -rf /var/lib/apt/lists/*

# Clone and build libfreenect2 (C++ core library)
WORKDIR /tmp/libfreenect2
RUN git clone https://github.com/OpenKinect/libfreenect2.git . && \
    mkdir build && \
    cd build && \
    # Configure and build the core C++ library
    cmake .. -DCMAKE_INSTALL_PREFIX=$LIBFREENECT2_INSTALL_PREFIX -DBUILD_OPENNI2_DRIVER=OFF -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && \
    make install

# Update dynamic linker run-time bindings so the system finds the newly installed libfreenect2 shared libraries
RUN ldconfig

# Install pylibfreenect2 (the Python wrapper)
WORKDIR /tmp
# Add this section to install necessary build dependencies for NumPy/OpenCV
# Add this section to install necessary build dependencies for NumPy/OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libtool \
    pkg-config \
    # The xlocale.h file is fixed by libc6-dev
    libc6-dev \
    # Clean up APT cache to keep the image small
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Fix: Install cython via pip before installing pylibfreenect2, and update pip
RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install numpy cython opencv-python && \
    pip install git+https://github.com/r9y9/pylibfreenect2.git

# Set the working directory for your application
WORKDIR /app

# The command to run your Python script (customize this)
# CMD ["python3", "your_kinect_script.py"]
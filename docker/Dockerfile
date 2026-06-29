FROM nvidia/cuda:12.4.1-devel-ubuntu20.04 AS runtime

ARG L4CASADI_REV=9fe2894533e05009bcbfa7706966745c5236fa4c

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git gnupg2 lsb-release \
 && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
      > /etc/apt/sources.list.d/ros1.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ninja-build python3-dev python3-pip python3-opencv \
      libgl1 libglib2.0-0 \
      ros-noetic-ros-base \
      ros-noetic-cv-bridge ros-noetic-image-geometry \
      ros-noetic-image-transport ros-noetic-compressed-image-transport \
      ros-noetic-message-filters ros-noetic-message-generation \
      ros-noetic-nav-msgs ros-noetic-tf ros-noetic-tf2-ros \
      ros-noetic-vision-msgs ros-noetic-visualization-msgs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/requirements-linux.txt /tmp/requirements-linux.txt
RUN python3 -m pip install --no-cache-dir --upgrade \
      pip==24.3.1 setuptools==75.3.2 wheel==0.45.1 \
 && python3 -m pip install --no-cache-dir \
      torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
      --index-url https://download.pytorch.org/whl/cu124 \
 && python3 -m pip install --no-cache-dir -r /tmp/requirements-linux.txt

# L4CasADi must see the already-installed PyTorch package during compilation.
RUN python3 -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/Tim-Salzmann/l4casadi.git@${L4CASADI_REV}"

COPY src /workspace/src
COPY docker /workspace/docker

RUN bash -lc 'source /opt/ros/noetic/setup.bash && cd /workspace && catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5'

RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --create-home app \
 && mkdir -p /runs/ros/log \
 && chown -R app:app /workspace /runs

USER app
ENV HOME=/home/app \
    ROS_MASTER_URI=http://127.0.0.1:11311 \
    ROS_IP=127.0.0.1 \
    PYTHONUNBUFFERED=1

EXPOSE 10000
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]

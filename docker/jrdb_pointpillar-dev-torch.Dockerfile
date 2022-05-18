FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3
RUN apt-get update \
    && apt-get install -y curl \
        net-tools \
        lsb-release \
        locales \
        ssh \
        python3-empy \
        apt-utils \
        gnupg2 \
    && apt-get upgrade -y

ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# install ROS with the base, robot, and perception variant as per REP-142
ENV ROS ros-noetic
RUN apt-get update && apt-get install -y $ROS-ros-base
RUN apt-get install -y $ROS-geometry $ROS-control-msgs \
    $ROS-diagnostics $ROS-executive-smach \
    $ROS-filters $ROS-geometry \
    $ROS-robot-state-publisher
# break if errors occur
RUN apt-get install -y $ROS-xacro $ROS-image-common \
    $ROS-image-pipeline $ROS-image-transport-plugins \
    $ROS-laser-pipeline $ROS-perception-pcl \
    $ROS-vision-opencv $ROS-angles
    
# python3 dependencies for ROS
RUN apt-get install -y python3-rosdep python3-rosinstall \
    python3-rosinstall-generator python3-wstool \
    build-essential

RUN rosdep init \
    && rosdep fix-permissions \
    && rosdep update

ENV WS=/workspace/catkin_ws
RUN ["/bin/bash","-c","mkdir -p $WS/src; \
    source /opt/ros/noetic/setup.bash; \
    cd $WS/src; \
    catkin_init_workspace; \
    cd $WS; \
    catkin_make"]

ENV EP=/workspace/ros_entrypoint.sh
RUN touch $EP \
    && chmod 777 $EP \
    && echo -e '#!/bin/bash \nset -e \nsource /opt/ros/noetic/setup.bash \nsource /workspace/catkin_ws/devel/setup.bash'>> $EP

RUN apt install -y $ROS-realsense2-camera $ROS-realsense2-description

RUN sudo apt install -y usbutils lsb-release curl zip unzip tar libcanberra-gtk-module libcanberra-gtk3-module autoconf libudev-dev
RUN pip3 install pyrealsense2 empy catkin_pkg opencv-python rospkg testresources

# install Open3D and ROS helper package
RUN pip3 install open3d open3d-python open3d-ros-helper

RUN source /workspace/ros_entrypoint.sh \
    && cd /workspace/catkin_ws \
    && catkin_make 

RUN sudo apt install -y ros-noetic-ros-numpy
RUN pip3 install pyyaml

RUN pip3 install tensorflow==2.5.3
RUN cd /workspace \
    && git clone --depth=1 https://github.com/y-lai/Open3D-ML.git \
    && cd Open3D-ML \
    && pip3 install --default-timeout=100 -r requirements-torch-cuda.txt

RUN cd /workspace/Open3D-ML \
    && pip3 install -e .

RUN pip install numpy-ros

WORKDIR /workspace

# Ubuntu 24.04 with ROS2 jazzys

# ros
```bash
wget http://fishros.com/install -O fishros && . fishros

```

# mavros
```bash
sudo apt-get install ros-jazzy-mavros*
# 或者
sudo apt-get install -y ros-jazzy-mavros ros-jazzy-mavros-extras
#..
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash ./install_geographiclib_datasets.sh

# Ceres Solver
```bash
sudo apt-get install -y libceres-dev
sudo apt-get install -y cmake libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev
wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar zxf ceres-solver-2.1.0.tar.gz
cd ceres-solver-2.1.0
mkdir build && cd build
cmake -DSUITESPARSE=OFF -DCXSPARSE=OFF ..
make -j4
sudo make install
```

#  RealSense SDK
```bash
sudo apt-get install git
download from: https://github.com/IntelRealSense/librealsense/releases/tag/v2.56.1
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=release
make -j4
sudo make install
(CLOSE IMU TOPIC)
```

# Vicon
```bash
sudo apt-get install ros-jazzy-vrpn-mocap
```

# 其他
```bash
sudo apt-get install -y screen ssh
sudo apt-get update
sudo apt-get install -y python3-colcon-common-extensions python3-rosdep python3-vcstool build-essential
sudo rosdep init || true
rosdep update

```

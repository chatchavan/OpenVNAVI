# OpenVNAVI
**OpenVNAVI** (Vibrotactile Navigation Aid for the Visually Impaired)

**Bachelor Thesis at the Media Computing Group, Computer Science Department @ RWTH Aachen University (Germany)**
Author: David Antón Sánchez
https://hci.rwth-aachen.de/sanchez

## System Description
OpenVNAVI is a vest equipped with a depth sensor and array of vibration motor units that allow people with visual impairment to avoid obstacles in the environment.

The ASUS Xtion PRO LIVE depth sensor, positioned onto the user’s chest scans the environment as the user moves. From the video feed of the depth sensor a frame is captured and then processed by the Raspberry Pi 2. Each frame is downsampled from 640x480 to 16x8 and each pixel is then mapped to a vibration motor unit forming an array positioned onto the user’s belly.

The grayscale value of each pixel on the lower resolution frame is assigned to a PWM voltage value generated by the Raspberry Pi 2 via PWM drivers that will drive each vibration motor obtaining a vibration amplitude value as a function of the proximity of an object.

With this method the vibration motor unit array is able to represent a vibratory image onto the user’s belly to help create a mental representation of the obstacles in the scene.

## Setting Up the Raspberry Pi 2

### Installing Raspbian Wheezy

First you need to install the Raspbian Wheezy operating system on the Pi. The official guides provide step-by-step instructions for Linux, Mac OS and Windows.

https://www.raspberrypi.org/documentation/installation/installing-images/

### Connecting to the Raspberry Pi 2

If you want to access the Pi using SSH you first need to find its IP address. The official guides provide step-by-step instructions for Linux, Mac OS and Windows.

https://www.raspberrypi.org/documentation/troubleshooting/hardware/networking/ip-address.md

Now you can connect to the Pi using SSH by running the following command on Linux or Mac OS, replacing `<IP>` by the Pi's IP address and using the default password `raspberry`:

```bash
ssh pi@<IP>
```
The instructions for using SSH on Windows can be found in the official documentation: https://www.raspberrypi.org/documentation/remote-access/ssh/

Having to remember an IP address can be cumbersome, for that reason we're going to assign the .local domain to be able to access the Pi using `ssh pi@raspberrypi.local`.

First run the following commands to make sure everything si up-to-date:

```bash
sudo apt-get update
sudo apt-get upgrade
```

Then run:

```bash
sudo apt-get install -y avahi-daemon
```

Now that we can talk to the Pi, run `raspi-config` and perform a setup: https://www.raspberrypi.org/documentation/configuration/raspi-config.md

### Configuring the GPIO

Install `python-dev` and `python-rpi.gpio`:

```bash
sudo apt-get install -y python-dev
sudo apt-get install -y python-rpi.gpio
```

### Configuring I2C

Install `python-smbus`and `i2c-tools`:

```bash
sudo apt-get install -y python-smbus
sudo apt-get install -y i2c-tools
```

Run `sudo raspi-config` and manually enable I2C on *Advanced Options > I2C*

Edit the module file by running the following line:

```bash
sudo nano /etc/modules
```
And add the following lines (save with CRL+X):

```bash
i2c-bcm2708
i2c-dev
```
Then reboot the Pi by running `sudo reboot`.

You can test I2C devices by running:

```bash
sudo i2cdetect -y 1
```

### Removing the Current Limitation

The current of each USB port is limited by software to 0.6 A. With this modification, the limitation is set to 1.2 A, allowing for the Pi to run devices that demand more current without the need for using a powered USB hub.

Run:

```bash
sudo nano /boot/config.txt
```

And add the following line:

```bash
max_usb_current=1
```



## Upgrading GCC to version 4.8

1. Allow `apt-get` to use source from `jessie`  by changing the content of `/etc/apt/sources.list`

  ```bash
  sudo nano /etc/apt/sources.list
  ```

  to:

  ```
  deb http://mirrordirector.raspbian.org/raspbian/ wheezy main contrib non-free rpi
  deb http://archive.raspbian.org/raspbian wheezy main contrib non-free rpi
  # Source repository to add
  deb-src http://archive.raspbian.org/raspbian wheezy main contrib non-free rpi
  deb http://mirrordirector.raspbian.org/raspbian/ jessie main contrib non-free rpi
  deb http://archive.raspbian.org/raspbian jessie main contrib non-free rpi
  ```
2. Add `/etc/apt/preferences`

  ```bash
  sudo nano /etc/apt/preferences
  ```

  with the following content:

  ```
  Package: *
  Pin: release n=wheezy
  Pin-Priority: 900
  Package: *
  Pin: release n=jessie
  Pin-Priority: 300
  Package: *
  Pin: release o=Raspbian
  Pin-Priority: -10
  ```

3. Update `apt-get` and install `gcc`

  ```bash
  sudo apt-get update
  sudo apt-get install -t jessie gcc-4.8 g++-4.8
  ```

4. Change links

  ```bash
  sudo update-alternatives --remove-all gcc
  sudo update-alternatives --remove-all g++
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 20
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.6 20
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 50
  ```

[Source](https://somewideopenspace.wordpress.com/2014/02/28/gcc-4-8-on-raspberry-pi-wheezy/)

## Installing Libraries and Tools

1. Developer tools

  ```bash
  sudo apt-get -y install build-essential cmake pkg-config cmake-curses-gui htop
  ```

2. Dependencies required by OpenCV

  ```bash
  sudo apt-get -y install libpng12-0 libpng12-dev libpng++-dev libpng3 \
  libpnglite-dev \
  zlib1g-dbg zlib1g zlib1g-dev \
  pngtools libtiff4-dev libtiff4 libtiffxx0c2 libtiff-tools \
  libjpeg8 libjpeg8-dev libjpeg8-dbg libjpeg-progs \
  ffmpeg libavcodec-dev libavcodec53 libavformat53 libavformat-dev \
  libgstreamer0.10-0-dbg libgstreamer0.10-0  libgstreamer0.10-dev \
  libxine1-ffmpeg  libxine-dev libxine1-bin \
  libunicap2 libunicap2-dev \
  libv4l-0 libv4l-dev \
  libdc1394-22-dev libdc1394-22 libdc1394-utils \
  libatlas-base-dev gfortran swig zlib1g-dbg zlib1g zlib1g-dev
  ```

3. Dependencies required by OpenNI

  ```bash
  sudo apt-get -y install libusb-1.0 doxygen freeglut3-dev openjdk-6-jdk graphviz
  ```

4. Python

  ```bash
  sudo apt-get -y install libpython2.7 python-dev python2.7-dev python-numpy python-pip
  ```


## Installing OpenNI

1. Clone source

  ```
  git clone https://github.com/OpenNI/OpenNI.git -b unstable
  git clone https://github.com/PrimeSense/Sensor.git -b unstable
  ```

2. Enable compiler optimization for OpenNI specific to Raspberry Pi 2

  ```bash
  sudo nano OpenNI/Platform/Linux/Build/Common/Platform.Arm
  ```
  Replace

  ```
  CFLAGS += -march=armv7-a -mtune=cortex-a8 -mfpu=neon -mfloat-abi=softfp #-mcpu=cortex-a8
  ```
  with

  ```bash
  CFLAGS += -march=native  -mfpu=neon-vfpv4 -mfloat-abi=hard
  ```

3. Enable compiler optimization for Sensor specific to Raspberry Pi 2

  ```bash
  sudo nano ~/Sensor/Platform/Linux/Build/Common/Platform.Arm
  ```

  Replace

  ```
  CFLAGS += -march=armv7-a -mtune=cortex-a8 -mfpu=neon -mfloat-abi=softfp #-mcpu=cortex-a8
  ```
  with

  ```bash
  CFLAGS += -march=native  -mfpu=neon-vfpv4 -mfloat-abi=hard
  ```

4. Install

  ```bash
  cd ~/OpenNI/Platform/Linux/CreateRedist/
  ./RedistMaker.Arm
  cd ~/OpenNI/Platform/Linux/Redist/OpenNI-Bin-Dev-Linux-Arm-v1.5.8.5   #NOTE: version number may change
  sudo ./install.sh

  cd ~/Sensor/Platform/Linux/CreateRedist/
  ./RedistMaker Arm
  cd ~/Sensor/Platform/Linux/Redist/Sensor-Bin-Linux-Arm-v5.1.6.5  #NOTE: version number may change
  sudo ./install.sh
  ```



## Installing OpenCV

1. Download OpenCV 2.4.10
```
wget https://github.com/Itseez/opencv/archive/2.4.10.tar.gz
tar -xzf 2.4.10.tar.gz
```
2. Configure

```bash
cd opencv-2.4.10
mkdir build
cd build

sudo cmake -D CMAKE_BUILD_TYPE=RELEASE \
   -D CMAKE_INSTALL_PREFIX=/usr/local \
   -D BUILD_NEW_PYTHON_SUPPORT=ON \
   -D WITH_OPENNI=ON \
   -D ENABLE_NEON=ON -D ENABLE_VFPV3=ON \
   ..
```

Make sure the configuration is as follows:

```
--     Linker flags (Release):
--     Linker flags (Debug):
--     Precompiled headers:         YES
--
--   OpenCV modules:
--     To be built:                 core flann imgproc highgui features2d calib3d ml video legacy objdetect photo gpu ocl nonfree contrib python stitching superres ts videostab
--     Disabled:                    world
--     Disabled by dependency:      -
--     Unavailable:                 androidcamera dynamicuda java viz
--
--   GUI:
--     QT:                          NO
--     GTK+ 2.x:                    NO
--     GThread :                    YES (ver 2.40.0)
--     GtkGlExt:                    NO
--     OpenGL support:              NO
--     VTK support:                 NO
--
--   Media I/O:
--     ZLib:                        /usr/lib/arm-linux-gnueabihf/libz.so (ver 1.2.7)
--     JPEG:                        /usr/lib/arm-linux-gnueabihf/libjpeg.so (ver 80)
--     PNG:                         /usr/lib/arm-linux-gnueabihf/libpng.so (ver 1.2.49)
--     TIFF:                        /usr/lib/arm-linux-gnueabihf/libtiff.so (ver 42 - 3.9.6)
--     JPEG 2000:                   build (ver 1.900.1)
--     OpenEXR:                     build (ver 1.7.1)
--
--   Video I/O:
--     DC1394 1.x:                  NO
--     DC1394 2.x:                  YES (ver 2.2.0)
--     FFMPEG:                      NO
--       codec:                     YES (ver 54.35.0)
--       format:                    YES (ver 54.20.4)
--       util:                      YES (ver 52.3.0)
--       swscale:                   NO
--       gentoo-style:              YES
--     GStreamer:                   NO
--     OpenNI:                      YES (ver 1.5.8, build 5)
--     OpenNI PrimeSensor Modules:  YES (/usr/lib/libXnCore.so)
--     PvAPI:                       NO
--     GigEVisionSDK:               NO
--     UniCap:                      NO
--     UniCap ucil:                 NO
--     V4L/V4L2:                    Using libv4l1 (ver 1.0.0) / libv4l2 (ver 1.0.0)
--     XIMEA:                       NO
--     Xine:                        NO
--
--   Other third-party libraries:
--     Use IPP:                     NO
--     Use Eigen:                   NO
--     Use TBB:                     NO
--     Use OpenMP:                  NO
--     Use GCD                      NO
--     Use Concurrency              NO
--     Use C=:                      NO
--     Use Cuda:                    NO
--     Use OpenCL:                  YES
--
--   OpenCL:
--     Version:                     dynamic
--     Include path:                /home/pi/opencv-2.4.10/3rdparty/include/opencl/1.2
--     Use AMD FFT:                 NO
--     Use AMD BLAS:                NO
--
--   Python:
--     Interpreter:                 /usr/bin/python2 (ver 2.7.3)
--     Libraries:                   /usr/lib/libpython2.7.so (ver 2.7.3)
--     numpy:                       /usr/lib/pymodules/python2.7/numpy/core/include (ver 1.6.2)
--     packages path:               lib/python2.7/dist-packages
--
--   Java:
--     ant:                         NO
--     JNI:                         NO
--     Java tests:                  NO
--
--   Documentation:
--     Build Documentation:         NO
--     Sphinx:                      NO
--     PdfLaTeX compiler:           /usr/bin/pdflatex
--
--   Tests and samples:
--     Tests:                       YES
--     Performance tests:           YES
--     C/C++ Examples:              NO
--
--   Install path:                  /usr/local
--
--   cvconfig.h is in:              /home/pi/opencv-2.4.10/build
-- -----------------------------------------------------------------
```

3. Build and install

```bash
sudo make
sudo make install
```

## Cloining the Repository and Running Script at boot

Clone the OpenVNAVI repository:

```bash
cd ~
git clone https://github.com/davidanton/OpenVNAVI.git
```

Run the following command:

```bash
sudo crontab -e
```
and add this line at the end

```
@reboot python /home/pi/OpenVNAVI/code/main.py &
```

---

Special thanks to Chat for testing and improving the build process.

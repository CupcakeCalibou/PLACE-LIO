# PLACE-LIO
This is the implementation for a plane-centric LiDAR-inertial odometry system tailored to exploit global planes (GPs).

## Dependency
* ROS
* PCL
* TBB

## Run PLACE-LIO
### 1. Compile
```
cd ~/catkin_ws/src
git clone https://github.com/CupcakeCalibou/PLACE-LIO.git
cd ../
catkin_make
source devel/setup.bash
``` 
### 2. Launch
* test on [UrbanNav](https://github.com/IPNL-POLYU/UrbanNavDataset)
```
roslaunch place_lio run_UrbanNav.launch
rosbag play UrbanNav-HK_TST-20210517_sensors.bag
```
* test on [M2DGR](https://github.com/SJTU-ViSYS/M2DGR)
```
roslaunch place_lio run_M2DGR.launch
rosbag play street_01.bag
``` 

## Related Papers
```
@ARTICLE{he2025place,
  author={He, Linkun and Li, Bofeng and Chen, Guang’e},
  journal={IEEE Robotics and Automation Letters}, 
  title={PLACE-LIO: Plane-Centric LiDAR-Inertial Odometry}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2025.3564790}
}

@ARTICLE{he2024noise,
  author={He, Linkun and Li, Bofeng and Chen, Guang’e},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Noise Model-Based Line Segmentation for Plane Extraction in Sparse 3-D LiDAR Data}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2024.3394059}
}
```

## Acknowledgements
Thanks for the great work of [FAST-LIO](https://github.com/hku-mars/FAST_LIO) and [KISS-ICP](https://github.com/PRBonn/kiss-icp).
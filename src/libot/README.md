# LiBot: Autonomous Library Robot

<p align="center">
  <img src="doc/Initial vid.gif" alt="Library Robot Demo" height="260">
</p>

LiBot is an **autonomous library robot** designed to help libraries manage, organize, and inventory their shelves efficiently.  
It’s built for the **[SICK LiDAR $10K Challenge](https://www.sick.com/us/en/sick-10k-challenge/w/10K-Challenge)** — only **"15 teams across the US and Canada to support innovation and student achievement in automation and technology"**. We were selected, and our team is proud to represent the University of Pennsylvania.  
The project spans one year, focusing on leveraging a **SICK 2D LiDAR (picoScan150)** with complementary **vision-based perception** to achieve precise navigation, book interaction and autonomous library management.

---

## Overview

LiBot autonomously:
- **Navigates** narrow aisles using LiDAR-based SLAM and obstacle avoidance  
- **Scans** and tracks books via onboard cameras and barcodes  
- **Organizes** shelves by locating misplaced or missing books  
- **Extends** its telescopic arm to pick & place target books  

---

## Hardware & Control

### Base
4-wheel differential drive (CMD velocity control)
```
/front_left_wheel/cmd_vel  | std_msgs/msg/Float64
/front_right_wheel/cmd_vel | std_msgs/msg/Float64
/back_left_wheel/cmd_vel   | std_msgs/msg/Float64
/back_right_wheel/cmd_vel  | std_msgs/msg/Float64
```

### Telescopic Arm
- **Revolute Joint:** `/arm_revolute/cmd_vel`  
- **Vertical Prismatic Joint:** `/arm_vertical/cmd_vel`  
- **Horizontal Prismatic Joint:** `/arm_horizontal/cmd_vel`  
- The vertical and horizontal links use **mimic joints** for synchronized extension.  
- Simple **inverse kinematics** allows for target grasping based on shelf position.

---

## Usage

To launch in ROS 2:
```bash
ros2 launch libot_bringup libot_basic.launch.xml
```

Command wheel or arm joints with:
```bash
ros2 topic pub /front_left_wheel/cmd_vel std_msgs/msg/Float64 "{data: 0.1}"
```

### Team

Developed at the **University of Pennsylvania** for the **SICK LiDAR 10K Challenge**.  
Focus on LiDAR-driven autonomy and interaction, perception, and manipulation.

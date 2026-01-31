#include "flex_bot_odometry/wheel_odom_node.hpp"

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WheelOdomNode>());
  rclcpp::shutdown();
  return 0;
}

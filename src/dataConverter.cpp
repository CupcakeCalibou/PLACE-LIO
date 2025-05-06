#include <common_lib.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

// 0: M2DGR, HongKong UrbanNav
int dataType;
std::string cloudTopic;
float minRange_;
float maxRange_;
ros::Subscriber subRaw;
ros::Publisher pubConverted;

const bool orderByTime(PointWithStamp &x, PointWithStamp &y) {return (x.time < y.time);};

// M2DGR数据集点云类型
struct PointM2DGR
{
  PCL_ADD_POINT4D;
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointM2DGR,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
  (std::uint16_t, ring, ring)
  (float, time, time)
)

template <typename PointT>
bool has_nan(PointT point)
{
  return std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z);
}

void publishCloud(const sensor_msgs::PointCloud2 &msg_old, pcl::PointCloud<PointWithStamp>::Ptr &cloud)
{
  std::sort(cloud->points.begin(), cloud->points.end(), orderByTime);
  cloud->is_dense = true;
  sensor_msgs::PointCloud2 msg_new;
  pcl::toROSMsg(*cloud, msg_new);
  msg_new.header = msg_old.header;
  msg_new.header.stamp = ros::Time().fromSec(cloud->points.front().time);
  msg_new.header.frame_id = "lidar";

  pubConverted.publish(msg_new);
}

void M2DGR_Handler(const sensor_msgs::PointCloud2 &msg)
{
  pcl::PointCloud<PointM2DGR>::Ptr cloudIn(new pcl::PointCloud<PointM2DGR>());
  pcl::fromROSMsg(msg, *cloudIn);
  pcl::PointCloud<PointWithStamp>::Ptr cloudOut(new pcl::PointCloud<PointWithStamp>());
  cloudOut->reserve(cloudIn->size());

  PointWithStamp newPoint;
  double baseTime = msg.header.stamp.toSec();
  for (size_t i = 0; i < cloudIn->size(); ++i)
  {
    const auto &point = cloudIn->points[i];
    if (has_nan(point)) continue;
    float range = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    if (range < minRange_ || range > maxRange_) continue;

    newPoint.x = point.x;
    newPoint.y = point.y;
    newPoint.z = point.z;
    newPoint.intensity = point.intensity;
    newPoint.ring = point.ring;
    newPoint.range = range;
    newPoint.time = baseTime + double(point.time);
    cloudOut->emplace_back(newPoint);
  }

  publishCloud(msg, cloudOut);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "dataConverter");
  ros::NodeHandle nh;

  nh.param<string>("lidar/lidar_topic", cloudTopic, "/velodyne_points");
  nh.param<int>("lidar/data_type", dataType, 0);
  nh.param<float>("lidar/minimum_range", minRange_, 0.1);
  nh.param<float>("lidar/maximum_range", maxRange_, 100);
  switch (dataType)
  {
    case 0:
      subRaw = nh.subscribe(cloudTopic, 10, M2DGR_Handler);
      break;
    case 1:
      break;
    default:
      ROS_ERROR("UNSUPPORTED INPUT TYPE!");
      exit(1);
  }
  
  pubConverted = nh.advertise<sensor_msgs::PointCloud2>("/laser_scan", 1);

  ros::spin();
  return 0;
}
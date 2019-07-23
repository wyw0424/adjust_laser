/*
 * pointcloud process ROS.cpp
 * Date: 2019-07-19
*/
#include <mutex>
#include <queue>
#include <fstream>
#include <csignal>
#include <cstdio>
#include <sstream>
#include <cstdlib>

#include "ros/ros.h"
#include <std_srvs/SetBool.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>  

#include <geometry_msgs/PoseStamped.h>
// PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/registration/gicp.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pthread.h>
#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

#include <point_cloud_odometry/PointCloudOdometry.h>
#include <point_cloud_filter/PointCloudFilter.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>


typedef pcl::PointCloud<pcl::PointXYZ> pcl_cloud;

namespace gu = geometry_utils;
using PointT = pcl::PointXYZ;
using pcl::GeneralizedIterativeClosestPoint;

class slam_front_end_ros_node
{

private:
  ros::NodeHandle node_;

  // Subscriber
  ros::Subscriber pc_sub_, imu_sub_;

  // Publisher
  ros::Publisher filter_pub_, odom_pub_, pub_undistorted_pc_;

  // Service
  //ros::ServiceServer clear_num_service_;

  // Params
  double filter_bot_threshold_;
  double filter_top_threshold_;
  double filter_range_;
  double scan_period_;
  double downsample_resolution_;
  bool first_cloud_;
  bool use_ndt_;
  bool use_icp_;
  bool adjust_distortion_;
  int pc_count_;

  PointCloudFilter filter_;
  PointCloudOdometry odometry_;



  std::string target_frame_;

  pcl::PointCloud<PointT>::Ptr globalmap_;
  std::mutex registration_mutex;
  pcl::Registration<PointT, PointT>::Ptr registration_;
  // Create a container for the data.
  std::mutex buffer_lock;
  std::queue<sensor_msgs::PointCloud2::Ptr> cloud_buffer_; 

  pcl::PointCloud<PointT>::Ptr last_cloud_;
  Eigen::Vector3f last_pos_;
  Eigen::Quaternionf last_quat_;
  Eigen::Matrix4f last_tf_matrix_;
  geometry_msgs::PoseStamped out_odom_;

  // imu 相关
  static const int imu_queue_len_ = 100;

  int imu_ptr_front_, imu_ptr_last_, imu_ptr_last_iter_;
  Eigen::Vector3f rpy_cur_, velo_xyz_cur_, shift_xyz_cur_;
  Eigen::Vector3f rpy_start_, velo_xyz_start_, shift_xyz_start_;
  Eigen::Vector3f shift_from_start_;
  Eigen::Matrix3f imu_ENU_to_NED_;

  std::array<double, imu_queue_len_> imu_time_;
  std::array<float, imu_queue_len_> imu_roll_;
  std::array<float, imu_queue_len_> imu_pitch_;
  std::array<float, imu_queue_len_> imu_yaw_;

  std::array<float, imu_queue_len_> imu_acc_x_;
  std::array<float, imu_queue_len_> imu_acc_y_;
  std::array<float, imu_queue_len_> imu_acc_z_;
  std::array<float, imu_queue_len_> imu_velo_x_;
  std::array<float, imu_queue_len_> imu_velo_y_;
  std::array<float, imu_queue_len_> imu_velo_z_;
  std::array<float, imu_queue_len_> imu_shift_x_;
  std::array<float, imu_queue_len_> imu_shift_y_;
  std::array<float, imu_queue_len_> imu_shift_z_;

  std::array<float, imu_queue_len_> imu_angular_velo_x_;
  std::array<float, imu_queue_len_> imu_angular_velo_y_;
  std::array<float, imu_queue_len_> imu_angular_velo_z_;
  std::array<float, imu_queue_len_> imu_angular_rot_x_;
  std::array<float, imu_queue_len_> imu_angular_rot_y_;
  std::array<float, imu_queue_len_> imu_angular_rot_z_;

public:

  slam_front_end_ros_node();
  ~slam_front_end_ros_node();

  void adjustDistortion(pcl::PointCloud<PointT>::Ptr &cloud, double scan_time);
  void imuCallback(const sensor_msgs::ImuConstPtr &msg);
  void laserCallback(const sensor_msgs::PointCloud2::ConstPtr& data);
  //bool clear_num_service(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& resp);
  void update();


};

slam_front_end_ros_node::slam_front_end_ros_node()
: use_ndt_(false)
, use_icp_(false)
, first_cloud_(true)
, scan_period_(0.1)
, pc_count_(0)
{

  if (!odometry_.Initialize(node_)) 
  {
      ROS_ERROR("Failed to initialize point cloud odometry.");
  }
  if (!filter_.Initialize(node_)) 
  {
    ROS_ERROR("Failed to initialize point cloud filter.");
  }
  // parameters
  std::string pointcloud_topic, imu_topic;
  std::string globalmap_pcd;
  ros::param::get("/slam_front_end_ros_node/pointcloud_topic", pointcloud_topic);
  ros::param::get("/slam_front_end_ros_node/imu_topic", imu_topic);
  ros::param::get("/slam_front_end_ros_node/map_name", globalmap_pcd);
  ros::param::get("/slam_front_end_ros_node/use_icp", use_icp_);
  ros::param::get("/slam_front_end_ros_node/use_ndt", use_ndt_);
  ros::param::get("/slam_front_end_ros_node/adjust_distortion", adjust_distortion_);
  ros::param::get("/slam_front_end_ros_node/downsample_resolution", downsample_resolution_);
  ros::param::get("/slam_front_end_ros_node/scan_period", scan_period_);
  // subscribers
  pc_sub_ = node_.subscribe<sensor_msgs::PointCloud2> (pointcloud_topic, 1, &slam_front_end_ros_node::laserCallback, this);
  imu_sub_ = node_.subscribe<sensor_msgs::Imu> (imu_topic, 50, &slam_front_end_ros_node::imuCallback, this);
  // publishers
  //image_pub_ = node_.advertise<sensor_msgs::Image>("/image", 1);
  filter_pub_ = node_.advertise<sensor_msgs::PointCloud2> ("filter_cloud", 1);
  odom_pub_ = node_.advertise<geometry_msgs::PoseStamped> ("odom_out", 1);
  pub_undistorted_pc_ = node_.advertise<sensor_msgs::PointCloud2>("/undistorted_pc", 1);
  // Services
  //clear_num_service_ = node_.advertiseService("/image_process/clear_num_service", &slam_front_end_ros_node::clear_num_service, this);

  if(use_ndt_)
  {
    double ndt_resolution = 1.0;
    pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
    ndt->setTransformationEpsilon(0.01);
    ndt->setResolution(ndt_resolution);
    ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
    //ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
    registration_mutex.lock();
    registration_ = ndt;
    registration_mutex.unlock();
  }

  //pcl::io::loadPCDFile(globalmap_pcd, *globalmap_);

/*  pcl::ApproximateVoxelGrid<PointT> voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
  voxelgrid->setLeafSize(downsample_resolution_, downsample_resolution_, downsample_resolution_);
  voxelgrid->setInputCloud(globalmap_);

  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  voxelgrid->filter(*filtered);

  globalmap_ = filtered;*/
  //registration->setInputTarget(globalmap_);
  out_odom_.header.frame_id = "velodyne";
  out_odom_.pose.position.x = 0.0;
  out_odom_.pose.position.y = 0.0;
  out_odom_.pose.position.z = 0.0;

  out_odom_.pose.orientation.x = 0.0;
  out_odom_.pose.orientation.y = 0.0;
  out_odom_.pose.orientation.z = 0.0;
  out_odom_.pose.orientation.w = 1.0;

  last_pos_(0) = 0;
  last_pos_(1) = 0;
  last_pos_(2) = 0;

  Eigen::Quaternionf quaternion(1.0,0,0,0);
  last_quat_ = quaternion;

  //imu init
  imu_ptr_front_ = 0;
  imu_ptr_last_ = -1;
  imu_ptr_last_iter_ = 0;

  imu_time_.fill(0);
  imu_roll_.fill(0);
  imu_pitch_.fill(0);
  imu_yaw_.fill(0);

  imu_acc_x_.fill(0);
  imu_acc_y_.fill(0);
  imu_acc_z_.fill(0);
  imu_velo_x_.fill(0);
  imu_velo_y_.fill(0);
  imu_velo_z_.fill(0);
  imu_shift_x_.fill(0);
  imu_shift_y_.fill(0);
  imu_shift_z_.fill(0);

  imu_angular_velo_x_.fill(0);
  imu_angular_velo_y_.fill(0);
  imu_angular_velo_z_.fill(0);
  imu_angular_rot_x_.fill(0);
  imu_angular_rot_y_.fill(0);
  imu_angular_rot_z_.fill(0);

  //imu_ENU_to_NED_ = (Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX())).toRotationMatrix();
  imu_ENU_to_NED_ << 0.0, 1.0, 0.0,
                     1.0, 0.0, 0.0,
                     0.0, 0.0, -1.0;

  std::cout << imu_ENU_to_NED_ << std::endl;
}

slam_front_end_ros_node::~slam_front_end_ros_node()
{
  //cloud_buffer_.clear();
}



/*
// service: start or stop recording image.
bool slam_front_end_ros_node::clear_num_service(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& resp)
{
    resp.success = req.data;
    node_num_=0; 
    resp.message = req.data ? "cleared.." : "cleared..";

    return resp.success;
}*/


/**
 * @brief 参考 loam 的点云去运动畸变（基于匀速运动假设）
 * 
 */
void slam_front_end_ros_node::adjustDistortion(pcl::PointCloud<PointT>::Ptr &cloud, double scan_time)
{
  pcl::PCDWriter writer;
  pc_count_ ++;
  std::stringstream count;
  count<<pc_count_; 
  std::string origin_name = "/home/yd/Desktop/pc/origin/";
  origin_name =  origin_name+count.str()+".pcd";
  writer.write<pcl::PointXYZ> (origin_name, *cloud, true);

  bool half_passed = false;
  int cloud_size = cloud->points.size();

  float start_ori = -std::atan2(cloud->points[0].y, cloud->points[0].x);
  float end_ori = -std::atan2(cloud->points[cloud_size - 1].y, cloud->points[cloud_size - 1].x);
  if (end_ori - start_ori > 3 * M_PI)
  {
    end_ori -= 2 * M_PI;
  }
  else if (end_ori - start_ori < M_PI)
  {
    end_ori += 2 * M_PI;
  }
  float ori_diff = end_ori - start_ori;

  Eigen::Vector3f rpy_start, shift_start, velo_start, rpy_cur, shift_cur, velo_cur;
  Eigen::Vector3f shift_from_start;
  Eigen::Matrix3f r_s_i, r_c;
  Eigen::Vector3f adjusted_p;
  float ori_h;
  for (int i = 0; i < cloud_size; ++i)
  {
    PointT &p = cloud->points[i];
    ori_h = -std::atan2(p.y, p.x);
    if (!half_passed)
    {
      if (ori_h < start_ori - M_PI * 0.5)
      {
        ori_h += 2 * M_PI;
      }
      else if (ori_h > start_ori + M_PI * 1.5)
      {
        ori_h -= 2 * M_PI;
      }

      if (ori_h - start_ori > M_PI)
      {
        half_passed = true;
      }
    }
    else
    {
      ori_h += 2 * M_PI;
      if (ori_h < end_ori - 1.5 * M_PI)
      {
        ori_h += 2 * M_PI;
      }
      else if (ori_h > end_ori + 0.5 * M_PI)
      {
        ori_h -= 2 * M_PI;
      }
    }

    float rel_time = (ori_h - start_ori) / ori_diff * scan_period_;

    if (imu_ptr_last_ > 0)
    {
      imu_ptr_front_ = imu_ptr_last_iter_;
      while (imu_ptr_front_ != imu_ptr_last_)
      {
        if (scan_time + rel_time < imu_time_[imu_ptr_front_])
        {
          break;
        }
        imu_ptr_front_ = (imu_ptr_front_ + 1) % imu_queue_len_;
      }
      if (std::abs(scan_time + rel_time - imu_time_[imu_ptr_front_]) > scan_period_)
      {
        ROS_WARN_COND(i < 10, "unsync imu and pc msg");
        continue;
      }

      if (scan_time + rel_time > imu_time_[imu_ptr_front_])
      {
        rpy_cur(0) = imu_roll_[imu_ptr_front_];
        rpy_cur(1) = imu_pitch_[imu_ptr_front_];
        rpy_cur(2) = imu_yaw_[imu_ptr_front_];
        shift_cur(0) = imu_shift_x_[imu_ptr_front_];
        shift_cur(1) = imu_shift_y_[imu_ptr_front_];
        shift_cur(2) = imu_shift_z_[imu_ptr_front_];
        velo_cur(0) = imu_velo_x_[imu_ptr_front_];
        velo_cur(1) = imu_velo_y_[imu_ptr_front_];
        velo_cur(2) = imu_velo_z_[imu_ptr_front_];
        //std::cout <<"scan_time + rel_time > imu_time_[imu_ptr_front" << std::endl;
        // std::cout << "shift: " << shift_cur(0) << ", " << shift_cur(1) << ", " << shift_cur(2) << std::endl;
        // std::cout << "velo: " << velo_cur(0) << ", " << velo_cur(1) << ", " << velo_cur(2) << std::endl;
      }
      else
      {
        int imu_ptr_back = (imu_ptr_front_ - 1 + imu_queue_len_) % imu_queue_len_;
        float ratio_front = (scan_time + rel_time - imu_time_[imu_ptr_back]) / (imu_time_[imu_ptr_front_] - imu_time_[imu_ptr_back]);
        float ratio_back = 1. - ratio_front;
        rpy_cur(0) = imu_roll_[imu_ptr_front_] * ratio_front + imu_roll_[imu_ptr_back] * ratio_back;
        rpy_cur(1) = imu_pitch_[imu_ptr_front_] * ratio_front + imu_pitch_[imu_ptr_back] * ratio_back;
        rpy_cur(2) = imu_yaw_[imu_ptr_front_] * ratio_front + imu_yaw_[imu_ptr_back] * ratio_back;
        shift_cur(0) = imu_shift_x_[imu_ptr_front_] * ratio_front + imu_shift_x_[imu_ptr_back] * ratio_back;
        shift_cur(1) = imu_shift_y_[imu_ptr_front_] * ratio_front + imu_shift_y_[imu_ptr_back] * ratio_back;
        shift_cur(2) = imu_shift_z_[imu_ptr_front_] * ratio_front + imu_shift_z_[imu_ptr_back] * ratio_back;
        velo_cur(0) = imu_velo_x_[imu_ptr_front_] * ratio_front + imu_velo_x_[imu_ptr_back] * ratio_back;
        velo_cur(1) = imu_velo_y_[imu_ptr_front_] * ratio_front + imu_velo_y_[imu_ptr_back] * ratio_back;
        velo_cur(2) = imu_velo_z_[imu_ptr_front_] * ratio_front + imu_velo_z_[imu_ptr_back] * ratio_back;
      }

      r_c = (Eigen::AngleAxisf(rpy_cur(2), Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(rpy_cur(1), Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(rpy_cur(0), Eigen::Vector3f::UnitX())).toRotationMatrix();
      r_c = r_c * imu_ENU_to_NED_;

      // std::cout << "rpy: " << rpy_cur(0) << ", " << rpy_cur(1) << ", " << rpy_cur(2) << std::endl;
      // std::cout << "shift: " << shift_cur(0) << ", " << shift_cur(1) << ", " << shift_cur(2) << std::endl;
      // std::cout << "velo: " << velo_cur(0) << ", " << velo_cur(1) << ", " << velo_cur(2) << std::endl;

      if (i == 0)
      {
        rpy_start = rpy_cur;
        shift_start = shift_cur;
        velo_start = velo_cur;
        r_s_i = r_c.inverse();
      }
      else
      {
        shift_from_start = shift_cur - shift_start - velo_start * rel_time;
        adjusted_p = r_s_i * (r_c * Eigen::Vector3f(p.x, p.y, p.z) + shift_from_start);
        p.x = adjusted_p.x();
        p.y = adjusted_p.y();
        p.z = adjusted_p.z();
      }
    }
    imu_ptr_last_iter_ = imu_ptr_front_;
  }

  if (pub_undistorted_pc_.getNumSubscribers() > 0)
  {
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.stamp.fromSec(scan_time);
    msg.header.frame_id = "/velodyne";
    pub_undistorted_pc_.publish(msg);
  }
  std::string adjust_name = "/home/yd/Desktop/pc/adjusted/";
  adjust_name =  adjust_name + count.str()+".pcd";
  writer.write<pcl::PointXYZ> (adjust_name, *cloud, true);
}

void slam_front_end_ros_node::imuCallback(const sensor_msgs::ImuConstPtr &msg)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(msg->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  // std::cout << "rpy: " << roll << ", " << pitch << ", " << yaw << std::endl;
  // Eigen::Vector3f rpy(roll, pitch, yaw);
  // rpy = imu_ENU_to_NED_ *rpy;

  // roll = rpy(0);
  // pitch = rpy(1);
  // yaw = rpy(2);
  // std::cout << "after rpy: " << roll << ", " << pitch << ", " << yaw << std::endl;

  float acc_x = msg->linear_acceleration.x + 9.81 * sin(pitch);
  float acc_y = msg->linear_acceleration.y - 9.81 * cos(pitch) * sin(roll);
  float acc_z = msg->linear_acceleration.z + 9.81 * cos(pitch) * cos(roll);

  // std::cout << "9.81 * sin(pitch): " << 9.81 * sin(pitch) << std::endl;
  // std::cout << "9.81 * cos(pitch) * sin(roll): " << 9.81 * cos(pitch) * sin(roll) << std::endl;
  // std::cout << "9.81 * cos(pitch) * cos(roll): " << 9.81 * cos(pitch) * cos(roll) << std::endl;

  // std::cout << "acc: " << acc_x << ", " << acc_y << ", " << acc_z << std::endl;

  imu_ptr_last_ = (imu_ptr_last_ + 1) % imu_queue_len_;

  if ((imu_ptr_last_ + 1) % imu_queue_len_ == imu_ptr_front_)
  {
    imu_ptr_front_ = (imu_ptr_front_ + 1) % imu_queue_len_;
  }

  imu_time_[imu_ptr_last_] = msg->header.stamp.toSec();

  imu_roll_[imu_ptr_last_] = roll;
  imu_pitch_[imu_ptr_last_] = pitch;
  imu_yaw_[imu_ptr_last_] = yaw;

  imu_acc_x_[imu_ptr_last_] = acc_x;
  imu_acc_y_[imu_ptr_last_] = acc_y;
  imu_acc_z_[imu_ptr_last_] = acc_z;

  imu_angular_velo_x_[imu_ptr_last_] = msg->angular_velocity.x;
  imu_angular_velo_y_[imu_ptr_last_] = msg->angular_velocity.y;
  imu_angular_velo_z_[imu_ptr_last_] = msg->angular_velocity.z;

  // 转换到 imu 的全局坐标系中
  Eigen::Matrix3f rot = Eigen::Quaternionf(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z).toRotationMatrix();
  Eigen::Vector3f acc = rot * Eigen::Vector3f(acc_x, acc_y, acc_z);
  // TODO: lego_loam 里没有对角速度转换，是否需要尚且存疑
  // Eigen::Vector3f angular_velo = rot * Eigen::Vector3f(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
  Eigen::Vector3f angular_velo(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  int imu_ptr_back = (imu_ptr_last_ - 1 + imu_queue_len_) % imu_queue_len_;
  double time_diff = imu_time_[imu_ptr_last_] - imu_time_[imu_ptr_back];
  if (time_diff < scan_period_)
  {
    imu_shift_x_[imu_ptr_last_] = imu_shift_x_[imu_ptr_back] + imu_velo_x_[imu_ptr_back] * time_diff + acc(0) * time_diff * time_diff * 0.5;
    imu_shift_y_[imu_ptr_last_] = imu_shift_y_[imu_ptr_back] + imu_velo_y_[imu_ptr_back] * time_diff + acc(1) * time_diff * time_diff * 0.5;
    imu_shift_z_[imu_ptr_last_] = imu_shift_z_[imu_ptr_back] + imu_velo_z_[imu_ptr_back] * time_diff + acc(2) * time_diff * time_diff * 0.5;

    imu_velo_x_[imu_ptr_last_] = imu_velo_x_[imu_ptr_back] + acc(0) * time_diff;
    imu_velo_y_[imu_ptr_last_] = imu_velo_y_[imu_ptr_back] + acc(1) * time_diff;
    imu_velo_z_[imu_ptr_last_] = imu_velo_z_[imu_ptr_back] + acc(2) * time_diff;

    imu_angular_rot_x_[imu_ptr_last_] = imu_angular_rot_x_[imu_ptr_back] + angular_velo(0) * time_diff;
    imu_angular_rot_y_[imu_ptr_last_] = imu_angular_rot_y_[imu_ptr_back] + angular_velo(1) * time_diff;
    imu_angular_rot_z_[imu_ptr_last_] = imu_angular_rot_z_[imu_ptr_back] + angular_velo(2) * time_diff;
  }
}

// PointCloud2 call back
void slam_front_end_ros_node::laserCallback(const sensor_msgs::PointCloud2::ConstPtr& data)
{
  sensor_msgs::PointCloud2::Ptr cloud (new sensor_msgs::PointCloud2);
  *cloud = *data;
  buffer_lock.lock();
  cloud_buffer_.push(cloud);
  buffer_lock.unlock();
  //std::cout << "Buffer Size: " << cloud_buffer_.size() << std::endl;
}

void slam_front_end_ros_node::update()
{
  if (cloud_buffer_.size() > 0)
  {
    pcl_cloud::Ptr curr_cloud (new pcl_cloud() );
    //std::cout << "Buffer Size: " << cloud_buffer_.size() << std::endl;
    buffer_lock.lock();
    sensor_msgs::PointCloud2::Ptr cloud = cloud_buffer_.front();
    cloud_buffer_.pop();
    buffer_lock.unlock();

    // Convert ROS -> PCL
    pcl::fromROSMsg(*cloud, *curr_cloud);

    pcl::PointCloud<PointT>::Ptr msg_filtered(new pcl::PointCloud<PointT>);
    filter_.Filter(curr_cloud, msg_filtered);

    if(first_cloud_)
    {
      last_cloud_ = msg_filtered;
      first_cloud_ = false;
    }

    if (adjust_distortion_)
    {
      adjustDistortion(curr_cloud, cloud->header.stamp.toSec());
    }

    if(use_icp_)
    {
      // GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
      // icp.setTransformationEpsilon(0.000001);
      // icp.setMaxCorrespondenceDistance(0.2);
      // icp.setMaximumIterations(5);
      // icp.setRANSACIterations(0);

      // icp.setInputSource(msg_filtered);
      // icp.setInputTarget(last_cloud_);

      // pcl::PointCloud<pcl::PointXYZ> unused_result;
      // icp.align(unused_result);
      // const Eigen::Matrix4f trans = icp.getFinalTransformation();
      // Eigen::Vector3f p = trans.block<3, 1>(0, 3);
      // Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

      // last_cloud_ = msg_filtered;
      // std::cout << out_odom_.pose.position.x << ", " <<out_odom_.pose.position.y <<", " << out_odom_.pose.position.z << std::endl;
      // out_odom_.pose.position.x += p(0);
      // out_odom_.pose.position.y += p(1);
      // out_odom_.pose.position.z += p(2);

      // Eigen::Quaternionf tmp_q(out_odom_.pose.orientation.w, out_odom_.pose.orientation.x, out_odom_.pose.orientation.y, out_odom_.pose.orientation.z);
      // Eigen::Quaternionf out_q = q*tmp_q;

      // out_odom_.pose.orientation.w = out_q.w();
      // out_odom_.pose.orientation.x = out_q.x();
      // out_odom_.pose.orientation.y = out_q.y();
      // out_odom_.pose.orientation.z = out_q.z();
      // odom_pub_.publish(out_odom_);

            // Update odometry by performing ICP.
      if (!odometry_.UpdateEstimate(*msg_filtered)) 
      {
        gu::Transform3 estimate = odometry_.GetIncrementalEstimate();
      }
    }


    if(use_ndt_)
    {
      ros::Time begin = ros::Time::now();
      registration_->setInputTarget(last_cloud_);
      ros::Time now = ros::Time::now();
      double sec = (now - begin).toSec();
      std::cout << "spend time: " << sec << std::endl;

      Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
      //init_guess.block<3, 3>(0, 0) = last_quat_.toRotationMatrix();
      //init_guess.block<3, 1>(0, 3) = last_pos_;
      
      pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
      registration_->setInputSource(msg_filtered);
      registration_->align(*aligned, init_guess);

      Eigen::Matrix4f trans = registration_->getFinalTransformation();
      Eigen::Vector3f p = trans.block<3, 1>(0, 3);
      Eigen::Quaternionf q(trans.block<3, 3>(0, 0));
      //last_pos_ = p;
      //last_quat_ = q;
      last_cloud_ = msg_filtered;
      std::cout << out_odom_.pose.position.x << ", " <<out_odom_.pose.position.y <<", " << out_odom_.pose.position.z << std::endl;
      out_odom_.pose.position.x += p(0);
      out_odom_.pose.position.y += p(1);
      out_odom_.pose.position.z += p(2);

      Eigen::Quaternionf tmp_q(out_odom_.pose.orientation.w, out_odom_.pose.orientation.x, out_odom_.pose.orientation.y, out_odom_.pose.orientation.z);
      Eigen::Quaternionf out_q = q*tmp_q;

      out_odom_.pose.orientation.w = out_q.w();
      out_odom_.pose.orientation.x = out_q.x();
      out_odom_.pose.orientation.y = out_q.y();
      out_odom_.pose.orientation.z = out_q.z();
      odom_pub_.publish(out_odom_);
    }

  }
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "slam_front_end_ros_node");
  slam_front_end_ros_node node;

  ROS_INFO("slam front end ros node started...");

  ros::Rate rate(200);

  while(ros::ok())
  {
    ros::spinOnce();
    node.update();
    rate.sleep();
  }
  return 0;
}



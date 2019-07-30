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

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


typedef pcl::PointCloud<pcl::PointXYZ> pcl_cloud;

using PointT = pcl::PointXYZ;

class laser_adjust_offline
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
  bool inside_one_laser_;
  bool publish_undistored_pc2_;
  bool save_result_;
  int pc_count_;
  int imu_count_;
  int laser_count_;

  std::string target_frame_;
  std::string pointcloud_topic_, imu_topic_;
  std::string input_bag_, output_bag_;

  pcl::PointCloud<PointT>::Ptr globalmap_;
  std::mutex registration_mutex;
  pcl::Registration<PointT, PointT>::Ptr registration_;
  // Create a container for the data.
  std::mutex buffer_lock;
  std::queue<sensor_msgs::PointCloud2::Ptr> cloud_buffer_; 
  std::mutex imu_lock_;
  std::queue<sensor_msgs::Imu::Ptr> imu_buffer_; 

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

  laser_adjust_offline();
  ~laser_adjust_offline();

  sensor_msgs::PointCloud2 adjustDistortion(pcl::PointCloud<PointT>::Ptr &cloud, double scan_time);
  void update_imu_data(sensor_msgs::Imu::Ptr imu_curr);
  void update();

};

laser_adjust_offline::laser_adjust_offline()
: first_cloud_(true)
, scan_period_(0.05)
, pc_count_(0)
, laser_count_(0)
, imu_count_(0)
, publish_undistored_pc2_(false)
, save_result_(false)
{
  // parameters

  std::string globalmap_pcd;
  ros::param::get("/laser_adjust_offline/pointcloud_topic", pointcloud_topic_);
  ros::param::get("/laser_adjust_offline/imu_topic", imu_topic_);
  ros::param::get("/laser_adjust_offline/map_name", globalmap_pcd);
  ros::param::get("/laser_adjust_offline/downsample_resolution", downsample_resolution_);
  ros::param::get("/laser_adjust_offline/scan_period", scan_period_);
  ros::param::get("/laser_adjust_offline/publish_undistored_pc2", publish_undistored_pc2_);
  ros::param::get("/laser_adjust_offline/save_result", save_result_);
  ros::param::get("/laser_adjust_offline/input_bag", input_bag_);
  ros::param::get("/laser_adjust_offline/output_bag", output_bag_);
  // subscribers

  // pc_sub_ = node_.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 1, &laser_adjust_offline::laserCallback, this);
  // imu_sub_ = node_.subscribe<sensor_msgs::Imu> ("/imu/data", 1, &laser_adjust_offline::imuCallback, this);

  // publishers
  //image_pub_ = node_.advertise<sensor_msgs::Image>("/image", 1);
  if(publish_undistored_pc2_)
  {
    pub_undistorted_pc_ = node_.advertise<sensor_msgs::PointCloud2>("/undistorted_pc", 1);
  }
  // Services
  //clear_num_service_ = node_.advertiseService("/image_process/clear_num_service", &laser_adjust_offline::clear_num_service, this);


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

  inside_one_laser_ = true;
}

laser_adjust_offline::~laser_adjust_offline()
{
  //cloud_buffer_.clear();
}



/**
 * @brief 参考 loam 的点云去运动畸变（基于匀速运动假设）
 * 
 */
sensor_msgs::PointCloud2 laser_adjust_offline::adjustDistortion(pcl::PointCloud<PointT>::Ptr &cloud, double scan_time)
{
  pcl::PCDWriter writer;
  pc_count_ ++;
  std::stringstream count;
  count<<pc_count_; 
  if(save_result_)
  {
    std::string origin_name = "/home/derek/Desktop/pc/origin/";
    origin_name =  origin_name+count.str()+".pcd";
    writer.write<pcl::PointXYZ> (origin_name, *cloud, true);
  }

  bool half_passed = false;
  int cloud_size = cloud->points.size();

  float start_ori = -std::atan2(cloud->points[0].y, cloud->points[0].x);
  float end_ori = -std::atan2(cloud->points[cloud_size - 1].y, cloud->points[cloud_size - 1].x);
  std::cout << "start_ori:" << start_ori << "  end_ori:" << end_ori <<std::endl;
  std::cout << "end_ori - start_ori :" << end_ori - start_ori  <<std::endl;
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

  if(save_result_)
  {
    std::string adjust_name = "/home/derek/Desktop/pc/adjusted/";
    adjust_name =  adjust_name + count.str()+".pcd";
    writer.write<pcl::PointXYZ> (adjust_name, *cloud, true);
  }

  sensor_msgs::PointCloud2 adjusted_pc2;
  pcl::toROSMsg(*cloud, adjusted_pc2);
  adjusted_pc2.header.stamp.fromSec(scan_time);
  adjusted_pc2.header.frame_id = "/velodyne";
  if (publish_undistored_pc2_)
  {
    pub_undistorted_pc_.publish(adjusted_pc2);
  }
  return adjusted_pc2;
}

void laser_adjust_offline::update_imu_data(sensor_msgs::Imu::Ptr imu_curr)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imu_curr->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  float acc_x = imu_curr->linear_acceleration.x + 9.81 * sin(pitch);
  float acc_y = imu_curr->linear_acceleration.y - 9.81 * cos(pitch) * sin(roll);
  float acc_z = imu_curr->linear_acceleration.z + 9.81 * cos(pitch) * cos(roll);

  imu_ptr_last_ = (imu_ptr_last_ + 1) % imu_queue_len_;

  if ((imu_ptr_last_ + 1) % imu_queue_len_ == imu_ptr_front_)
  {
    imu_ptr_front_ = (imu_ptr_front_ + 1) % imu_queue_len_;
  }

  imu_time_[imu_ptr_last_] = imu_curr->header.stamp.toSec();

  imu_roll_[imu_ptr_last_] = roll;
  imu_pitch_[imu_ptr_last_] = pitch;
  imu_yaw_[imu_ptr_last_] = yaw;

  imu_acc_x_[imu_ptr_last_] = acc_x;
  imu_acc_y_[imu_ptr_last_] = acc_y;
  imu_acc_z_[imu_ptr_last_] = acc_z;

  imu_angular_velo_x_[imu_ptr_last_] = imu_curr->angular_velocity.x;
  imu_angular_velo_y_[imu_ptr_last_] = imu_curr->angular_velocity.y;
  imu_angular_velo_z_[imu_ptr_last_] = imu_curr->angular_velocity.z;

  // 转换到 imu 的全局坐标系中
  Eigen::Matrix3f rot = Eigen::Quaternionf(imu_curr->orientation.w, imu_curr->orientation.x, imu_curr->orientation.y, imu_curr->orientation.z).toRotationMatrix();
  Eigen::Vector3f acc = rot * Eigen::Vector3f(acc_x, acc_y, acc_z);
  // TODO: lego_loam 里没有对角速度转换，是否需要尚且存疑

  Eigen::Vector3f angular_velo(imu_curr->angular_velocity.x, imu_curr->angular_velocity.y, imu_curr->angular_velocity.z);

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

void laser_adjust_offline::update()
{
    rosbag::Bag bag;
    rosbag::Bag write_bag;
    bag.open(input_bag_, rosbag::bagmode::Read);
    write_bag.open(output_bag_, rosbag::bagmode::Write);

    rosbag::View view_laser(bag, rosbag::TopicQuery(pointcloud_topic_));
    rosbag::View view_imu(bag, rosbag::TopicQuery(imu_topic_));
    ros::Time imu_curr_time;
    ros::Time laser_curr_time;
    pcl_cloud::Ptr curr_cloud (new pcl_cloud() );
    foreach(rosbag::MessageInstance const m, view_imu)
    {
        sensor_msgs::Imu::Ptr imu = m.instantiate<sensor_msgs::Imu>();
        if (imu != NULL)
            imu_buffer_.push(imu);
            imu_count_++;
    }

    foreach(rosbag::MessageInstance const m, view_laser)
    {
        sensor_msgs::PointCloud2::ConstPtr laser = m.instantiate<sensor_msgs::PointCloud2>();
        if (laser != NULL)
        {
            pcl::fromROSMsg(*laser, *curr_cloud);
            laser_curr_time = laser->header.stamp;
            inside_one_laser_ = true;
            laser_count_++;
            std::cout << "Processing laser: " << laser_count_ << std::endl;
            while(inside_one_laser_)
            {
              if(imu_buffer_.size()>0)
              {
                sensor_msgs::Imu::Ptr imu_curr = imu_buffer_.front();
                imu_buffer_.pop();
                imu_curr_time = imu_curr->header.stamp;
                ros::Duration off_time = imu_curr_time -laser_curr_time;
                write_bag.write(imu_topic_, imu_curr_time, imu_curr);
                update_imu_data(imu_curr);
                if(off_time > ros::Duration(scan_period_-0.001))
                {
                  inside_one_laser_ = false;
                }
              }else
              {
                inside_one_laser_ = false;
              }
            }
            std::cout << "laser time: " << laser_curr_time << " imu time: " << imu_curr_time << std::endl;
            sensor_msgs::PointCloud2 adjusted_pc2 = adjustDistortion(curr_cloud, laser_curr_time.toSec());
            write_bag.write(pointcloud_topic_, adjusted_pc2.header.stamp, adjusted_pc2);
        }
    }

    std::cout << "Total imu: " << imu_count_ << std::endl;
    std::cout << "Total laser: " << laser_count_ << std::endl;
    bag.close();
    write_bag.close();
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "laser_adjust_offline");
  laser_adjust_offline node;

  ROS_INFO("slam front end ros node started...");

  ros::Rate rate(100);

  //while(ros::ok())
  {
    ros::spinOnce();
    node.update();
    rate.sleep();
  }
  return 0;
}



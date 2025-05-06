// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#define PCL_NO_PRECOMPILE

#include <csignal>
#include <fstream>
#include <iostream>
#include <mutex>
#include <math.h>
#include <omp.h>
#include <thread>
#include <unistd.h>

#include <Eigen/Core>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <so3_math.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/Marker.h>

#include "IMU_Processing.hpp"
#include "planeExtraction.hpp"
#include <pov-Map/pov_Map.hpp>

#define INIT_TIME           (0.1)
#define PUBFRAME_PERIOD     (20)
#define MAX_PLANE           (200)

using namespace pov_map;

using Vector2 = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>; // Vector × Vector
using Vector3 = std::vector<Vector2>;                                                    // Vector × Vector × Vector
using UncertainPatch = std::tuple<V3D, V3D, Eigen::Matrix<double, 6, 6>>;

double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;
                                                                                                               
std::string root_dir = ROOT_DIR;
std::string imu_topic;

int scanNum, planeNum;
float resolution, rangeSigma, bearingSigma;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double corr_time = 1.0;
double voxel_size = 1.0, detection_range = 100;
int max_points_per_voxel = 20;
float T_dist;
double lidar_end_time = 0, first_lidar_time = 0.0;
int pointToPointNum, pointToPlaneNum, planeToPlaneNum; 
int constraintNum = 0;
int pcd_save_interval = -1, pcd_index = 0;
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_deskew = true;
bool pcd_save_en = false, scan_pub_en = false, time_sync_en = false, path_en = true;
double planeExtractionTime = 0, preprocessingTime = 0, solvingTime = 0, mapUpdateTime = 0;
double totalTimeSquares = 0, planeTimeSquares = 0;

vector<double> gyr_rw(3, 0.0);
vector<double> acc_rw(3, 0.0);
vector<double> b_gyr_std(3, 0.0);
vector<double> b_acc_std(3, 0.0);
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);

deque<POINTCLOUD_PTR> lidar_buffer;
deque<double> plane_time_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

std::vector<int> singleOriIndex;     // point-to-point
Vector2 closestPoint;
std::vector<int> collinearOriIndex;  // point-to-plane
std::vector<int> collinearOriIndex2;
Vector3 closestCluster;
std::vector<UncertainPatch, Eigen::aligned_allocator<UncertainPatch>> closestCluster2;
std::vector<int> closestPlane;       // plane-to-plane
std::vector<M4D, Eigen::aligned_allocator<M4D>> pointClusters; 

POINTCLOUD_PTR feats_undistort(new POINTCLOUD());
POINTCLOUD_PTR featsFromMap(new POINTCLOUD());
// Coplanar points
vector<POINTCLOUD_PTR> fullPlanes(MAX_PLANE);
vector<POINTCLOUD_PTR> densePlanes(MAX_PLANE);  // for map update 
vector<POINTCLOUD_PTR> sparsePlanes(MAX_PLANE); // for data association
vector<POINTCLOUD> wrldPlanes(MAX_PLANE);       // transformed sparse planes in world frame
vector<HessianPlane, Eigen::aligned_allocator<HessianPlane>> planeParams(MAX_PLANE);
// Collinear points
POINTCLOUD_PTR sparseLinearPoints(new POINTCLOUD());
POINTCLOUD_PTR wrldLinearPoints(new POINTCLOUD());
// Single points
POINTCLOUD_PTR sparseSinglePoints(new POINTCLOUD());
POINTCLOUD_PTR extendedSinglePoints(new POINTCLOUD());
POINTCLOUD_PTR wrldSinglePoints(new POINTCLOUD());

V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;
Eigen::Matrix<double, 6, 6> cov_pr;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

unique_ptr<moi::LineBasedDouglasPeucker> p_pre;
unique_ptr<ImuProcess> p_imu(new ImuProcess());
shared_ptr<PlaneOccupiedVoxelMap> p_map;

void SigHandle(int sig)
{
	flg_exit = true;
	ROS_WARN("catch sig %d", sig);
	sig_buffer.notify_all();
}

void transformBodyToWorldLite(POINTCLOUD_PTR &cloud)
{
	for (auto it = cloud->begin(); it != cloud->end(); ++it)
	{
		V3D point_L(it->x, it->y, it->z);
		V3D point_W(state_point.rot * (state_point.offset_R_L_I * point_L + state_point.offset_T_L_I) + state_point.pos);
		it->x = point_W.x();
		it->y = point_W.y();
		it->z = point_W.z();
	}
}

void transformBodyToWorld(POINTCLOUD_PTR &cloud)
{
	for (auto it = cloud->begin(); it != cloud->end(); ++it)
	{
		V3D point_L(it->x, it->y, it->z);
		V3D point_I(state_point.offset_R_L_I * point_L + state_point.offset_T_L_I);
		V3D point_W(state_point.rot * point_I + state_point.pos);
		it->x = point_W.x();
		it->y = point_W.y();
		it->z = point_W.z();

		Eigen::Map<Eigen::Matrix3d> cov_point(it->covariance);
		M3D crossMat_I;
		crossMat_I << SKEW_SYM_MATRX(point_I);
		Eigen::Matrix<double, 3, 6> jacobiMat;
		jacobiMat.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();
		jacobiMat.topRightCorner<3, 3>() = -state_point.rot.toRotationMatrix() * crossMat_I;
		cov_point += jacobiMat * cov_pr * jacobiMat.transpose();
		std::memcpy(it->covariance, cov_point.data(), 9*sizeof(double));
	}
}

void RGBpointBodyToWorld(PointType const * const pi, pcl::PointXYZI * const po)
{
	V3D p_body(pi->x, pi->y, pi->z);
	V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

	po->x = p_global(0);
	po->y = p_global(1);
	po->z = p_global(2);
	po->intensity = pi->label;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
	double t_start = omp_get_wtime();
	POINTCLOUD_PTR ptr(new POINTCLOUD());
	p_pre->process(msg, ptr);
	double interval = omp_get_wtime() - t_start;
	double timestamp = msg->header.stamp.toSec();

	mtx_buffer.lock();
	if (timestamp < last_timestamp_lidar)
	{
		ROS_ERROR("lidar loop back, clear buffer");
		lidar_buffer.clear();
	}
	lidar_buffer.emplace_back(ptr);
	plane_time_buffer.emplace_back(interval);
	last_timestamp_lidar = timestamp;
	mtx_buffer.unlock();

	sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
	sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
	msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
	if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
	{
		msg->header.stamp = ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
	}

	double timestamp = msg->header.stamp.toSec();
	
	mtx_buffer.lock();
	if (timestamp < last_timestamp_imu)
	{
		ROS_WARN("imu loop back, clear buffer");
		imu_buffer.clear();
	}
	imu_buffer.emplace_back(msg);
	last_timestamp_imu = timestamp;
	mtx_buffer.unlock();

	sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
	if (lidar_buffer.empty() || imu_buffer.empty())
	{
		return false;
	}

	/*** push a lidar scan ***/
	if(!lidar_pushed)
	{
		meas.lidar = lidar_buffer.front();
		meas.plane_cost_time = plane_time_buffer.front();
		meas.lidar_beg_time = meas.lidar->points.front().time;

		if (meas.lidar->points.size() <= 1) // time too little
		{
			lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
			ROS_WARN("Too few input point cloud!\n");
		}
		else if (meas.lidar->points.back().time - meas.lidar_beg_time < 0.5 * lidar_mean_scantime)
		{
			lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
		}
		else
		{
			scan_num ++;
			lidar_end_time = meas.lidar->points.back().time;
			lidar_mean_scantime += (lidar_end_time - meas.lidar_beg_time - lidar_mean_scantime) / scan_num;
		}

		meas.lidar_end_time = lidar_end_time;

		lidar_pushed = true;
	}

	if (last_timestamp_imu < lidar_end_time)
	{
		return false;
	}

	/*** push imu data, and pop from imu buffer ***/
	double imu_time = imu_buffer.front()->header.stamp.toSec();
	meas.imu.clear();
	while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
	{
		imu_time = imu_buffer.front()->header.stamp.toSec();
		if(imu_time > lidar_end_time) break;
		meas.imu.emplace_back(imu_buffer.front());
		imu_buffer.pop_front();
	}

	lidar_buffer.pop_front();
	plane_time_buffer.pop_front();
	lidar_pushed = false;
	return true;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_wait_save(new pcl::PointCloud<pcl::PointXYZI>());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
	int size = feats_undistort->points.size();
	pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudWorld(new pcl::PointCloud<pcl::PointXYZI>(size, 1));
	for (int i = 0; i < size; i++)
	{
		RGBpointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
	}

	if(scan_pub_en)
	{
		sensor_msgs::PointCloud2 laserCloudmsg;
		pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
		laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
		laserCloudmsg.header.frame_id = "camera_init";
		pubLaserCloudFull.publish(laserCloudmsg);
	}

	/**************** save map ****************/
	/* 1. make sure you have enough memories
	/* 2. noted that pcd save will influence the real-time performences **/
	if (pcd_save_en)
	{
		*pcl_wait_save += *laserCloudWorld;

		static int scan_wait_num = 0;
		scan_wait_num ++;
		if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
		{
			pcd_index ++;
			string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
			pcl::PCDWriter pcd_writer;
			cout << "current scan saved to /PCD/" << all_points_dir << endl;
			pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
			pcl_wait_save->clear();
			scan_wait_num = 0;
		}
	}
}

void publish_planes_world(const ros::Publisher & pubLaserCloudFull)
{
	if(scan_pub_en)
	{
		int size = feats_undistort->points.size();
		pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudWorld(new pcl::PointCloud<pcl::PointXYZI>());
		laserCloudWorld->reserve(size);
		pcl::PointXYZI point_tf;
		for (int i = 0; i < size; i++)
		{
			const PointType &point = feats_undistort->points[i];
			if (point.label < 0) continue;

			V3D point_L(point.x, point.y, point.z);
			V3D point_W(state_point.rot * (state_point.offset_R_L_I * point_L + state_point.offset_T_L_I) + state_point.pos);
			point_tf.x = point_W(0);
			point_tf.y = point_W(1);
			point_tf.z = point_W(2);
			point_tf.intensity = point.label;
			laserCloudWorld->emplace_back(point_tf);
		}
		sensor_msgs::PointCloud2 laserCloudmsg;
		pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
		laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
		laserCloudmsg.header.frame_id = "camera_init";
		pubLaserCloudFull.publish(laserCloudmsg);
	}
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
	sensor_msgs::PointCloud2 laserCloudMap;
	pcl::toROSMsg(*featsFromMap, laserCloudMap);
	laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
	laserCloudMap.header.frame_id = "camera_init";
	pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
	out.pose.position.x = state_point.pos(0);
	out.pose.position.y = state_point.pos(1);
	out.pose.position.z = state_point.pos(2);
	out.pose.orientation.x = geoQuat.x;
	out.pose.orientation.y = geoQuat.y;
	out.pose.orientation.z = geoQuat.z;
	out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
	odomAftMapped.header.frame_id = "camera_init";
	odomAftMapped.child_frame_id = "body";
	odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
	set_posestamp(odomAftMapped.pose);
	pubOdomAftMapped.publish(odomAftMapped);
	auto P = kf.get_P();
	for (int i = 0; i < 6; i ++)
	{
		int k = i < 3 ? i + 3 : i - 3;
		odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
		odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
		odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
		odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
		odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
		odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
	}

	static tf::TransformBroadcaster br;
	tf::Transform transform;
	tf::Quaternion q;
	transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z));
	q.setW(odomAftMapped.pose.pose.orientation.w);
	q.setX(odomAftMapped.pose.pose.orientation.x);
	q.setY(odomAftMapped.pose.pose.orientation.y);
	q.setZ(odomAftMapped.pose.pose.orientation.z);
	transform.setRotation( q );
	br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
	set_posestamp(msg_body_pose);
	msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
	msg_body_pose.header.frame_id = "camera_init";

	/*** if path is too large, the rvis will crash ***/
	static int jjj = 0;
	jjj++;
	if (jjj % 10 == 0) 
	{
		path.poses.push_back(msg_body_pose);
		pubPath.publish(path);
	}
}

void computePlaneParamsBody(const POINTCLOUD &cloud, HessianPlane &plane)
{
	M3D moment = M3D::Zero();
	V3D sum = V3D::Zero();
	double weight = 0.0;
	for (const auto &point : cloud.points)
	{
		V3D point_L(point.x, point.y, point.z);
		V3D point_I(state_point.offset_R_L_I * point_L + state_point.offset_T_L_I);                 
		double w = 1.0 / (point.covariance[0] + point.covariance[4] + point.covariance[8] + pow(point.intensity, 2));
		moment += w * point_I * point_I.transpose();
		sum += w * point_I;
		weight += w;
	}

	auto &center = plane.center;
	auto &normal = plane.normal;
	auto &d = plane.d;
	center = sum / weight;
	M3D scatter = moment - center * sum.transpose();
	Eigen::SelfAdjointEigenSolver<M3D> solver(scatter);
	normal = solver.eigenvectors().col(0);
	d = center.dot(normal);
	if (d > 0) // normal should point to the origin in local frame
	{
		normal = -normal;
		d = -d;
	}
	M3D H_nn = -moment + normal.dot(scatter * normal) * Eigen::Matrix3d::Identity();
	M3D H_nn_d = H_nn + center * sum.transpose();
	Eigen::JacobiSVD<M3D> decoupledSolver(H_nn_d, Eigen::ComputeFullU | Eigen::ComputeFullV); 
	M3D decoupledSV_inv = M3D::Zero();
	for (int i = 0; i < 2; ++i)
	{
		decoupledSV_inv(i,i) = 1.0 / decoupledSolver.singularValues()[i];
	}
	plane.D_nn = -decoupledSolver.matrixV() * decoupledSV_inv * (decoupledSolver.matrixU().transpose());

	M3D H_nn_i = H_nn.inverse();
	double upper = -normal.dot(H_nn_i * normal);
	double lower = normal.dot(H_nn_i * sum);
	plane.D_dd = upper / (lower * lower);
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
	collinearOriIndex2.clear();
	closestCluster2.clear();

	/** Plane-to-Plane Correspondences **/
	tbb::parallel_for(tbb::blocked_range<size_t>(0, planeNum), [&](const tbb::blocked_range<size_t>& r)
	{
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			// transform plane parameters
			for (size_t j = 0; j < sparsePlanes[i]->size(); ++j)
			{
				const auto &point = sparsePlanes[i]->points[j];
				auto &point_tf = wrldPlanes[i].points[j];
				V3D point_L(point.x, point.y, point.z);
				V3D point_W = s.rot * (s.offset_R_L_I * point_L + s.offset_T_L_I) + s.pos;
				point_tf.x = point_W.x();
				point_tf.y = point_W.y();
				point_tf.z = point_W.z();     
			}
			// find correspondences
			if (ekfom_data.converge)
			{
				V3D transformedNormal = s.rot * planeParams[i].normal;
				V3D transformedCenter = s.rot * planeParams[i].center + s.pos;
				M3D crossCenter;
				crossCenter << SKEW_SYM_MATRX(transformedCenter);
				Eigen::Matrix<double, 3, 6> mat_temp;
				mat_temp.topLeftCorner<3, 3>() = -s.rot.toRotationMatrix() * crossCenter;
				mat_temp.topRightCorner<3, 3>() = M3D::Identity();
				closestPlane[i] = p_map->getNearestPlaneID(wrldPlanes[i], transformedNormal, transformedCenter, mat_temp * cov_pr * mat_temp.transpose());
			}
		}
	});

	/** Point-to-Point Correspondences **/
	wrldSinglePoints->resize(sparseSinglePoints->size());
	for (size_t i = 0; i < sparseSinglePoints->size(); ++i)
	{
		const auto &point = sparseSinglePoints->points[i];
		auto &point_tf = wrldSinglePoints->points[i];
		V3D point_L(point.x, point.y, point.z);
		V3D point_W(s.rot * (s.offset_R_L_I * point_L + s.offset_T_L_I) + s.pos);
		point_tf.x = point_W.x();
		point_tf.y = point_W.y();
		point_tf.z = point_W.z();
	}

	if (ekfom_data.converge)
	{
		int trialNum = 0;
		std::tie(trialNum, singleOriIndex, closestPoint) = p_map->getNearestPoint(*wrldSinglePoints, T_dist * T_dist);
 	}

	/** Point-to-Plane Correspondences **/
	POINTCLOUD_PTR extendedLinearPoints(new POINTCLOUD(*sparseLinearPoints));
	planeToPlaneNum = 0;
	for (int i = 0; i < closestPlane.size(); ++i)
	{
		if (closestPlane[i] == -1)
		{
			*extendedLinearPoints += *sparsePlanes[i];
			continue;
		}
		++planeToPlaneNum;
	}

	wrldLinearPoints->resize(extendedLinearPoints->size());
	for (size_t i = 0; i < extendedLinearPoints->size(); ++i)
	{
		const auto &point = extendedLinearPoints->points[i];
		auto &point_tf = wrldLinearPoints->points[i];
		V3D point_L(point.x, point.y, point.z);
		V3D point_W(s.rot * (s.offset_R_L_I * point_L + s.offset_T_L_I) + s.pos);
		point_tf.x = point_W.x();
		point_tf.y = point_W.y();
		point_tf.z = point_W.z();
		point_tf.label = point.label;
	}

	if (ekfom_data.converge)
	{
		std::tie(collinearOriIndex, closestCluster) = p_map->getNearestPatch(*wrldLinearPoints, T_dist * T_dist);
		extendedSinglePoints.reset(new POINTCLOUD(*sparseSinglePoints));
		for (size_t i = 0; i < collinearOriIndex.size(); ++i)
		{
			int idx = collinearOriIndex[i];
			if (closestCluster[i].size() == 1)
			{
				singleOriIndex.emplace_back(static_cast<int>(extendedSinglePoints->size()));
				closestPoint.emplace_back(closestCluster[i][0]);
				extendedSinglePoints->emplace_back(extendedLinearPoints->points[idx]);
			}
		}
	}

	for (size_t i = 0; i < collinearOriIndex.size(); ++i)
	{
		int idx = collinearOriIndex[i];
		if (closestCluster[i].size() == 1)
		{
			wrldSinglePoints->emplace_back(wrldLinearPoints->points[idx]);
			continue;
		}

		M3D moment = M3D::Zero();
		V3D sum = V3D::Zero();
		std::vector<double> weights(5);
		for (int j = 0; j < 5; ++j)
		{
			const V3D &vec = closestCluster[i][j].head(3);
			double w = 1.0 / closestCluster[i][j](3);
			moment += w * vec * vec.transpose();
			sum += w * vec;
			weights[j] = w;
		}
		double weight = std::accumulate(weights.begin(), weights.end(), 0.0);
		V3D center = sum / weight;
		M3D scatter = moment - sum * center.transpose();
		Eigen::SelfAdjointEigenSolver<M3D> solver(scatter);
		if (solver.eigenvalues()[0] > 0.01 * weight)
		{
			continue;
		}
		V3D normal = solver.eigenvectors().col(0);
		double d = center.dot(normal);
		int planeID = extendedLinearPoints->points[collinearOriIndex[i]].label;
		if (planeID >= 0 && fabs(normal.dot(s.rot * planeParams[planeID].normal)) < 0.9)
		{
			continue;
		}
		const auto &p_body = extendedLinearPoints->points[idx];
		const auto &p_wrld = wrldLinearPoints->points[idx];
		float pd2 = normal(0) * p_wrld.x + normal(1) * p_wrld.y + normal(2) * p_wrld.z - d;
		float scale = 1 - 0.9 * fabs(pd2) / sqrt(p_body.range);
		if (flg_deskew && scale < 0.9)
		{
			continue;
		}
		M3D dn_dp = M3D::Zero();
		Eigen::Matrix<double, 3, 6> mat_F;
		Eigen::Matrix<double, 6, 6> mat_C = Eigen::Matrix<double, 6, 6>::Zero();
		const M3D &mat_U = solver.eigenvectors();
		const V3D &u_1 = mat_U.col(2);
		const V3D &u_2 = mat_U.col(1);
		for (int j = 0; j < 5; ++j)
		{
			const V3D &vec = closestCluster[i][j].head(3);
			M3D cov = (closestCluster[i][j](3) / 3.0) * M3D::Identity();
			double w = weights[j] / weight;
			dn_dp.row(0) = w * (vec - center).transpose() * (u_1 * normal.transpose() + normal * u_1.transpose()) / (solver.eigenvalues()[0] - solver.eigenvalues()[2]);
			dn_dp.row(1) = w * (vec - center).transpose() * (u_2 * normal.transpose() + normal * u_2.transpose()) / (solver.eigenvalues()[0] - solver.eigenvalues()[1]);
			mat_F.topLeftCorner<3, 3>() = mat_U * dn_dp;
			mat_F.topRightCorner<3, 3>() = w * M3D::Identity();
			mat_C += mat_F.transpose() * cov * mat_F;
		}
		collinearOriIndex2.emplace_back(collinearOriIndex[i]);
		closestCluster2.emplace_back(normal, center, mat_C);
	}

	pointToPlaneNum = collinearOriIndex2.size();
	pointToPointNum = singleOriIndex.size();
	constraintNum = 2 * planeToPlaneNum + pointToPlaneNum + pointToPointNum;
	if (constraintNum < 1)
	{
		ekfom_data.valid = false;
		ROS_WARN("No Effective Points! \n");
		return;
	}
	
	/*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
	ekfom_data.n_gp = planeToPlaneNum;
	ekfom_data.h_x = MatrixXd::Zero(constraintNum, 12);
	ekfom_data.h.resize(constraintNum);
	ekfom_data.R.resize(constraintNum, 1);
	
	int count = 0;
	for (size_t i = 0; i < closestPlane.size(); ++i)
	{
		auto normalResult = p_map->getPlaneNormal(closestPlane[i]);
		auto dResult = p_map->getPlaneD(closestPlane[i]);
		if (normalResult && dResult)
		{
			const auto &[wrldNormal, wrldD_nn] = normalResult.value();
			M3D crossNormal;
			crossNormal << SKEW_SYM_MATRX(planeParams[i].normal);
			V3D vec_dNormal_W = s.rot * planeParams[i].normal;
			V3D vec_dNormal_L = s.rot.conjugate() * wrldNormal;
			V3D vec_dR = crossNormal * vec_dNormal_L;
			ekfom_data.h_x.block<1, 12>(count, 0) << 0.0, 0.0, 0.0, VEC_FROM_ARRAY(vec_dR), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			ekfom_data.h(count) = 1 - wrldNormal.dot(s.rot * planeParams[i].normal);
			ekfom_data.R(count) = 1.0 / (vec_dNormal_W.dot(wrldD_nn * vec_dNormal_W) + vec_dNormal_L.dot(planeParams[i].D_nn * vec_dNormal_L));
			++count;
			
			const auto &[wrldD, wrldD_dd] = dResult.value();
			ekfom_data.h_x.block<1, 12>(count, 0) << VEC_FROM_ARRAY(wrldNormal), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			ekfom_data.h(count) = wrldD - planeParams[i].d - wrldNormal.dot(s.pos);
			ekfom_data.R(count) = 1.0 / (wrldD_dd + s.pos.dot(wrldD_nn * s.pos) + planeParams[i].D_dd);
			++count;
		}
	}
	
	for (int i = 0; i < pointToPlaneNum; ++i)
	{
		PointType &point = extendedLinearPoints->points[collinearOriIndex2[i]];
		const PointType &point_tf = wrldLinearPoints->points[collinearOriIndex2[i]];
		V3D point_L(point.x, point.y, point.z);
		V3D point_I = s.offset_R_L_I * point_L + s.offset_T_L_I;
		V3D point_W(point_tf.x, point_tf.y, point_tf.z);
		M3D crossMat_I;
		crossMat_I << SKEW_SYM_MATRX(point_I);
		const V3D &normal = get<0>(closestCluster2[i]);
		const V3D &center = get<1>(closestCluster2[i]);
		const Eigen::Matrix<double, 6, 6> & mat_C = get<2>(closestCluster2[i]);
		V3D A(crossMat_I * s.rot.conjugate() * normal);
		Eigen::Matrix<double, 6, 1> B;
		B << point_W - center, -normal;
		Eigen::Map<M3D> cov_point(point.covariance);

		ekfom_data.h_x.block<1, 12>(count, 0) << VEC_FROM_ARRAY(normal), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		ekfom_data.h(count) = normal.dot(center - point_W);
		ekfom_data.R(count) = 1.0 / (normal.dot(cov_point * normal) + B.dot(mat_C * B));
		++count;
	}

	for (int i = 0; i < pointToPointNum; ++i)
	{
		int idx = singleOriIndex[i];
		const PointType& point = extendedSinglePoints->points[idx];
		const PointType& point_tf = wrldSinglePoints->points[idx];
		V3D point_L(point.x, point.y, point.z);
		V3D point_I = s.offset_R_L_I * point_L + s.offset_T_L_I;
		M3D crossMat_I;
		crossMat_I << SKEW_SYM_MATRX(point_I);
		V3D vec_diff(point_tf.x - closestPoint[i].x(), point_tf.y - closestPoint[i].y(), point_tf.z - closestPoint[i].z());
		V3D vec_diff_unit(vec_diff.normalized());
		V3D A(crossMat_I * s.rot.conjugate() * vec_diff_unit);

		ekfom_data.h_x.block<1, 12>(count, 0) << VEC_FROM_ARRAY(vec_diff_unit), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		ekfom_data.h(count) = -vec_diff.norm();
		ekfom_data.R(count) = 1.0 / (point.covariance[0] + point.covariance[4] + point.covariance[8] + closestPoint[i](3));
		++count;
	}
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	nh.param<int>("lidar/scan_line", scanNum, 16);
	nh.param<int>("mapping/max_points_per_voxel", max_points_per_voxel, 20);
	nh.param<int>("publish/interval", pcd_save_interval, -1);
	nh.param<bool>("preprocess/deskew_en", flg_deskew, true);
	nh.param<bool>("mapping/time_sync_en", time_sync_en, false);
	nh.param<bool>("publish/path_en", path_en, true);
	nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
	nh.param<bool>("publish/pcd_save_en", pcd_save_en, false);
	nh.param<float>("lidar/range_sigma", rangeSigma, 0.02);
	nh.param<float>("lidar/bearing_sigma", bearingSigma, 0.01);
	nh.param<float>("preprocess/resolution", resolution, 0.2);
	nh.param<string>("imu/imu_topic", imu_topic, "/livox/imu");
	nh.param<double>("lidar/maximum_range", detection_range, 100.0);
	nh.param<double>("imu/corr_time", corr_time, 1.0);
	nh.param<double>("mapping/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
	nh.param<double>("mapping/voxel_size", voxel_size, 1.0);
	nh.param<vector<double>>("imu/gyr_rw", gyr_rw, vector<double>());
	nh.param<vector<double>>("imu/acc_rw", acc_rw, vector<double>());
	nh.param<vector<double>>("imu/b_gyr_std", b_gyr_std, vector<double>());
	nh.param<vector<double>>("imu/b_acc_std", b_acc_std, vector<double>());
	nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
	nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
	
	path.header.stamp = ros::Time::now();
	path.header.frame_id = "camera_init";

	/*** variables definition ***/
	float mapUpdateStep = voxel_size / sqrt(static_cast<float>(max_points_per_voxel));
	T_dist = 0.5 * mapUpdateStep + 3.0 * rangeSigma;
	ROS_INFO("Distance threshold = %f", T_dist);

	Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
	Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
	V3D gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
	memcpy(gyr_cov.data(),   &gyr_rw[0],    3 * sizeof(double));
  memcpy(acc_cov.data(),   &acc_rw[0],    3 * sizeof(double));
  memcpy(b_gyr_cov.data(), &b_gyr_std[0], 3 * sizeof(double));
  memcpy(b_acc_cov.data(), &b_acc_std[0], 3 * sizeof(double));
	corr_time *= 3600;

	p_pre = std::make_unique<moi::LineBasedDouglasPeucker>(scanNum, resolution);
	p_pre->setNoise(rangeSigma, bearingSigma);
	p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
	p_imu->set_gyr_cov((gyr_cov * (M_PI / 180.0) / 60.0).cwiseAbs2());
	p_imu->set_acc_cov((acc_cov / 60.0).cwiseAbs2());
	p_imu->set_gyr_bias_cov((b_gyr_cov * (M_PI / 180.0) / 3600.0).cwiseAbs2(), corr_time);
	p_imu->set_acc_bias_cov((b_acc_cov * 1e-5).cwiseAbs2(), corr_time);
	p_imu->set_deskew_flag(flg_deskew);
	p_map = std::make_unique<PlaneOccupiedVoxelMap>(voxel_size, detection_range, max_points_per_voxel);

	for (int i = 0; i < MAX_PLANE; ++i)
	{
		fullPlanes[i]   = boost::make_shared<POINTCLOUD>();
		densePlanes[i]  = boost::make_shared<POINTCLOUD>();
		sparsePlanes[i] = boost::make_shared<POINTCLOUD>();
	}

	int maxIterNum = 3;
	double epsi[23] = {0.001};
	fill(epsi, epsi + 23, 0.001);
	kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, maxIterNum, epsi);

	/*** ROS subscriber initialization ***/
	ros::Subscriber sub_pcl = nh.subscribe("/laser_scan", 10, standard_pcl_cbk);
	ros::Subscriber sub_imu = nh.subscribe(imu_topic, 2000, imu_cbk);

	ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 10);
	ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 10);
	ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 10);
	ros::Publisher pubPath = nh.advertise<nav_msgs::Path> ("/path", 10);

//------------------------------------------------------------------------------------------------------
	signal(SIGINT, SigHandle);
	ros::Rate rate(5000);
	bool status = ros::ok();
	while (status)
	{
		if (flg_exit) break;
		ros::spinOnce();
		if(sync_packages(Measures))
		{
			if (flg_first_scan)
			{
				first_lidar_time = Measures.lidar_beg_time;
				p_imu->first_lidar_time = first_lidar_time;
				flg_first_scan = false;
				continue;
			}

			double t0, t1, t2, t3, t4, t5;
			planeExtractionTime += Measures.plane_cost_time;
			planeTimeSquares += Measures.plane_cost_time * Measures.plane_cost_time;;
			/*** undistort current scan using IMU ***/
			t0 = omp_get_wtime();
			p_imu->Process(Measures, kf, feats_undistort);
			state_point = kf.get_x();
			pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
			cov_pr = kf.get_P().topLeftCorner<6, 6>();
			if (cov_pr.trace() > 1) cov_pr.setZero();
			if(feats_undistort->size() < 100)
			{
				ROS_WARN("No point, skip this scan!\n");
				continue;
			}

			/*** segment ***/
			t1 = omp_get_wtime();
			std::unordered_map<int16_t, POINTCLOUD_PTR> segmentedClouds;
			for (const auto& point : feats_undistort->points)
			{
				int16_t label = point.label;
				if (segmentedClouds.find(label) == segmentedClouds.end())
				{
					segmentedClouds[label] = boost::make_shared<POINTCLOUD>();
				}
				segmentedClouds[label]->emplace_back(point);
			}

			POINTCLOUD_PTR denseSinglePoints(new POINTCLOUD());	
			POINTCLOUD_PTR denseLinearPoints(new POINTCLOUD());
			POINTCLOUD_PTR densePlanarPoints(new POINTCLOUD());			
			planeNum = 0;
			for (const auto &[label, segmentedCloud] : segmentedClouds)
			{
				switch (label)
				{
					case -2:
						denseSinglePoints = boost::move(segmentedCloud);
						break;
					case -1:
						denseLinearPoints = boost::move(segmentedCloud);
						break;
					default:
						fullPlanes[label] = boost::move(segmentedCloud);
						++planeNum;
						break;
				}
			}
			std::sort(fullPlanes.begin(), fullPlanes.begin() + planeNum, [](const POINTCLOUD_PTR& a, const POINTCLOUD_PTR& b) { return a->size() < b->size(); });
			pointClusters.resize(planeNum);
			closestPlane.assign(planeNum, -1);
			/*** downsample ***/
			downsampleByVoxelGrid(*denseSinglePoints, *sparseSinglePoints, 0.5 * voxel_size);
			downsampleByVoxelGrid(*sparseSinglePoints, *denseSinglePoints, 1.5 * voxel_size);
			sparseSinglePoints->swap(*denseSinglePoints);

			downsampleByVoxelGrid(*denseLinearPoints, *sparseLinearPoints, 0.5 * voxel_size);
			downsampleByVoxelGrid(*sparseLinearPoints, *denseLinearPoints, 1.5 * voxel_size);
			sparseLinearPoints->swap(*denseLinearPoints);

			for (int i = 0; i < planeNum; ++i)
			{
				for (auto & point : fullPlanes[i]->points) { point.label = i; }
				downsampleByVoxelGrid(*fullPlanes[i], *densePlanes[i], 0.5 * voxel_size);
				downsampleByVoxelGrid(*densePlanes[i], *sparsePlanes[i], 1.5 * voxel_size);

				*densePlanarPoints += *densePlanes[i];
				computePlaneParamsBody(*fullPlanes[i], planeParams[i]);
				wrldPlanes[i].resize(sparsePlanes[i]->size());
			}
			*densePlanarPoints += *denseLinearPoints;
			*densePlanarPoints += *denseSinglePoints;

			t2 = omp_get_wtime();
			preprocessingTime += t2 - t0;
			if (p_map->empty() == false)
			{
				/*** iterated state estimation ***/
				kf.update_iterated_dyn_share_diagonal();
				state_point = kf.get_x();
				pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
				cov_pr = kf.get_P().topLeftCorner<6, 6>();
				if (cov_pr.trace() > 1) cov_pr.setZero();
				geoQuat.x = state_point.rot.coeffs()[0];
				geoQuat.y = state_point.rot.coeffs()[1];
				geoQuat.z = state_point.rot.coeffs()[2];
				geoQuat.w = state_point.rot.coeffs()[3];
			}

			/*** Update the voxel map ***/
			tbb::parallel_for(tbb::blocked_range<size_t>(0, planeNum), [&](const tbb::blocked_range<size_t>& r)
			{
				for (size_t i = r.begin(); i != r.end(); ++i)
				{
					pointClusters[i].setZero(); // in world frame, consider pose uncertainty
					for (size_t j = 0; j < fullPlanes[i]->size(); ++j)
					{
						auto &point = fullPlanes[i]->points[j];
						V3D point_L(point.x, point.y, point.z);
						V3D point_I = state_point.offset_R_L_I * point_L + state_point.offset_T_L_I;
						V3D point_W = state_point.rot * point_I + state_point.pos;

						Eigen::Map<Eigen::Matrix3d> cov_point(point.covariance);                       
						M3D crossMat_I;
						crossMat_I << SKEW_SYM_MATRX(point_I);
						Eigen::Matrix<double, 3, 6> jacobiMat;
						jacobiMat.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();
						jacobiMat.topRightCorner<3, 3>() = -state_point.rot.toRotationMatrix() * crossMat_I;
						cov_point += jacobiMat * cov_pr * jacobiMat.transpose();                                

						double weight = 1.0 / (cov_point.trace() + pow(point.intensity, 2));
						pointClusters[i].topLeftCorner<3, 3>() += weight * point_W * point_W.transpose();
						pointClusters[i].topRightCorner<3, 1>() += weight * point_W;
						pointClusters[i](3, 3) += weight;
					}
					pointClusters[i].bottomLeftCorner<1, 3>() = pointClusters[i].topRightCorner<3, 1>().transpose();
				}
			});
			t3 = omp_get_wtime();
			solvingTime += t3 - t2;
			// * Update plane attributes 
			p_map->updatePlaneInfoBatch(closestPlane, pointClusters, pos_lid);
			// * Update voxel labels
			transformBodyToWorld(densePlanarPoints);
			p_map->updateLabel(*densePlanarPoints, closestPlane);
			// * Update local map
			if (p_map->empty() || (pos_lid - position_last).norm() > mapUpdateStep)
			{
				position_last = pos_lid;
				POINTCLOUD_PTR pointsToUpdateMap(new POINTCLOUD());
				downsampleByVoxelGrid(*densePlanarPoints, *pointsToUpdateMap, 0.5 * voxel_size);
				p_map->updateMap(*pointsToUpdateMap);
				p_map->removePointsFarFromLocation(pos_lid);
			}

			t4 = omp_get_wtime();
			//ROS_INFO("plane-to-plane: %d (%d) / point-to-plane: %d / point-to-point: %d", planeToPlaneNum, planeNum, pointToPlaneNum, pointToPointNum);
			//ROS_INFO("Processing current scan uses %f = %f + %f + %f + %f s.", t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3);
			mapUpdateTime += t4 - t3;
			totalTimeSquares += (Measures.plane_cost_time + t4 - t0) * (Measures.plane_cost_time + t4 - t0);
			/******* Publish odometry *******/
			publish_odometry(pubOdomAftMapped);
			if (scan_num % 2 == 0)
			{
				p_map->getPointCloud(*featsFromMap);
				publish_map(pubLaserCloudMap);
			}

			/******* Publish points *******/
			if (path_en) publish_path(pubPath);
			if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFull);
		}

		status = ros::ok();
		rate.sleep();
	}
	
	// Time performance
	double meanPlaneTime = planeExtractionTime / (double)scan_num;
	double meanTotalTime = (planeExtractionTime + preprocessingTime + solvingTime + mapUpdateTime) / (double)scan_num;
	double stdPlaneTime = sqrt(planeTimeSquares / (double)scan_num - meanPlaneTime * meanPlaneTime);
	double stdTotalTime = sqrt(totalTimeSquares / (double)scan_num - meanTotalTime * meanTotalTime);
	ROS_INFO("%f - %f s per scan on average (plane extraction: %f - %f).", meanTotalTime, stdTotalTime, meanPlaneTime, stdPlaneTime);
	/**************** save map ****************/
	/* 1. make sure you have enough memories
	/* 2. pcd save will largely influence the real-time performences **/
	if (pcl_wait_save->size() > 0 && pcd_save_en)
	{
		string file_name = string("scans.pcd");
		string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
		pcl::PCDWriter pcd_writer;
		cout << "current scan saved to /PCD/" << file_name << endl;
		pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
	}

	return 0;
}
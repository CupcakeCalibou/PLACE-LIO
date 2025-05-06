#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <place_lio/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>

using namespace std;
using namespace Eigen;

#define USE_IKFOM

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)         // Gravaty const in GuangDong/China
#define DIM_STATE (18)        // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12)       // Dimension of process noise (Let Dim(SO(3)) = 3)
#define INIT_COV   (1)
#define NUM_MATCH_POINTS    (5)
#define MAX_MEAS_DIM        (10000)

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define CONSTRAIN(v,min,max)     ((v>min)?((v<max)?v:max):min)
#define ARRAY_FROM_EIGEN(mat)    mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)  vector<decltype(mat)::Scalar> (mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name)     (string(string(ROOT_DIR) + "Log/"+ name))

typedef place_lio::Pose6D Pose6D;
typedef Vector3d V3D;
typedef Vector4d V4D;
typedef Matrix3d M3D;
typedef Matrix4d M4D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

#define MD(a,b)  Matrix<double, (a), (b)>
#define VD(a)    Matrix<double, (a), 1>
#define MF(a,b)  Matrix<float, (a), (b)>
#define VF(a)    Matrix<float, (a), 1>

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

// 自定义点云类型，包含各点所在扫描线与时间戳
struct PointWithStamp
{
  PCL_ADD_POINT4D;
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float range;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointWithStamp,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
  (float, range, range)
  (std::uint16_t, ring, ring)
  (double, time, time)
)

struct PointWithCovariance
{
  PCL_ADD_POINT4D;
  PCL_ADD_INTENSITY;
  double covariance[9];
  float range;
  int16_t label;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointWithCovariance,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
  (float, range, range)
  (double[9], covariance, covariance)
  (std::int16_t, label, label)
  (double, time, time)
)

typedef PointWithCovariance PointType;
typedef pcl::PointCloud<PointType> POINTCLOUD;
typedef pcl::PointCloud<PointType>::Ptr POINTCLOUD_PTR;
typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

struct MeasureGroup     // Lidar data and imu dates for the curent process
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        this->lidar.reset(new POINTCLOUD());
    };
    double lidar_beg_time;
    double lidar_end_time;
    double plane_cost_time;
    POINTCLOUD_PTR lidar;
    deque<sensor_msgs::Imu::ConstPtr> imu;
};

struct StatesGroup
{
    StatesGroup()
    {
		this->rot_end = M3D::Identity();
		this->pos_end = Zero3d;
        this->vel_end = Zero3d;
        this->bias_g  = Zero3d;
        this->bias_a  = Zero3d;
        this->gravity = Zero3d;
        this->cov     = MD(DIM_STATE,DIM_STATE)::Identity() * INIT_COV;
        this->cov.block<9,9>(9,9) = MD(9,9)::Identity() * 0.00001;
	};

    StatesGroup(const StatesGroup& b)
    {
		this->rot_end = b.rot_end;
		this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g  = b.bias_g;
        this->bias_a  = b.bias_a;
        this->gravity = b.gravity;
        this->cov     = b.cov;
	};

    StatesGroup& operator=(const StatesGroup& b)
	{
        this->rot_end = b.rot_end;
		this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g  = b.bias_g;
        this->bias_a  = b.bias_a;
        this->gravity = b.gravity;
        this->cov     = b.cov;
        return *this;
	};

    StatesGroup operator+(const Matrix<double, DIM_STATE, 1> &state_add)
	{
        StatesGroup a;
		a.rot_end = this->rot_end * Exp(state_add(0,0), state_add(1,0), state_add(2,0));
		a.pos_end = this->pos_end + state_add.block<3,1>(3,0);
        a.vel_end = this->vel_end + state_add.block<3,1>(6,0);
        a.bias_g  = this->bias_g  + state_add.block<3,1>(9,0);
        a.bias_a  = this->bias_a  + state_add.block<3,1>(12,0);
        a.gravity = this->gravity + state_add.block<3,1>(15,0);
        a.cov     = this->cov;
		return a;
	};

    StatesGroup& operator+=(const Matrix<double, DIM_STATE, 1> &state_add)
	{
        this->rot_end = this->rot_end * Exp(state_add(0,0), state_add(1,0), state_add(2,0));
		this->pos_end += state_add.block<3,1>(3,0);
        this->vel_end += state_add.block<3,1>(6,0);
        this->bias_g  += state_add.block<3,1>(9,0);
        this->bias_a  += state_add.block<3,1>(12,0);
        this->gravity += state_add.block<3,1>(15,0);
		return *this;
	};

    Matrix<double, DIM_STATE, 1> operator-(const StatesGroup& b)
	{
        Matrix<double, DIM_STATE, 1> a;
        M3D rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3,1>(0,0)  = Log(rotd);
        a.block<3,1>(3,0)  = this->pos_end - b.pos_end;
        a.block<3,1>(6,0)  = this->vel_end - b.vel_end;
        a.block<3,1>(9,0)  = this->bias_g  - b.bias_g;
        a.block<3,1>(12,0) = this->bias_a  - b.bias_a;
        a.block<3,1>(15,0) = this->gravity - b.gravity;
		return a;
	};

    void resetpose()
    {
        this->rot_end = M3D::Identity();
		this->pos_end = Zero3d;
        this->vel_end = Zero3d;
    }

	M3D rot_end;      // the estimated attitude (rotation matrix) at the end lidar point
    V3D pos_end;      // the estimated position at the end lidar point (world frame)
    V3D vel_end;      // the estimated velocity at the end lidar point (world frame)
    V3D bias_g;       // gyroscope bias
    V3D bias_a;       // accelerator bias
    V3D gravity;      // the estimated gravity acceleration
    Matrix<double, DIM_STATE, DIM_STATE>  cov;     // states covariance
};

struct HessianPlane
{   
    V3D center;
    V3D normal;
    M3D D_nn;
    double d;
    double D_dd;
};

template<typename T>
T rad2deg(T radians)
{
  return radians * 180.0 / PI_M;
}

template<typename T>
T deg2rad(T degrees)
{
  return degrees * PI_M / 180.0;
}

template<typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g, \
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    return move(rot_kp);
}

#endif
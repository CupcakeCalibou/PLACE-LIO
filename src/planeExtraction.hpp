#ifndef PLANE_EXTRACTION_HPP_
#define PLANE_EXTRACTION_HPP_

#include <common_lib.h>
#include <opencv2/core/core.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pov-Map/pov_Map.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace moi
{
typedef std::vector<std::deque<int>> DPatches;

struct Collar 
{
	Collar(int _row_1, int _row_2, int _row_3)
	{
		row_1 = _row_1;
		row_2 = _row_2;
		row_3 = _row_3;
	}

	int row_1;
	int row_2;
	int row_3;
};

struct Plane
{
	Plane() = default;
	Plane(size_t cloud_size)
	{
		size = 0;
		d = 0.0;
		vec_n.setZero();
		mat_Q.setZero();
		vec_s.setZero();
		mat_J.setZero();
		eigenvalues.setZero();
		weight_sum = 0.0;
		sigma = 0.0;

		queue_row = new int[cloud_size];
		queue_col = new int[cloud_size];
	}

	~Plane()
	{
		delete [] queue_row;
		delete [] queue_col;

		queue_row = nullptr;
		queue_col = nullptr;
	}

	Eigen::Vector3d vec_n;
	Eigen::Matrix4d mat_Q;
	Eigen::Vector3d vec_s;
	Eigen::Matrix3d mat_J;
	Eigen::Vector3d eigenvalues;
	int size;
	double d;
	double weight_sum;
	double sigma;
	int *queue_row;
	int *queue_col;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

class LineBasedDouglasPeucker
{
public:
	LineBasedDouglasPeucker(int scanNum, float resolution);
	~LineBasedDouglasPeucker();

	void setNoise(float rangeSigma, float bearingSigma);
	void process(const sensor_msgs::PointCloud2::ConstPtr &msg, POINTCLOUD_PTR &cloud, Eigen::Vector3f gv);

	int scanRate_ = 10;
	int scanNum_ = 32;
	float resolution_ = 0.2 * M_PI / 180;
	float rangeSigma_ = 0.02;
	float bearingSigma_ = 0.01;

private:
	void extract();
	void reset();
	Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& vec);
	Eigen::Matrix4d calcCovariance(const Eigen::Matrix3d& mat_C, const Eigen::Matrix3d& mat_J,
																 const Eigen::Vector3d& vec_m, const Eigen::Vector3d& vec_n, double weight_sum) const;
	double DeviationFromLine(int ind_A, int ind_B, int ind_C);
	double DeviationFromPlane(int pointInd, const Eigen::Vector3d& normal, double d, const Eigen::Matrix4d& cov);
	int OverlappedNeighbor(int neighborCol, int core_line, int& lower_bound, int& upper_bound) const;
	DPatches Douglas_Peucker(const std::deque<int>& line_group);
	float mergePlanes(int plane_1, int plane_2) const;

	// Constant
	double D2R_ = M_PI / 180;
	double R2D_ = 180 / M_PI;
	// Tuning
	float D_Po2L_  = 2.0;
	float D_L2P_   = 3.0;
	float D_Po2P_  = 2.0;
	float T_angle_ = 10 * D2R_;
	float T_ratio_ = 0.01;
	float T_count_ = 30;
	// Fixed
	float sin_resolution_;
	float cos_resolution_;
	float cos_angle_ = cos(T_angle_);
	float tan_angle_ = tan(T_angle_);
	float cos_80_ = cos(80 * D2R_);
	float cos_85_ = cos(85 * D2R_);
	int update_step_ = 5;
	int buffer_size_ = 20;

	POINTCLOUD_PTR currentCloudPtr_;
	cv::Mat indexMat_;
	cv::Mat lineMat_;
	cv::Mat planeMat_;
	float *pWeight_;

	std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> lSum_;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> lMatrix_;
	std::vector<double> lWeight_;
	std::vector<std::vector<std::pair<int, int>>> lPoints_;
	std::vector<std::vector<int>> lPos_1d_;
	std::vector<std::pair<int, int>> lPos_2d_;
	std::vector<std::pair<int, float>> lPriority_;

	std::vector<int8_t> line_iter_;
	std::vector<std::pair<int8_t, int8_t>> point_iter_;
	std::vector<Plane, Eigen::aligned_allocator<Plane>> planes_;
};

LineBasedDouglasPeucker::LineBasedDouglasPeucker(int scanNum, float resolution)
	:scanNum_(scanNum), resolution_(resolution)
{
	currentCloudPtr_.reset(new POINTCLOUD());
	int columnNum = static_cast<int>(std::ceil(360 / resolution));
	int approxPoints = scanNum_ * columnNum;
	int approxLines = approxPoints / 3;
	int approxPlanes = approxPoints / T_count_;
	buffer_size_ = static_cast<int>(5.0 / resolution);
	resolution_ *= D2R_;
	sin_resolution_ = sin(resolution_);
	cos_resolution_ = cos(resolution_);

	indexMat_ = cv::Mat(scanNum_, columnNum, CV_32S);
	lineMat_  = cv::Mat(scanNum_, columnNum, CV_32S);
	planeMat_ = cv::Mat(scanNum_, columnNum, CV_32S);
	pWeight_ = new float[approxPoints];

	lSum_.reserve(approxLines);
	lMatrix_.reserve(approxLines);
	lWeight_.reserve(approxLines);
	lPoints_.reserve(approxLines);
	lPos_1d_.resize(columnNum);
	lPos_2d_.reserve(approxLines);
	lPriority_.reserve(approxLines);

	planes_.reserve(approxPlanes);

	line_iter_ = {1};
	point_iter_.emplace_back(std::make_pair(-1,  0));
	point_iter_.emplace_back(std::make_pair( 0,  1));
	point_iter_.emplace_back(std::make_pair( 0, -1));
	point_iter_.emplace_back(std::make_pair( 1,  0));
	point_iter_.emplace_back(std::make_pair( 0,  2));
	point_iter_.emplace_back(std::make_pair( 0, -2));
}

LineBasedDouglasPeucker::~LineBasedDouglasPeucker() {}

void LineBasedDouglasPeucker::setNoise(float rangeSigma, float bearingSigma)
{
	rangeSigma_ = rangeSigma;
	bearingSigma_ = bearingSigma;
}

void LineBasedDouglasPeucker::process(const sensor_msgs::PointCloud2::ConstPtr &msg, POINTCLOUD_PTR &cloud, Eigen::Vector3f gv = Eigen::Vector3f::Zero())
{
	pcl::PointCloud<PointWithStamp> cloudIn;
  pcl::fromROSMsg(*msg, cloudIn);
	int cloudSize = cloudIn.size();
  if (cloudSize == 0)
    return;

	reset();

	bool noAvailableTime = std::isnan(cloudIn.points[cloudSize - 1].time);
	double azimuth_0 = atan2(cloudIn.points[0].y, cloudIn.points[0].x);
	/*** These variables only works when no point timestamps given ***/
  double angularVelocity = 0.361 * scanRate_;
  std::vector<bool> isFirst(scanNum_,true);
  std::vector<double> firstAzimuth(scanNum_, 0.0);
  std::vector<double> lastAzimuthDiff(scanNum_, 0.0);
	/*** These variables only works when no point timestamps given ***/
	currentCloudPtr_->reserve(cloudSize);
	PointWithCovariance point;
  for (int i = 0; i < cloudSize; i++)
  {
		const auto &pointIn = cloudIn.points[i];
		// 三维坐标 + 反射强度
		point.x = pointIn.x;
		point.y = pointIn.y;
		point.z = pointIn.z;
		// point.intensity = cloudIn.points[i].intensity;
		// 行号
		int row = pointIn.ring;
		// 列号
		float azimuth = atan2(point.y, point.x);
		float diff = azimuth_0 - azimuth;
		if (diff < 0) diff += 2 * M_PI;
		int col = int(diff / resolution_);
		// 时间戳
		if (noAvailableTime)
		{
			if (isFirst[row])
			{
				isFirst[row] = false;
				firstAzimuth[row] = azimuth;
				lastAzimuthDiff[row] = 0.0;
			}
			double azimuthDiff = firstAzimuth[row] - azimuth;
			if (azimuthDiff < 0) azimuthDiff += 2 * M_PI;
			if (azimuthDiff < lastAzimuthDiff[row]) azimuthDiff += 2 * M_PI;
			lastAzimuthDiff[row] = azimuthDiff;

			point.time = azimuthDiff * R2D_/angularVelocity;
		}
		else
		{
			point.time = pointIn.time;
		}

		/*** Only record the first projected on each pixel ***/
		auto& ind = indexMat_.at<int>(row, col);
		if (ind > -1) continue;
		ind = (int)currentCloudPtr_->size();
		/*****************************************************/

		// 测距值
		point.range = pointIn.range;
		// 曲率
		if (i < 5 || i > cloudSize - 6)
		{
			point.intensity = 5e-3;
		}
		else
		{
			float diff = cloudIn.points[i - 5].range + cloudIn.points[i - 4].range + cloudIn.points[i - 3].range + cloudIn.points[i - 2].range + cloudIn.points[i - 1].range +
									 cloudIn.points[i + 5].range + cloudIn.points[i + 4].range + cloudIn.points[i + 3].range + cloudIn.points[i + 2].range + cloudIn.points[i + 1].range -
									 cloudIn.points[i].range * 10;
			point.intensity = std::min(5e-3, 0.1 * fabs(diff) / pointIn.range);
		}		

		currentCloudPtr_->emplace_back(point);
  }

	cloudSize = currentCloudPtr_->size();
	for (int i = 0; i < cloudSize; ++i)
	{
		auto &point = currentCloudPtr_->points[i];
		// 协方差矩阵
		Eigen::Vector3d pv(point.x, point.y, point.z);
		pv.normalize();
		Eigen::Matrix3d vvt = pv * pv.transpose();
		double rangeVariance = rangeSigma_ * rangeSigma_;
		double bearingVariance = pow(bearingSigma_ * point.range * D2R_, 2);
		Eigen::Matrix3d cov = rangeVariance * vvt + bearingVariance * (Eigen::Matrix3d::Identity() - vvt);
		std::memcpy(point.covariance, cov.data(), 9 * sizeof(double));
		// 权重
		pWeight_[i] = 1.0 / (cov.trace() + pow(point.intensity, 2));
		// 标签
		point.label = -2;
	}

	extract();

	cloud.reset(new POINTCLOUD(*currentCloudPtr_));
}

void LineBasedDouglasPeucker::extract()
{
	/*** Only used for line segmentation ***/
	std::vector<float> flatness(scanNum_);
	std::vector<bool> clustered(scanNum_);
	std::vector<int> compact;
	std::vector<Collar> collars;
	std::vector<std::pair<int, int>> linePts;
	/***************************************/ 
	compact.reserve(scanNum_);
	collars.reserve(scanNum_);
	linePts.reserve(scanNum_);
	for (int i = 0; i < indexMat_.cols; ++i)
	{
		compact.clear();
		for (int j = 0; j < scanNum_; ++j)
		{
			if (indexMat_.at<int>(j, i) > -1) compact.emplace_back(j);
			else lineMat_.at<int>(j, i) = 100000;
		}
		// 点数太少不足以分割
		if (compact.size() < 3) continue;

		std::fill(flatness.begin(), flatness.end(), 0);
		collars.clear();
		for (size_t j = 1; j < compact.size() - 1; ++j)
		{
			bool isCollinear = true;
			auto &pointB = currentCloudPtr_->points[indexMat_.at<int>(compact[j - 1], i)];
			auto &pointA = currentCloudPtr_->points[indexMat_.at<int>(compact[j], i)];
			auto &pointC = currentCloudPtr_->points[indexMat_.at<int>(compact[j + 1], i)];
			Eigen::Vector3d vec_AB(pointB.x - pointA.x, pointB.y - pointA.y, pointB.z - pointA.z);
			Eigen::Vector3d vec_AC(pointC.x - pointA.x, pointC.y - pointA.y, pointC.z - pointA.z);
			Eigen::Vector3d vec_CB(vec_AB - vec_AC);

			double norm_CB = vec_CB.norm();
			Eigen::Vector3d AB_cross_AC(vec_AB.cross(vec_AC));
			double norm_cross = AB_cross_AC.norm();
			if (norm_cross > 0)
			{
				Eigen::Vector3d commonVec = AB_cross_AC / (norm_cross * norm_CB);
				Eigen::Vector3d df_dA = -skewSymmetric(vec_CB) * commonVec;
				Eigen::Vector3d df_dB = -skewSymmetric(vec_AC) * commonVec - norm_cross * vec_CB / pow(norm_CB, 3);
				Eigen::Vector3d df_dC = -skewSymmetric(vec_AB) * commonVec + norm_cross * vec_CB / pow(norm_CB, 3);

				Eigen::Map<Eigen::Matrix3d> cov_A(pointA.covariance);
				Eigen::Map<Eigen::Matrix3d> cov_B(pointB.covariance);
				Eigen::Map<Eigen::Matrix3d> cov_C(pointC.covariance);
				float Po2L = norm_cross / norm_CB;
				float sigma = sqrt(df_dA.dot(cov_A * df_dA) + df_dB.dot(cov_B * df_dB) + df_dC.dot(cov_C * df_dC));
				flatness[compact[j]] = Po2L;
				isCollinear = Po2L < D_Po2L_ * sigma;
			}
			if (isCollinear) collars.emplace_back(compact[j - 1], compact[j], compact[j + 1]);
		}

		if (collars.empty()) continue;

		linePts.clear();
		linePts.emplace_back(collars[0].row_1, i);
		linePts.emplace_back(collars[0].row_2, i);
		linePts.emplace_back(collars[0].row_3, i);

		bool popFirst = false;
		int collarNum = collars.size();
		std::fill(clustered.begin(), clustered.end(), false);
		for (int j = 0; j < collarNum; ++j)
		{
			if (clustered[j]) continue;

			// 1) Initialization
			if (popFirst) popFirst = false; // the first point was assigned to line segment	
			else linePts.emplace_back(collars[j].row_1, i);
			linePts.emplace_back(collars[j].row_2, i);
			linePts.emplace_back(collars[j].row_3, i);

			// 2) Clustering
			for (int k = j + 1; k < collarNum; ++k)
			{
				if (collars[k].row_1 == linePts[linePts.size() - 2].first && collars[k].row_2 == linePts[linePts.size() - 1].first)
				{
					linePts.emplace_back(collars[k].row_3, i);
					clustered[k] = true;
				}
				else if (collars[k].row_1 == linePts[linePts.size() - 1].first)
				{
					if (flatness[collars[j].row_2] > flatness[linePts[linePts.size() - 2].first]) popFirst = true;
					else linePts.pop_back();
					break;
				}
				else
				{
					break;
				}                 
			}

			// 3) Validation
			int lineSize = linePts.size();
			if(lineSize >= 3)
			{
				const Eigen::Vector3f &vec_front = currentCloudPtr_->points[indexMat_.at<int>(linePts.front().first, i)].getVector3fMap();
				const Eigen::Vector3f &vec_back  = currentCloudPtr_->points[indexMat_.at<int>(linePts.back().first, i)].getVector3fMap();
				Eigen::Vector3f vec_base  = vec_back - vec_front;
				float max_dist = 0;
				float baseNorm = vec_base.norm();
				for (int k = 1; k < lineSize - 1; ++k)
				{
					int ind = indexMat_.at<int>(linePts[k].first, i);
					const Eigen::Vector3f &vec_pt = currentCloudPtr_->points[ind].getVector3fMap();
					float dist = (vec_base.cross(vec_pt - vec_front)).norm() / baseNorm;
					if (dist > max_dist) max_dist = dist;
				}

				float curvature = max_dist / baseNorm;
				if (curvature <= 0.1)
				{
					int currLineInd = lSum_.size();
					lPos_2d_.emplace_back(std::make_pair(i, lPos_1d_[i].size()));
					lPos_1d_[i].emplace_back(currLineInd);

					Eigen::Vector3d vec_s = Eigen::Vector3d::Zero();
					Eigen::Matrix3d mat_J = Eigen::Matrix3d::Zero();
					double lWeight = 0;
					for (int k = 0; k < lineSize; ++k)
					{
						int ind = indexMat_.at<int>(linePts[k].first, i);
						Eigen::Vector3d vec_pt(currentCloudPtr_->points[ind].x, currentCloudPtr_->points[ind].y, currentCloudPtr_->points[ind].z);
						vec_s += pWeight_[ind] * vec_pt;
						mat_J += pWeight_[ind] * vec_pt * vec_pt.transpose();
						lWeight += pWeight_[ind];

						lineMat_.at<int>(linePts[k].first, i) = currLineInd;
						currentCloudPtr_->points[ind].label = -1;
					}

					lPriority_.emplace_back(currLineInd, (float)linePts.size());
					lSum_.emplace_back(vec_s);
					lMatrix_.emplace_back(mat_J);
					lWeight_.emplace_back(lWeight);
					lPoints_.emplace_back(linePts);
				}   
			}
			linePts.clear();   
		}
	}

	std::vector<std::pair<int, int>> lNeighbors(lSum_.size(), std::make_pair(-1, -1));
	#pragma omp parallel for num_threads(4)
	for (size_t i = 0; i < lSum_.size(); ++i) 
	{
		// middle bottom & middle top
		const auto &mb = currentCloudPtr_->points[indexMat_.at<int>(lPoints_[i].front().first, lPoints_[i].front().second)];
		const auto &mt = currentCloudPtr_->points[indexMat_.at<int>(lPoints_[i].back().first,  lPoints_[i].back().second)];
		// Left Side
		for (size_t j = 0; j < line_iter_.size(); ++j) 
		{
			int left_column = lPos_2d_[i].first - line_iter_[j];
			if (left_column < 0) break;
			int left_top = 0, left_bottom = 0;
			int leftNeighbor = OverlappedNeighbor(left_column, i, left_bottom, left_top);
			if (leftNeighbor > -1) 
			{
				// left bottom & left top
				const auto &lb = currentCloudPtr_->points[indexMat_.at<int>(lPoints_[leftNeighbor].front().first, lPoints_[leftNeighbor].front().second)];
				const auto &lt = currentCloudPtr_->points[indexMat_.at<int>(lPoints_[leftNeighbor].back().first,  lPoints_[leftNeighbor].back().second)];
				Eigen::Vector3f parallel_1(mt.x - mb.x, mt.y - mb.y, mt.z - mb.z);
				Eigen::Vector3f parallel_2(lt.x - lb.x, lt.y - lb.y, lt.z - lb.z);
				if (fabs(parallel_1.dot(parallel_2) / (parallel_1.norm()*parallel_2.norm())) < cos_angle_)
				{
					break;
				}

				bool isOccluded = true;
				for (int k = left_bottom; k <= left_top; ++k)
				{
					int leftInd = indexMat_.at<int>(k, left_column);
					int midInd = indexMat_.at<int>(k, lPos_2d_[i].first);
					if (leftInd == -1 || midInd == -1) continue;

					float d_1 = std::max(currentCloudPtr_->points[leftInd].range, currentCloudPtr_->points[midInd].range);
					float d_2 = std::min(currentCloudPtr_->points[leftInd].range, currentCloudPtr_->points[midInd].range);
					if (d_2 * sin_resolution_ / (d_1 - d_2 * cos_resolution_) > tan_angle_)
					{
						isOccluded = false;
						break;
					}
				}
				if (isOccluded) break;

				lNeighbors[i].first = leftNeighbor;
			}
		}
		// Right Side
		for (size_t j = 0; j < line_iter_.size(); ++j) 
		{
			int right_column = lPos_2d_[i].first + line_iter_[j];
			if (right_column >= indexMat_.cols) break;
			int right_top = 0, right_bottom = 0;
			int rightNeighbor = OverlappedNeighbor(right_column, i, right_bottom, right_top);
			if (rightNeighbor > -1) 
			{
				// right bottom & right top
				const auto &rb = currentCloudPtr_->points[indexMat_.at<int>(lPoints_[rightNeighbor].front().first, lPoints_[rightNeighbor].front().second)];
				const auto &rt = currentCloudPtr_->points[indexMat_.at<int>(lPoints_[rightNeighbor].back().first, lPoints_[rightNeighbor].back().second)];
				Eigen::Vector3f parallel_1(mt.x - mb.x, mt.y - mb.y, mt.z - mb.z);
				Eigen::Vector3f parallel_2(rt.x - rb.x, rt.y - rb.y, rt.z - rb.z);
				if (fabs(parallel_1.dot(parallel_2) / (parallel_1.norm() * parallel_2.norm())) < cos_angle_) 
				{
					break;
				}
				bool isOccluded = true;
				for (int k = right_bottom; k <= right_top; ++k)
				{
					int rightInd = indexMat_.at<int>(k, right_column);
					int midInd = indexMat_.at<int>(k, lPos_2d_[i].first);
					if (rightInd == -1 || midInd == -1) continue;

					float d_1 = std::max(currentCloudPtr_->points[rightInd].range, currentCloudPtr_->points[midInd].range);
					float d_2 = std::min(currentCloudPtr_->points[rightInd].range, currentCloudPtr_->points[midInd].range);
					if (d_2 * sin_resolution_ / (d_1 - d_2 * cos_resolution_) > T_angle_)
					{
						isOccluded = false;
						break;
					}
				}
				if (isOccluded) break;
				
				lNeighbors[i].second = rightNeighbor;
			}
		}
	}

	std::sort(lPriority_.begin(), lPriority_.end(), 
		[](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second > b.second; });
	bool *is_lChecked = new bool[lSum_.size()](); // 记录已经遍历过的直线段
	bool *is_lLabeled = new bool[lSum_.size()](); // 标记已构成平面的直线段
	for (size_t i = 0; i < lSum_.size(); ++i)
	{
		int core_line = lPriority_[i].first;
		if (is_lChecked[core_line])
			continue;

		is_lChecked[core_line] = true;
		std::deque<int> line_group(1, core_line);
		int curr_line = core_line;
		while (true)
		{
			if (lNeighbors[curr_line].first == -1 || is_lLabeled[lNeighbors[curr_line].first])
			{
				break;
			}
			else
			{
				curr_line = lNeighbors[curr_line].first;
				is_lChecked[curr_line] = true;
				line_group.emplace_front(curr_line);
			}
		}
		curr_line = core_line; // 刷新初始位置
		while (true)
		{
			if (lNeighbors[curr_line].second == -1 || is_lLabeled[lNeighbors[curr_line].second])
			{
				break;
			}
			else
			{
				curr_line = lNeighbors[curr_line].second;
				is_lChecked[curr_line] = true;
				line_group.emplace_back(curr_line);
			}
		}

		DPatches patches = Douglas_Peucker(line_group);

		for (size_t j = 0; j < patches.size(); ++j)
		{
			Eigen::Vector3d mass_temp = Eigen::Vector3d::Zero();
			Eigen::Matrix3d moment2_temp = Eigen::Matrix3d::Zero();
			double weight_temp = 0;
			for (size_t k = 0; k < patches[j].size(); ++k)
			{
				mass_temp += lSum_[patches[j][k]];
				moment2_temp += lMatrix_[patches[j][k]];
				weight_temp += lWeight_[patches[j][k]];
			}
			Eigen::Vector3d centroid_temp = mass_temp/weight_temp;
			Eigen::Matrix3d scatter_temp = moment2_temp - mass_temp * centroid_temp.transpose();
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver_temp(scatter_temp);
			if (solver_temp.eigenvalues()[0]/solver_temp.eigenvalues()[1] > T_ratio_)
			{
				continue;
			}

			Eigen::Vector3d normal_temp = solver_temp.eigenvectors().col(0);
			if (fabs(centroid_temp.dot(normal_temp)) / centroid_temp.norm() <= cos_85_)
			{
				continue;
			}
			planes_.emplace_back(currentCloudPtr_->size());
			planes_.back().vec_s = mass_temp;
			planes_.back().mat_J = moment2_temp;
			planes_.back().weight_sum = weight_temp;
			planes_.back().vec_n = normal_temp;
			planes_.back().d = normal_temp.dot(centroid_temp);
			planes_.back().mat_Q = calcCovariance(scatter_temp, moment2_temp, centroid_temp, normal_temp, weight_temp);
			for (size_t k = 0; k < patches[j].size(); ++k)
			{
				int line_ind = patches[j][k];
				for (size_t n = 0; n < lPoints_[line_ind].size(); ++n)
				{
					planes_.back().queue_row[planes_.back().size] = lPoints_[line_ind][n].first;
					planes_.back().queue_col[planes_.back().size] = lPoints_[line_ind][n].second;
					++planes_.back().size;
				}
			}

			for (size_t k = 0; k < patches[j].size(); ++k)
			{
				is_lLabeled[patches[j][k]] = true;
			}
		}
	}

	for (size_t i = 0; i < lSum_.size(); ++i)
	{
		if (is_lLabeled[i]) continue;
		for (size_t j = 0; j < lPoints_[i].size(); ++j)
		{
			lineMat_.at<int>(lPoints_[i][j].first, lPoints_[i][j].second) = -1;
		}
	}

	delete [] is_lChecked;
	delete [] is_lLabeled;

	for (size_t i = 0; i < planes_.size(); ++i)
	{
		cv::Mat is_pChecked = cv::Mat(indexMat_.rows, indexMat_.cols, CV_8S, cv::Scalar::all(-1));
		int queue_start = 0;
		int queue_size = planes_[i].size;

		int new_points = 0;
		while (queue_size > 0)
		{
			int from_x = planes_[i].queue_row[queue_start];
			int from_y = planes_[i].queue_col[queue_start];
			--queue_size;
			++queue_start;
			for (size_t j = 0; j < point_iter_.size(); ++j)
			{
				int this_x = from_x + point_iter_[j].first;
				int this_y = from_y + point_iter_[j].second;
				if (this_x < 0 || this_x >= indexMat_.rows)
					continue;
				if (this_y < 0 || this_y >= indexMat_.cols)
					continue;

				int ind = indexMat_.at<int>(this_x, this_y);
				if (lineMat_.at<int>(this_x, this_y) > -1)
					continue;
				if (planeMat_.at<int>(this_x, this_y) > -1)
					continue;
				if (is_pChecked.at<int8_t>(this_x, this_y) > -1)
					continue;

				is_pChecked.at<int8_t>(this_x, this_y) = 1;
				Eigen::Vector3d pt(currentCloudPtr_->points[ind].x, currentCloudPtr_->points[ind].y, currentCloudPtr_->points[ind].z);
				if (DeviationFromPlane(ind, planes_[i].vec_n, planes_[i].d, planes_[i].mat_Q) > D_Po2P_)
					continue;

				planes_[i].vec_s += pWeight_[ind]*pt;
				planes_[i].weight_sum += pWeight_[ind];
				planes_[i].mat_J += pWeight_[ind]*pt*pt.transpose();

				lineMat_.at<int>(this_x, this_y) = 99998; // 之前未被分配到直线但却属于平面的散点
				planes_[i].queue_row[planes_[i].size] = this_x;
				planes_[i].queue_col[planes_[i].size] = this_y;
				++planes_[i].size;
				++queue_size;
				++new_points;
			}

			if (new_points >= update_step_)
			{
				Eigen::Vector3d vec_m = planes_[i].vec_s / planes_[i].weight_sum;
				Eigen::Matrix3d mat_C = planes_[i].mat_J - planes_[i].vec_s * vec_m.transpose();
				Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(mat_C);
				planes_[i].vec_n = solver.eigenvectors().col(0);
				planes_[i].mat_Q = calcCovariance(mat_C, planes_[i].mat_J, vec_m, planes_[i].vec_n, planes_[i].weight_sum);
				planes_[i].d = planes_[i].vec_n.dot(vec_m);
				new_points -= update_step_;
			}
		}

		Eigen::Vector3d vec_m = planes_[i].vec_s / planes_[i].weight_sum;
		Eigen::Matrix3d mat_C = planes_[i].mat_J - planes_[i].vec_s * vec_m.transpose();
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> final_solver(mat_C);
		planes_[i].eigenvalues = final_solver.eigenvalues();
		planes_[i].vec_n = final_solver.eigenvectors().col(0);
		planes_[i].d = planes_[i].vec_n.dot(vec_m);
		planes_[i].sigma = sqrt(final_solver.eigenvalues()[0] / planes_[i].weight_sum);
		// 统一朝向
		if (planes_[i].d < 0)
		{
			planes_[i].d = -planes_[i].d;
			planes_[i].vec_n = -planes_[i].vec_n;
		}
		// 标记各点所属的平面
		for (int j = 0; j < planes_[i].size; ++j)
			planeMat_.at<int>(planes_[i].queue_row[j], planes_[i].queue_col[j]) = i;
	}

	Eigen::MatrixXf connectGraph = Eigen::MatrixXf::Identity(planes_.size(), planes_.size());  
	for (int i = 0; i < planeMat_.rows; ++i) // 在列方向上搜索
	{
		std::vector<std::pair<int, int>> planeSlice;
		std::vector<int> planeInd;
		
		int lastCol   = -1;
		for (int j = 0; j < planeMat_.cols; ++j)
		{
			int currPlane = planeMat_.at<int>(i,j);
			if (currPlane > -1)
			{
				if (lastCol == -1)
				{
					planeSlice.emplace_back(std::make_pair(j,-1));
					planeInd.emplace_back(currPlane);
				}
				else
				{
						if (currPlane != planeMat_.at<int>(i, lastCol))
						{
							planeSlice.back().second = lastCol;
							planeSlice.emplace_back(std::make_pair(j, -1));
							planeInd.emplace_back(currPlane);
						}
				}

				lastCol = j; // 保留上一个有效点的列号
			}
		}

		if (planeSlice.size() == 0) continue;

		planeSlice.back().second = lastCol;
		int sliceNum = planeSlice.size();
		for (int j = 0; j < sliceNum; ++j)
		{
			// 左侧
			for (int k = j - 1; k > j - sliceNum; --k)
			{
				int slice = (k + sliceNum)%sliceNum;
				int colDiff = planeSlice[j].first - planeSlice[slice].second;
				if (colDiff < 0) colDiff += planeMat_.cols;
				if (colDiff < buffer_size_)
				{
					if (connectGraph(planeInd[slice], planeInd[j]) == 1) break;
					if (connectGraph(planeInd[slice], planeInd[j]) == 0)
					{
						float merged = mergePlanes(planeInd[slice], planeInd[j]);
						connectGraph(planeInd[slice], planeInd[j]) = merged;
						connectGraph(planeInd[j], planeInd[slice]) = merged;
						if (merged > 0) break;
					}
				}
				else
				{
					break;
				}
			}
			// 右侧
			for (int k = j + 1; k < j + sliceNum; ++k)
			{
				int slice = k%sliceNum;
				int colDiff = planeSlice[slice].first - planeSlice[j].second;
				if (colDiff < 0) colDiff += planeMat_.cols;
				if (colDiff < buffer_size_)
				{
					if (connectGraph(planeInd[slice], planeInd[j]) == 1) break;
					if (connectGraph(planeInd[slice], planeInd[j]) == 0)
					{
						float merged = mergePlanes(planeInd[slice], planeInd[j]);
						connectGraph(planeInd[slice], planeInd[j]) = merged;
						connectGraph(planeInd[j], planeInd[slice]) = merged;
						if (merged > 0) break;
					}
				}
				else
				{
					break;
				}
			}
		}
	}

	for (int i = 0; i < planeMat_.cols; ++i)// 在行方向上搜索
	{
		for (int j = 1; j < planeMat_.rows; ++j)
		{
			int lastPlane = planeMat_.at<int>(j - 1, i);
			int currPlane = planeMat_.at<int>(j    , i);
			if (lastPlane == -1 || currPlane == -1)
				continue;
			if (connectGraph(currPlane, lastPlane) != 0 || connectGraph(lastPlane, currPlane) != 0)
				continue;
			
			float merged = mergePlanes(currPlane, lastPlane);
			connectGraph(currPlane, lastPlane) = merged;
			connectGraph(lastPlane, currPlane) = merged;
		}
	}

	bool *isMerged = new bool[planes_.size()]();
	int *sub_planes = new int[1000];
	int16_t label = 0;
	for (size_t i = 0; i < planes_.size(); ++i)
	{
		if (isMerged[i])
			continue;

		sub_planes[0] = i;
		int queue_start = 0;
		int queue_size = 1;
		int queue_end = 1;

		isMerged[i] = true;
		while (queue_size > 0)
		{
			int seed = sub_planes[queue_start];
			++queue_start;
			--queue_size;
			for (size_t j = 0; j < planes_.size(); ++j)
			{
				if (!isMerged[j] && connectGraph(seed, j) > 0)
				{
					isMerged[j] = true;

					sub_planes[queue_end] = j;
					++queue_size;
					++queue_end;
				}
			}
		}

		std::vector<int> consistency(queue_end, 0);
		for (int j = 0; j < queue_end; ++j)
			for (int k = 0; k < queue_end; ++k)
				consistency[j] += planes_[sub_planes[k]].size*connectGraph(sub_planes[j], sub_planes[k]);

		std::vector<bool> outlierFlag(queue_end, false);
		for (int j = 0; j < queue_end; ++j)
		{
			for (int k = j + 1; k < queue_end; ++k)
			{
				if (connectGraph(sub_planes[j], sub_planes[k]) == -1 || connectGraph(sub_planes[k], sub_planes[j]) == -1)
				{
					outlierFlag[k] = consistency[j] > consistency[k];
				}
			}
		}

		for (int j = 0; j < queue_end; ++j)
		{
			if (outlierFlag[j])
			{
				i = std::min((int)i, sub_planes[j]);
				break;
			}
		}
		// 合并非outlier的平面
		pcl::PointIndices::Ptr memberList(new pcl::PointIndices());
		for (int j = 0; j < queue_end; ++j)
		{
			if (outlierFlag[j])
			{
				isMerged[sub_planes[j]] = false;
				continue;
			}

			for (int k = 0; k < planes_[sub_planes[j]].size; ++k)
			{
				int ind = indexMat_.at<int>(planes_[sub_planes[j]].queue_row[k], planes_[sub_planes[j]].queue_col[k]);
				memberList->indices.emplace_back(ind);
			}
		}
		// 舍弃点数不足的平面
		if (memberList->indices.size() < T_count_)
			continue;
		// 舍弃不能与其他平面合并的狭长平面
		if (queue_end == 1 && planes_[sub_planes[0]].eigenvalues(1)/planes_[sub_planes[0]].eigenvalues(2) < T_ratio_)
			continue;
		
		for (const auto& ind : memberList->indices) { currentCloudPtr_->points[ind].label = label; }
		++label;
	}

	delete [] sub_planes;
	delete [] isMerged;
}

void LineBasedDouglasPeucker::reset()
{
	currentCloudPtr_->clear();

	indexMat_.setTo(cv::Scalar(-1));
	lineMat_.setTo(cv::Scalar(-1));
	planeMat_.setTo(cv::Scalar(-1));

	for (auto& vec : lPoints_) vec.clear();
	for (auto& vec : lPos_1d_) vec.clear();

	lSum_.clear();
	lMatrix_.clear();
	lWeight_.clear();
	lPoints_.clear();
	lPos_2d_.clear();
	lPriority_.clear();

	planes_.clear();
}

Eigen::Matrix3d LineBasedDouglasPeucker::skewSymmetric(const Eigen::Vector3d& vec)
{
	Eigen::Matrix3d mat_sm = Eigen::Matrix3d::Zero();
	mat_sm(0,1) = -vec.z(); mat_sm(0,2) = vec.y();
	mat_sm(1,2) = -vec.x(); mat_sm(1,0) = vec.z();
	mat_sm(2,0) = -vec.y(); mat_sm(2,1) = vec.x();

	return mat_sm;
}

Eigen::Matrix4d LineBasedDouglasPeucker::calcCovariance(const Eigen::Matrix3d& mat_C, const Eigen::Matrix3d& mat_J,
  const Eigen::Vector3d& vec_m, const Eigen::Vector3d& vec_n, double weight_sum) const
{
	Eigen::Matrix4d mat_P = Eigen::Matrix4d::Zero();
	Eigen::Vector3d vec_side = weight_sum*vec_m;
	mat_P.block<3,3>(0,0) = -mat_J + (vec_n.transpose()*mat_C*vec_n)*Eigen::Matrix3d::Identity();
	mat_P.block<3,1>(0,3) = vec_side;
	mat_P.block<1,3>(3,0) = vec_side.transpose();
	mat_P(3,3) = -weight_sum;

	Eigen::JacobiSVD<Eigen::Matrix4d> svd(mat_P, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Vector4d sv = svd.singularValues();
	Eigen::Matrix4d sv_inv = Eigen::Matrix4d::Zero();
	// 矩阵秩为3，只保留特征值最大的三个
	for (int i = 0; i < 3; ++i)
		sv_inv(i,i) = 1.0/sv(i);

	return -(svd.matrixV())*(sv_inv)*(svd.matrixU().transpose());
}

double LineBasedDouglasPeucker::DeviationFromLine(int ind_A, int ind_B, int ind_C)
{
	// 基于张角判断
	Eigen::Vector3d vec_AB(currentCloudPtr_->points[ind_B].x - currentCloudPtr_->points[ind_A].x,
	                       currentCloudPtr_->points[ind_B].y - currentCloudPtr_->points[ind_A].y,
	                       currentCloudPtr_->points[ind_B].z - currentCloudPtr_->points[ind_A].z);
	Eigen::Vector3d vec_AC(currentCloudPtr_->points[ind_C].x - currentCloudPtr_->points[ind_A].x,
	                       currentCloudPtr_->points[ind_C].y - currentCloudPtr_->points[ind_A].y,
	                       currentCloudPtr_->points[ind_C].z - currentCloudPtr_->points[ind_A].z);
	double norm_AB = vec_AB.norm();
	double norm_AC = vec_AC.norm();
	double cosine = vec_AB.dot(vec_AC)/(norm_AB*norm_AC); // flatness = cosine + 1

	Eigen::Vector3d df_dA = -(vec_AB + vec_AC)/(norm_AB*norm_AC) + cosine*vec_AB/(norm_AB*norm_AB) + cosine*vec_AC/(norm_AC*norm_AC);
	Eigen::Vector3d df_dB = vec_AC/(norm_AB*norm_AC) - cosine*vec_AB/(norm_AB*norm_AB);
	Eigen::Vector3d df_dC = vec_AB/(norm_AB*norm_AC) - cosine*vec_AC/(norm_AC*norm_AC);

	Eigen::Map<Eigen::Matrix3d> cov_A(currentCloudPtr_->points[ind_A].covariance);
	Eigen::Map<Eigen::Matrix3d> cov_B(currentCloudPtr_->points[ind_B].covariance);
	Eigen::Map<Eigen::Matrix3d> cov_C(currentCloudPtr_->points[ind_C].covariance);
	double sigma = sqrt(df_dA.dot(cov_A*df_dA) + df_dB.dot(cov_B*df_dB) + df_dC.dot(cov_C*df_dC));

	return fabs(cosine + 1)/sigma;
}

double LineBasedDouglasPeucker::DeviationFromPlane(int pointInd, const Eigen::Vector3d& normal, double d, const Eigen::Matrix4d& cov)
{
	Eigen::Vector4d extended_pt(currentCloudPtr_->points[pointInd].x, currentCloudPtr_->points[pointInd].y, currentCloudPtr_->points[pointInd].z, -1);
	Eigen::Map<Eigen::Matrix3d> pointCov(currentCloudPtr_->points[pointInd].covariance);
	double sigma = sqrt(normal.dot(pointCov*normal) + extended_pt.dot(cov*extended_pt));

	return fabs(normal.dot(extended_pt.head(3)) - d)/sigma;
}

int LineBasedDouglasPeucker::OverlappedNeighbor(int neighborCol, int core_line, int& lower_bound, int& upper_bound) const
{
	int max_overlap = 0;
	int neighbor = -1;
	for (size_t i = 0; i < lPos_1d_[neighborCol].size(); ++i)
	{
		int curr_line = lPos_1d_[neighborCol][i];
		int top = std::min(lPoints_[core_line].back().first, lPoints_[curr_line].back().first);
		int bottom = std::max(lPoints_[core_line].front().first, lPoints_[curr_line].front().first);
		int overlap = top - bottom + 1; // 若两条直线不相交，该值则小于等于零
		if (overlap > max_overlap)
		{
			neighbor = curr_line;
			lower_bound = bottom;
			upper_bound = top;
			max_overlap = overlap;
		}
	}

	if (neighbor > -1)
	{
		// if ((float)max_overlap/points[core_line].size() >= T_ove_)
		//     return neighbor;
		if ((float)max_overlap/lPoints_[core_line].size() >= 0.5 || (float)max_overlap/lPoints_[neighbor].size() >= 0.5)
			return neighbor;
	}

	return -1;
}

DPatches LineBasedDouglasPeucker::Douglas_Peucker(const std::deque<int>& line_group)
{
	std::vector<std::deque<int>> patches;
	if (line_group.size() < 3)
		return patches;

	// 利用首尾两条直线拟合平面
	Eigen::Vector3d vec_s = lSum_[line_group.front()] + lSum_[line_group.back()];
	double weight_sum = lWeight_[line_group.front()] + lWeight_[line_group.back()];
	Eigen::Vector3d vec_m = vec_s/weight_sum;
	Eigen::Matrix3d mat_J = lMatrix_[line_group.front()] + lMatrix_[line_group.back()];
	Eigen::Matrix3d mat_C = mat_J - vec_s*vec_m.transpose();
	// 解算参数及协方差
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(mat_C);
	Eigen::Vector3d vec_n = solver.eigenvectors().col(0);
	double d = vec_m.dot(vec_n);
	Eigen::Matrix4d cov = calcCovariance(mat_C, mat_J, vec_m, vec_n, weight_sum);

	float max_deviation = 0;
	int location = -1;
	for (size_t i = 1; i < line_group.size() - 1; ++i)
	{
		int line_ind = line_group[i];
		// 只用两个端点判断判断直线偏离平面的程度
		int i_start = indexMat_.at<int>(lPoints_[line_ind].front().first, lPoints_[line_ind].front().second);
		int i_end = indexMat_.at<int>(lPoints_[line_ind].back().first, lPoints_[line_ind].back().second);
		double deviation = 0.5 * (DeviationFromPlane(i_start, vec_n, d, cov) + DeviationFromPlane(i_end, vec_n, d, cov));

		// 用线段上的所有点
		// double deviation = 0;
		// for (size_t j = 0; j < lPoints_[line_ind].size(); ++j)
		// {
		//     int point_ind = indexMat_.at<int>(lPoints_[line_ind][j].first, lPoints_[line_ind][j].second);
		//     deviation += DeviationFromPlane(point_ind, vec_n, d, cov);
		// }
		// deviation /= lPoints_[line_ind].size();

		if (deviation > max_deviation)
		{
			max_deviation = deviation;
			location = i;
		}
	}

	if (max_deviation < D_L2P_) // 所有直线均在平面误差范围内，不再继续拆分
	{
		patches.emplace_back(line_group);
	}
	else
	{
		std::deque<int> front_half;
		front_half.insert(front_half.end(), line_group.begin(), line_group.begin() + location);
		DPatches front_subs = Douglas_Peucker(front_half);
		if (front_subs.size() > 0) patches.insert(patches.end(), front_subs.begin(), front_subs.end());

		std::deque<int> back_half;
		back_half.insert(back_half.end(), line_group.begin() + location + 1, line_group.end());
		DPatches back_subs = Douglas_Peucker(back_half);
		if (back_subs.size() > 0) patches.insert(patches.end(), back_subs.begin(), back_subs.end());
	}

	return patches;
}

float LineBasedDouglasPeucker::mergePlanes(int plane_1, int plane_2) const
{
	// 1. 待合并的两平面相互平行
	if (planes_[plane_1].vec_n.dot(planes_[plane_2].vec_n) < cos_angle_)
		return -1;
	// 2. 新平面拟合误差小于阈值
	Eigen::Vector3d sum      = planes_[plane_1].vec_s + planes_[plane_2].vec_s;
	double weight_sum        = planes_[plane_1].weight_sum + planes_[plane_2].weight_sum;
	Eigen::Vector3d centroid = sum/weight_sum;
	Eigen::Matrix3d moment_2 = planes_[plane_1].mat_J + planes_[plane_2].mat_J;

	Eigen::Matrix3d mat_C = moment_2 - sum*centroid.transpose();
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(mat_C);
	float sigma_ij = sqrt(solver.eigenvalues()[0]/weight_sum);
	if (sigma_ij > planes_[plane_1].sigma + planes_[plane_2].sigma)
		return -1;
	
	return 1/(1 + sigma_ij);
}

}

#endif
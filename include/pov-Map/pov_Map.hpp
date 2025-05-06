// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <algorithm>
#include <queue>
#include <unordered_set>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <sophus/se3.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

namespace pov_map
{
using Voxel = Eigen::Vector3i;
using Vector4dVector = std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>;

struct VoxelHash
{
	size_t operator()(const Voxel &voxel) const
	{
		const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
		return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
	}
};

struct One2OneTuples
{
	One2OneTuples(std::size_t n)
	{
		trials = 0;
		source.reserve(n);
		target.reserve(n);
	}
	int trials;
	std::vector<int> source;
	Vector4dVector target;
};

struct One2MultiTuples
{
	One2MultiTuples(std::size_t n)
	{
		source.reserve(n);
		target.reserve(n);
	}
	std::vector<int> source;
	std::vector<Vector4dVector> target;
};

class PlaneOccupiedVoxelMap
{
	using Vector4dVectorTuple = std::tuple<Vector4dVector, Vector4dVector>;
	using IndexVectorTuple = std::tuple<int, std::vector<int>, Vector4dVector>;
	using IndexVectorVectorTuple = std::tuple<std::vector<int>, std::vector<Vector4dVector>>;
	using Vector3dVector = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
	using Vector3dMatrix = std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>;
	using Vector4dMatrix = std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;

	enum class OccupiedStatus
	{
		Plane,
		Patch,
		Point
	};

	struct VoxelBlock
	{
		Vector4dVector points;
		tsl::robin_set<int> planeIDs;
		OccupiedStatus status;
		int maximum;

		VoxelBlock(int _maximum)
		{
			maximum = _maximum;
			points.reserve(maximum);
			planeIDs.reserve(100);
		}

		inline void addPoint(const Eigen::Vector4d &point)
		{
			if (points.size() < static_cast<size_t>(maximum))
			{
				points.emplace_back(point);
			}
		}
	};

	struct Plane
	{
		Plane() = default;
		Plane(const Eigen::Vector3d &_anchor)
		{
			anchor = _anchor;
			moment = Eigen::Matrix3d::Zero();
			sum = Eigen::Vector3d::Zero();
			weight = 0.0;
			lastTimeBeSeen = 0;	
		}
		Plane(Eigen::Matrix3d _moment, Eigen::Vector3d _sum, double _weight)
		{
			moment = _moment;
			sum = _sum;
			weight = _weight;
			lastTimeBeSeen = 0;
		}
		Plane(Eigen::Matrix4d cluster)
		{
			moment = cluster.topLeftCorner<3, 3>();
			sum = cluster.topRightCorner<3, 1>();
			weight = cluster(3, 3);
			lastTimeBeSeen = 0;
		}

		bool updateParams(float anotherRMSE = 0)
		{
			Eigen::Vector3d center = sum / weight;
			Eigen::Matrix3d scatter = moment - center * sum.transpose();
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
			float updatedRMSE = sqrt(solver.eigenvalues()[0] / weight);
			if (updatedRMSE > rmse + anotherRMSE || solver.eigenvalues()[0] / solver.eigenvalues()[1] > 0.01)
			{
				//ROS_INFO("Reject To Update Plane: mse increase from %f to %f", mse, updatedMSE);
				//ROS_INFO("Reject To Update Plane: %f > 0.01", solver.eigenvalues()[0] / solver.eigenvalues()[1]);
				return false;
			}
			rmse = updatedRMSE;
			normal = solver.eigenvectors().col(0);
			d = normal.dot(center);
			if (normal.dot(anchor - center) < 0)
			{
				normal = -normal;
				d = -d;
			}

			Eigen::Matrix3d H_nn = -moment + normal.dot(scatter * normal) * Eigen::Matrix3d::Identity();
			Eigen::Matrix3d H_nn_d = H_nn + center * sum.transpose();
			Eigen::JacobiSVD<Eigen::Matrix3d> decoupledSolver(H_nn_d, Eigen::ComputeFullU | Eigen::ComputeFullV); 
			Eigen::Matrix3d decoupledSV_inv = Eigen::Matrix3d::Zero();
			for (int i = 0; i < 2; ++i)
					decoupledSV_inv(i,i) = 1.0 / decoupledSolver.singularValues()[i];
			D_nn = -decoupledSolver.matrixV() * decoupledSV_inv * (decoupledSolver.matrixU().transpose());

			Eigen::Matrix3d H_nn_i = H_nn.inverse();
			double upper = -normal.dot(H_nn_i * normal);
			double lower = normal.dot(H_nn_i * sum);
			D_dd = upper / (lower * lower);

			return true;
		}

		Eigen::Matrix3d moment;
		Eigen::Vector3d sum;
		Eigen::Vector3d normal;
		Eigen::Matrix3d D_nn;
		Eigen::Vector3d anchor;
		double weight;
		double d;
		double D_dd;
		float rmse = FLT_MAX;
		int lastTimeBeSeen = 0;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	} EIGEN_ALIGN16;

public:
	explicit PlaneOccupiedVoxelMap(double voxel_size, double max_distance, int max_points_per_voxel);
	// For PCL
	template <typename PointT>
	IndexVectorTuple getNearestPoint(const pcl::PointCloud<PointT> &cloud, double distanceThreshold2) const;
	template <typename PointT>
	IndexVectorVectorTuple getNearestPatch(const pcl::PointCloud<PointT> &cloud, double distanceThreshold2) const;
	template <typename PointT>
	int getNearestPlaneID(const pcl::PointCloud<PointT> &cloud, const Eigen::Vector3d &normal,
												const Eigen::Vector3d &center, const Eigen::Matrix3d &cov) const;
	template <typename PointT>
	void updateLabel(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &planeMap);
	template <typename PointT>
	void updateMap(const pcl::PointCloud<PointT> &cloud);
	template <typename PointT>
	void getPointCloud(pcl::PointCloud<PointT> &cloud) const;

	inline void clear();
	inline bool empty() const;
	void updatePlaneInfo(int &planeID, const Eigen::Matrix4d &pointCluster, const Eigen::Vector3d &origin);
	void updatePlaneInfoBatch(std::vector<int> &planeIDs, const Vector4dMatrix &pointClusters, const Eigen::Vector3d &origin);
	std::optional<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> getPlaneNormal(int planeID);
	std::optional<std::pair<double, double>> getPlaneD(int planeID);
	void removePointsFarFromLocation(const Eigen::Vector3d &origin);

private:
	int updateCount_ = 0;
	int currentLabel_ = 0;
	double voxelSize_;
	double maxDistance_;
	double maxDistance2_;
	int maxPointsPerVoxel_;

	tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
	tsl::robin_map<int, Plane> planes_;
};

template <typename PointT>
void downsampleByVoxelGrid(const pcl::PointCloud<PointT> &cloudIn, pcl::PointCloud<PointT> &cloudOut, double voxel_size)
{
	if (cloudIn.empty())
	{
		cloudOut.clear();
		return;
	}
	tsl::robin_map<Voxel, PointT, VoxelHash> grid;
	grid.reserve(cloudIn.size());
	for (const auto &point : cloudIn.points)
	{
		const auto voxel = Voxel(static_cast<int>(point.x / voxel_size), static_cast<int>(point.y / voxel_size), static_cast<int>(point.z / voxel_size));
		if (grid.contains(voxel)) continue;
		grid.insert({voxel, point});
	}
	
	cloudOut.clear();
	cloudOut.reserve(grid.size());
	for (const auto &[voxel, point] : grid)
	{
		(void)voxel;
		cloudOut.push_back(point);
	}
}

template <typename PointT>
void downsampleByVoxelGrid(const std::vector<typename pcl::PointCloud<PointT>::Ptr> &cloudsIn,
	pcl::PointCloud<PointT> &cloudOut, double voxel_size, bool reverse = false)
{
	tsl::robin_map<Voxel, PointT, VoxelHash> grid;
	int predictedSize = 0; 
	for (const auto &cloud : cloudsIn)
	{
		predictedSize += cloud->size();
	}
	grid.reserve(predictedSize);
	if (reverse)
	{
		for (auto it = cloudsIn.crbegin(); it != cloudsIn.crend(); ++it)
		{
			for (const auto &point : it->get()->points)
			{
				const auto voxel = Voxel(static_cast<int>(point.x / voxel_size), static_cast<int>(point.y / voxel_size), static_cast<int>(point.z / voxel_size));
				if (grid.contains(voxel)) continue;
				grid.insert({voxel, point});
			}	
		}
	}
	else
	{
		for (auto it = cloudsIn.cbegin(); it != cloudsIn.cend(); ++it)
		{
			for (const auto &point : it->get()->points)
			{
				const auto voxel = Voxel(static_cast<int>(point.x / voxel_size), static_cast<int>(point.y / voxel_size), static_cast<int>(point.z / voxel_size));
				if (grid.contains(voxel)) continue;
				grid.insert({voxel, point});
			}	
		}
	}
	
	cloudOut.clear();
	cloudOut.reserve(grid.size());
	for (const auto &[voxel, point] : grid)
	{
		(void)voxel;
		cloudOut.push_back(point);
	}
}

PlaneOccupiedVoxelMap::PlaneOccupiedVoxelMap(double voxel_size, double max_distance, int max_points_per_voxel)
	: voxelSize_(voxel_size), maxDistance_(max_distance), maxPointsPerVoxel_(max_points_per_voxel)
{
	maxDistance2_ = maxDistance_ * maxDistance_;
}

template <typename PointT>
PlaneOccupiedVoxelMap::IndexVectorTuple PlaneOccupiedVoxelMap::getNearestPoint(const pcl::PointCloud<PointT> &cloud, double distanceThreshold2) const
{
	// Lambda Function to obtain the KNN of one point, maybe refactor
	auto getClosestNeighbor = [&](const PointT &point)
	{
		auto kx = static_cast<int>(point.x / voxelSize_);
		auto ky = static_cast<int>(point.y / voxelSize_);
		auto kz = static_cast<int>(point.z / voxelSize_);
		std::vector<Voxel> voxels;
		voxels.reserve(27);
		for (int i = kx - 1; i < kx + 1 + 1; ++i)
		{
			for (int j = ky - 1; j < ky + 1 + 1; ++j)
			{
				for (int k = kz - 1; k < kz + 1 + 1; ++k)
				{
					voxels.emplace_back(i, j, k);
				}
			}
		}

		Vector4dVector neighbors;
		neighbors.reserve(27 * maxPointsPerVoxel_);
		std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel)
		{
			auto search = map_.find(voxel);
			if (search != map_.end())
			{
				const auto &points = search->second.points;
				if (!points.empty())
				{
					for (const auto &point : points)
					{
						neighbors.push_back(point);
					}
				}
			}
		});

		Eigen::Vector4d closestNeighbor = Eigen::Vector4d::Zero();
		float closestDistance2 = maxDistance2_;
		std::for_each(neighbors.cbegin(), neighbors.cend(), [&](const auto &neighbor)
		{
			float diff_x = point.x - neighbor.x();
			float diff_y = point.y - neighbor.y();
			float diff_z = point.z - neighbor.z();
			float distance2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
			if (distance2 < closestDistance2)
			{
				closestNeighbor = neighbor;
				closestDistance2 = distance2;
			}
		});

		return std::make_tuple(closestNeighbor, closestDistance2);
	};

	const auto [trials, source, target] = tbb::parallel_reduce(
		// Range
		tbb::blocked_range<size_t>{0, cloud.size()},
		// Identity
		One2OneTuples(cloud.size()),
		// 1st lambda: Parallel computation
		[distanceThreshold2, &getClosestNeighbor, cloud, this](const tbb::blocked_range<size_t> &r, One2OneTuples res) -> One2OneTuples
		{
			auto &[trials, src, tgt] = res;
			src.reserve(r.size());
			tgt.reserve(r.size());
			for (size_t i = r.begin(); i < r.end(); ++i)
			{
				const auto &point = cloud.points[i];
				++trials;
				const auto &[closestNeighbor, closestDistance2] = getClosestNeighbor(point);
				if (closestDistance2 < distanceThreshold2)
				{
					src.emplace_back(i);
					tgt.emplace_back(closestNeighbor);
				}
			}
			return res;
		},
		// 2nd lambda: Parallel reduction
		[](One2OneTuples a, const One2OneTuples &b) -> One2OneTuples
		{
			auto &[trl, src, tgt] = a;
			const auto &[trlp, srcp, tgtp] = b;
			trl += trlp;
			src.insert(src.end(), srcp.begin(), srcp.end());
			tgt.insert(tgt.end(), std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
			return a;
		});

	return std::make_tuple(trials, source, target);
}

template <typename PointT>
PlaneOccupiedVoxelMap::IndexVectorVectorTuple PlaneOccupiedVoxelMap::getNearestPatch(const pcl::PointCloud<PointT> &cloud, double distanceThreshold2) const
{
	// Lambda Function to obtain the KNN of one point
	auto getKClosestNeighbors = [&](const PointT &point)
	{
		auto kx = static_cast<int>(point.x / voxelSize_);
		auto ky = static_cast<int>(point.y / voxelSize_);
		auto kz = static_cast<int>(point.z / voxelSize_);
		std::vector<Voxel> voxels;
		voxels.reserve(27);
		for (int i = kx - 1; i < kx + 1 + 1; ++i)
		{
			for (int j = ky - 1; j < ky + 1 + 1; ++j)
			{
				for (int k = kz - 1; k < kz + 1 + 1; ++k)
				{
					voxels.emplace_back(i, j, k);
				}
			}
		}

		Vector4dVector neighbors;
		neighbors.reserve(27 * maxPointsPerVoxel_);
		std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel)
		{
			auto search = map_.find(voxel);
			if (search != map_.end()) {
				const auto &points = search->second.points;
				if (!points.empty())
				{
					for (const auto &point : points)
					{
						neighbors.emplace_back(point);
					}
				}
			}
		});

		// Find 5 nearest neighbors using priority queue
		using neighborWithDistance = std::pair<float, Eigen::Vector4d>;
		auto compare = [](const neighborWithDistance& a, const neighborWithDistance& b)
		{
      return a.first < b.first;
    };
		std::priority_queue<neighborWithDistance, std::vector<neighborWithDistance>, decltype(compare)> pq(compare);
		std::for_each(neighbors.cbegin(), neighbors.cend(), [&](const auto &neighbor)
		{
			float diff_x = point.x - neighbor.x();
			float diff_y = point.y - neighbor.y();
			float diff_z = point.z - neighbor.z();
			float distance2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
			if (pq.size() < 5)
			{
				pq.push({distance2, neighbor});
			} 
			else if (distance2 < pq.top().first)
			{
				pq.pop();
				pq.push({distance2, neighbor});
			}
		});

		// Convert priority queue to vectors of points and distances
		Vector4dVector kNearest;
		kNearest.reserve(5);
		float closestDistance2 = maxDistance2_;
		while (!pq.empty())
		{
			// points that are too far away are not considered
			if (pq.top().first < 1.0) 
			{
				kNearest.emplace_back(pq.top().second);
				closestDistance2 = pq.top().first;
			}
			pq.pop();
		}

		return std::make_tuple(kNearest, closestDistance2);
	};

	const auto [source, target] = tbb::parallel_reduce(
		// Range
		tbb::blocked_range<size_t>{0, cloud.size()},
		// Identity
		One2MultiTuples(cloud.size()),
		// 1st lambda: Parallel computation
		[distanceThreshold2, &getKClosestNeighbors, cloud, this](const tbb::blocked_range<size_t> &r, One2MultiTuples res) -> One2MultiTuples
		{
			auto &[src, tgt] = res;
			src.reserve(r.size());
			tgt.reserve(r.size());
			for (size_t i = r.begin(); i < r.end(); ++i)
			{
				auto [kNearest, closestDistance2] = getKClosestNeighbors(cloud.points[i]);
				if (kNearest.size() == 5)
				{
					src.emplace_back(i);
					tgt.emplace_back(kNearest);
				}
				else if (closestDistance2 < distanceThreshold2)
				{
					kNearest.erase(kNearest.begin(), kNearest.end() - 1);
					src.emplace_back(i);
					tgt.emplace_back(kNearest);
				}
			}
			return res;
		},
		// 2nd lambda: Parallel reduction
		[](One2MultiTuples a, const One2MultiTuples &b) -> One2MultiTuples {
			auto &[src, tgt] = a;
			const auto &[srcp, tgtp] = b;
			src.insert(src.end(), srcp.begin(), srcp.end());
			tgt.insert(tgt.end(), std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
			return a;
		});

	return std::make_tuple(source, target);
}

template <typename PointT>
int PlaneOccupiedVoxelMap::getNearestPlaneID(const pcl::PointCloud<PointT> &cloud, const Eigen::Vector3d &normal,
	const Eigen::Vector3d &center, const Eigen::Matrix3d &cov) const
{
	tsl::robin_set<int> nearPlanes;
	tsl::robin_map<int, int> votingResult;
	for (auto point : cloud.points)
	{
		auto kx = static_cast<int>(point.x / voxelSize_);
		auto ky = static_cast<int>(point.y / voxelSize_);
		auto kz = static_cast<int>(point.z / voxelSize_);
		std::vector<Voxel> voxels;
		voxels.reserve(7);
		voxels.emplace_back(kx,     ky,     kz);
		voxels.emplace_back(kx + 1, ky,     kz);
		voxels.emplace_back(kx - 1, ky,     kz);
		voxels.emplace_back(kx,     ky + 1, kz);
		voxels.emplace_back(kx,     ky - 1, kz);
		voxels.emplace_back(kx,     ky,     kz + 1);
		voxels.emplace_back(kx,     ky,     kz - 1);

		nearPlanes.clear();
		std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel)
		{
			auto search = map_.find(voxel);
			if (search != map_.end())
			{
				const auto &block = search.value();
				if (block.status == OccupiedStatus::Plane)
				{
					nearPlanes.insert(block.planeIDs.begin(), block.planeIDs.end());
				}
			}
		});

		for (const auto& planeID : nearPlanes)
		{
			auto search = votingResult.find(planeID);
			if (search != votingResult.end())
			{
				auto &votes = search.value();
				votes++;
			}			
			else
			{
				votingResult.insert({planeID, 1});
			}
		}
	}

	int minPlaneID = -1;
	float minRMSE = FLT_MAX;
	for (const auto &[planeID, votes] : votingResult)
	{
		//ROS_INFO("%d / %zu", votes, cloud.size());
		if (votes < 0.7 * cloud.size()) continue;
		auto search = planes_.find(planeID);
		if (search != planes_.end())
		{
			const auto &plane = search.value();
			if (normal.dot(plane.normal) < 0.985) continue; // 10 degrees
			float sigma = sqrt(center.dot(plane.D_nn * center) + plane.D_dd + normal.dot(cov * normal));
			float distance = fabs(center.dot(plane.normal) - plane.d);
			if (distance > 3 * sigma) continue;
			if (plane.rmse < minRMSE)
			{
				minPlaneID = planeID;
				minRMSE = plane.rmse;
			}			
		}			
	}

  return minPlaneID;
}

inline void PlaneOccupiedVoxelMap::clear()
{ 
	map_.clear(); 
}

inline bool PlaneOccupiedVoxelMap::empty() const
{ 
	return map_.empty();
}

template <typename PointT>
void PlaneOccupiedVoxelMap::getPointCloud(pcl::PointCloud<PointT> &cloud) const
{
	cloud.clear();
	cloud.reserve(maxPointsPerVoxel_ * map_.size());
	for (const auto &[voxel, voxel_block] : map_)
	{
		(void)voxel;
		for (const auto &point : voxel_block.points)
		{
			PointT pclPoint;
			pclPoint.x = point.x();
			pclPoint.y = point.y();
			pclPoint.z = point.z();
			cloud.emplace_back(pclPoint);
		}
	}
}

template <typename PointT>
void PlaneOccupiedVoxelMap::updateLabel(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &planeMap)
{
	// Only change status and planeIDs, no point inserted
	for (const auto &point : cloud.points)
	{
		auto voxel = Voxel(static_cast<int>(point.x / voxelSize_), static_cast<int>(point.y / voxelSize_), static_cast<int>(point.z / voxelSize_));
		auto search = map_.find(voxel);
		int pointLabel = static_cast<int>(point.label);
		if (pointLabel >= 0) pointLabel = planeMap[pointLabel]; // convert the plane ID from local to global
		if (search != map_.end())
		{
			auto &block = search.value();
			switch (block.status)
			{
				case OccupiedStatus::Point:
				case OccupiedStatus::Patch:
					if (pointLabel >= 0)
					{
						block.planeIDs.insert(pointLabel);
						block.status = OccupiedStatus::Plane;
					}
					else if (pointLabel == -1)
					{
						block.status = OccupiedStatus::Patch;
					}
					break;
				case OccupiedStatus::Plane:
					if (pointLabel >= 0) block.planeIDs.insert(pointLabel);
					break;
				default:
					break;
			}

			if (block.planeIDs.size() > 100 && block.planeIDs.load_factor() > 0.5)
			{
				ROS_WARN("purge over-loaded voxel!");
				std::vector<int> validPlaneIDs;
				validPlaneIDs.reserve(100);
				for (const auto &planeID : block.planeIDs)
				{
					const auto &search = planes_.find(planeID);
					if (search != planes_.end()) validPlaneIDs.emplace_back(planeID); 
				}
				block.planeIDs.clear();
				if (!validPlaneIDs.empty()) block.planeIDs.insert(validPlaneIDs.begin(), validPlaneIDs.end());
			}
		}
		else
		{
			VoxelBlock block(maxPointsPerVoxel_);
			if (pointLabel < 0)
			{
				block.status = pointLabel == -1 ? OccupiedStatus::Patch : OccupiedStatus::Point;
			}
			else
			{
				block.status = OccupiedStatus::Plane;
				block.planeIDs.insert(pointLabel);
			}
			map_.insert({voxel, block});
		}
	}
}

template <typename PointT>
void PlaneOccupiedVoxelMap::updateMap(const pcl::PointCloud<PointT> &cloud)
{
	std::for_each(cloud.points.cbegin(), cloud.points.cend(), [&](const auto &point)
	{
		double uncertainty = point.covariance[0] + point.covariance[4] + point.covariance[8];
		Eigen::Vector4d pv(point.x, point.y, point.z, uncertainty); // the fourth member is point uncertainty
		auto voxel = Voxel((pv.head(3) / voxelSize_).template cast<int>());
		auto search = map_.find(voxel);
		if (search != map_.end())
		{
			auto &block = search.value();
			block.addPoint(pv);
		}
		else 
		{
			// voxels are created when updating labels.
		}
	});
}

void PlaneOccupiedVoxelMap::updatePlaneInfo(int &planeID, const Eigen::Matrix4d &pointCluster, const Eigen::Vector3d &origin)
{
	Plane newPlane(pointCluster);
	newPlane.updateParams();
	newPlane.anchor = origin;
	newPlane.lastTimeBeSeen = updateCount_;

	auto search = planes_.find(planeID);
	if (search != planes_.end()) // existing plane
	{
		Plane oldPlane = search.value(); // trace back if update fails

		auto &plane = search.value();
		plane.moment += pointCluster.topLeftCorner<3, 3>();
		plane.sum += pointCluster.topRightCorner<3, 1>();
		plane.weight += pointCluster(3, 3);
		if (plane.updateParams(newPlane.rmse))
		{
			plane.lastTimeBeSeen = updateCount_;
		}
		else
		{
			if (newPlane.rmse < oldPlane.rmse)
			{
				planes_.erase(planeID);
				planeID = currentLabel_++;
				planes_.insert({planeID, newPlane});
			}
			else
			{
				plane = oldPlane;
				plane.lastTimeBeSeen = updateCount_;
			}
		}
	}
	else // planeID = -1
	{
		planeID = currentLabel_++;
		planes_.insert({planeID, newPlane});
	}
}

void PlaneOccupiedVoxelMap::updatePlaneInfoBatch(std::vector<int> &planeIDs, const Vector4dMatrix &pointClusters, const Eigen::Vector3d &origin)
{
	for (size_t i = 0; i < pointClusters.size(); ++i)
	{
		Plane newPlane(pointClusters[i]);
		newPlane.updateParams();
		newPlane.anchor = origin;
		newPlane.lastTimeBeSeen = updateCount_;

		auto search = planes_.find(planeIDs[i]);
		if (search != planes_.end()) // existing plane
		{
			Plane oldPlane = search.value(); // trace back if update fails

			auto &plane = search.value();
			plane.moment += pointClusters[i].topLeftCorner<3, 3>();
			plane.sum += pointClusters[i].topRightCorner<3, 1>();
			plane.weight += pointClusters[i](3, 3);
			if (plane.updateParams(newPlane.rmse))
			{
				plane.lastTimeBeSeen = updateCount_;
			}
			else
			{
				if (newPlane.rmse < oldPlane.rmse)
				{
					planes_.erase(planeIDs[i]);
					planeIDs[i] = currentLabel_++;
					planes_.insert({planeIDs[i], newPlane});
				}
				else
				{
					plane = oldPlane;
					plane.lastTimeBeSeen = updateCount_;
				}
			}
		}
		else // planeID = -1
		{
			planeIDs[i] = currentLabel_++;
			planes_.insert({planeIDs[i], newPlane});
		}
	}
}

std::optional<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> PlaneOccupiedVoxelMap::getPlaneNormal(int planeID)
{
	auto search = planes_.find(planeID);
	if (search != planes_.end())
	{
		auto &plane = search.value();
		return std::make_pair(plane.normal, plane.D_nn);
	}
	return std::nullopt;
}

std::optional<std::pair<double, double>> PlaneOccupiedVoxelMap::getPlaneD(int planeID)
{
	auto search = planes_.find(planeID);
	if (search != planes_.end())
	{
		auto &plane = search.value();
		return std::make_pair(plane.d, plane.D_dd);
	}
	return std::nullopt;
}

void PlaneOccupiedVoxelMap::removePointsFarFromLocation(const Eigen::Vector3d &origin)
{
	++updateCount_;
	if (planes_.size() > 1000)
	{
		for (auto it = planes_.begin(); it != planes_.end(); )
		{
			if (updateCount_ - it->second.lastTimeBeSeen > 10) it = planes_.erase(it);
			else ++it;
		}
	}

	for (auto it = map_.begin(); it != map_.end(); )
	{
		const auto &voxel = it->first;
		Eigen::Vector3d center = voxelSize_ * voxel.cast<double>();
		if ((center - origin).squaredNorm() > maxDistance2_) it = map_.erase(it);
		else ++it;
	}
}

}
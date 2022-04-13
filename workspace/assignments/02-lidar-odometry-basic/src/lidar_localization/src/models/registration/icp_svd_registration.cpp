/*
 * @Description: ICP SVD lidar odometry
 * @Author: Ge Yao
 * @Date: 2020-10-24 21:46:45
 */

#include <pcl/common/transforms.h>
#include <pcl/cloud_iterator.h>

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "glog/logging.h"

#include "lidar_localization/models/registration/icp_svd_registration.hpp"

#include <numeric>


#include <pcl/keypoints/uniform_sampling.h>

namespace lidar_localization {

ICPSVDRegistration::ICPSVDRegistration(
    const YAML::Node& node
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    // parse params:
    float max_corr_dist = node["max_corr_dist"].as<float>();
    float trans_eps = node["trans_eps"].as<float>();
    float euc_fitness_eps = node["euc_fitness_eps"].as<float>();
    int max_iter = node["max_iter"].as<int>();

    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

ICPSVDRegistration::ICPSVDRegistration(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) : input_target_kdtree_(new pcl::KdTreeFLANN<pcl::PointXYZ>()) {
    SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
}

bool ICPSVDRegistration::SetRegistrationParam(
    float max_corr_dist, 
    float trans_eps, 
    float euc_fitness_eps, 
    int max_iter
) {
    // set params:
    max_corr_dist_ = max_corr_dist;
    trans_eps_ = trans_eps;
    euc_fitness_eps_ = euc_fitness_eps;
    max_iter_ = max_iter;

    LOG(INFO) << "ICP SVD params:" << std::endl
              << "max_corr_dist: " << max_corr_dist_ << ", "
              << "trans_eps: " << trans_eps_ << ", "
              << "euc_fitness_eps: " << euc_fitness_eps_ << ", "
              << "max_iter: " << max_iter_ 
              << std::endl << std::endl;

    return true;
}

bool ICPSVDRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
    input_target_ = input_target;
    input_target_kdtree_->setInputCloud(input_target_);

    return true;
}

bool ICPSVDRegistration::ScanMatch(
    const CloudData::CLOUD_PTR& input_source, 
    const Eigen::Matrix4f& predict_pose, 
    CloudData::CLOUD_PTR& result_cloud_ptr,
    Eigen::Matrix4f& result_pose
) {
    input_source_ = input_source;

    // pre-process input source:
    CloudData::CLOUD_PTR transformed_input_source(new CloudData::CLOUD());
    pcl::transformPointCloud(*input_source_, *transformed_input_source, predict_pose);
    

    // init estimation:
    transformation_.setIdentity();
    
    //
    // TODO: first option -- implement all computing logic on your own
    //
    // do estimation:
    int curr_iter = 0;

    Eigen::Matrix4f accumulate_transformation = Eigen::Matrix4f::Identity();
    while (curr_iter < max_iter_) {
        // TODO: apply current estimation:
        
        pcl::transformPointCloud(*transformed_input_source, *transformed_input_source, transformation_);


        // TODO: get correspondence:
        std::vector<Eigen::Vector3f> xs;
        std::vector<Eigen::Vector3f> ys;
        size_t n_corr = ICPSVDRegistration::GetCorrespondence(transformed_input_source,xs,ys);
        std::cout << "correspond points number: " << n_corr << std::endl;




        // TODO: do not have enough correspondence -- break:
        if(n_corr == 0){
            break;
        }


        // TODO: update current transform:
        // Eigen::Matrix4f curr_transformation;
        ICPSVDRegistration::GetTransform(xs,ys,transformation_);

        // TODO: whether the transformation update is significant:
        if(!ICPSVDRegistration::IsSignificant(transformation_,trans_eps_)){
            break;
        }

        // TODO: update transformation:
        accumulate_transformation = transformation_ * accumulate_transformation;    
        
        ++curr_iter;
    }

    // set output:
    result_pose = accumulate_transformation * predict_pose;
    // result_pose = transformation_ * predict_pose;
    pcl::transformPointCloud(*input_source_, *result_cloud_ptr, result_pose);
    
    return true;
}

size_t ICPSVDRegistration::GetCorrespondence(
    const CloudData::CLOUD_PTR &input_source,
    std::vector<Eigen::Vector3f> &xs,
    std::vector<Eigen::Vector3f> &ys
) {
    const float MAX_CORR_DIST_SQR = max_corr_dist_ * max_corr_dist_;

    size_t num_corr = 0;

    float corr_dist_sqr = 0;
    xs.clear();
    ys.clear();
    // TODO: set up point correspondence
    std::vector<int> neigbour_indices(1);
    std::vector<float> neigbour_distances(1);

    // filter the input point cloud to get less point and reduce caculation
    pcl::UniformSampling<pcl::PointXYZ> US;
    US.setInputCloud(input_source);
 
    US.setRadiusSearch(0.2f);
 
    US.filter(*input_source);


    // for each point in transpormed point cloud, search its neighbour in the input source cloud
    for(const auto & transformed_pt : *input_source){
        if(input_target_kdtree_->nearestKSearchT(transformed_pt,1,neigbour_indices,neigbour_distances) > 0){
            
            // skip the matched points which are too farway from each other
            if(neigbour_distances[0] > MAX_CORR_DIST_SQR){
                continue;
            }
            xs.emplace_back(Eigen::Vector3f(input_target_->points[neigbour_indices[0]].x,
            input_target_->points[neigbour_indices[0]].y,
            input_target_->points[neigbour_indices[0]].z));
            ys.emplace_back(Eigen::Vector3f(transformed_pt.x,transformed_pt.y,transformed_pt.z));
            num_corr++;
            corr_dist_sqr += neigbour_distances[0];
        }
    }

    corr_dist_sqr /= xs.size();
    std:: cout << "Average corresponding points dist: " << corr_dist_sqr << std::endl;
    // if(corr_dist_sqr > MAX_CORR_DIST_SQR){
    //     num_corr = 0;
    // }

    return num_corr;
}

void ICPSVDRegistration::GetTransform(
    const std::vector<Eigen::Vector3f> &xs,
    const std::vector<Eigen::Vector3f> &ys,
    Eigen::Matrix4f &transformation_
) {
    const size_t N = xs.size();

    // TODO: find centroids of mu_x and mu_y:
    Eigen::Vector3f mu_x = std::accumulate(xs.begin(),xs.end(),Eigen::Vector3f{0.,0.,0.});
    mu_x /= static_cast<float>(xs.size());
    Eigen::Vector3f mu_y = std::accumulate(ys.begin(),ys.end(),Eigen::Vector3f{0.,0.,0.});
    mu_y /= static_cast<float>(ys.size());


    // TODO: build H:
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for(int i = 0; i < xs.size(); i++){
        Eigen::Vector3f yi = ys[i] - mu_y;
        Eigen::Vector3f xi = xs[i] - mu_x;
        H += yi * xi.transpose();
    }

    
    // TODO: solve R:
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV );
    Eigen::Matrix3f V = svd.matrixV(), U = svd.matrixU();

    Eigen::Matrix3f R = V * U.transpose();

    // TODO: solve t:
    Eigen::Vector3f t = mu_x - R * mu_y;

    // TODO: set output:
    transformation_.block<3,1>(0,3) = t;
    transformation_.block<3,3>(0,0) = R;
}

bool ICPSVDRegistration::IsSignificant(
    const Eigen::Matrix4f &transformation,
    const float trans_eps
) {
    // a. translation magnitude -- norm:
    float translation_magnitude = transformation.block<3, 1>(0, 3).norm();
    // b. rotation magnitude -- angle:
    float rotation_magnitude = fabs(
        acos(
            (transformation.block<3, 3>(0, 0).trace() - 1.0f) / 2.0f
        )
    );

    return (
        (translation_magnitude > trans_eps) || 
        (rotation_magnitude > trans_eps)
    );
}

} // namespace lidar_localization
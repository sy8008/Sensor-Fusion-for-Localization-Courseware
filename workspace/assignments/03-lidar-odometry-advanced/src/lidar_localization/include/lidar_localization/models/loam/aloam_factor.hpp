// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

//
// TODO: implement analytic Jacobians for LOAM residuals in this file
// 

#include <eigen3/Eigen/Dense>

//
// TODO: Sophus is ready to use if you have a good undestanding of Lie algebra.
// 
#include <sophus/so3.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

//struct LidarEdgeFactor
//{
//	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
//					Eigen::Vector3d last_point_b_, double s_)
//		: curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}
//
//	template <typename T>
//	bool operator()(const T *q, const T *t, T *residual) const
//	{
//
//		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
//		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
//		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};
//
//		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
//		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
//		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
//		q_last_curr = q_identity.slerp(T(s), q_last_curr);
//		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};
//
//		Eigen::Matrix<T, 3, 1> lp;
//		lp = q_last_curr * cp + t_last_curr;
//
//		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
//		Eigen::Matrix<T, 3, 1> de = lpa - lpb;
//
//		residual[0] = nu.x() / de.norm();
//		residual[1] = nu.y() / de.norm();
//		residual[2] = nu.z() / de.norm();
//
//		return true;
//	}
//
//	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
//									   const Eigen::Vector3d last_point_b_, const double s_)
//	{
//		return (new ceres::AutoDiffCostFunction<
//				LidarEdgeFactor, 3, 4, 3>(
//			new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
//	}
//
//	Eigen::Vector3d curr_point, last_point_a, last_point_b;
//	double s;
//};

//第一个参数为残差块的维数3,第二和第三个参数为参数块的维数。分别为四元数代表旋转4维。平移矩阵3维
class LidarEdgeFactor : public ceres::SizedCostFunction<3, 4, 3> {
public:
    /*
    curr_point_:当前帧某点，
    last_point_a_:上一帧点a，
    last_point_b_:上一帧点b
    s_:阈值
    */
    LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
                                  Eigen::Vector3d last_point_b_, double s_)
            : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}
    virtual bool Evaluate(double const* const* parameters,double* residuals,double** jacobians) const
    {
        /*
        q_last_curr:四元数，
        t_last_curr:代表旋转，平移
        */
//        Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
//        Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[1]);

        Eigen::Quaterniond q_last_curr(parameters[0][3],parameters[0][0],parameters[0][1],parameters[0][2]);
        q_last_curr.normalize();
        Eigen::Vector3d t_last_curr(parameters[1][0],parameters[1][1],parameters[1][2]);


        Eigen::Vector3d lp;
        Eigen::Vector3d lp_r;
        lp_r = q_last_curr * curr_point; // for computing Jacobian of Rotation: dp_by_dr
        //将当前帧投影到上一帧：对应公式
        lp = q_last_curr * curr_point + t_last_curr; //new point
        //叉乘求四边形面积
        Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
        //上一帧两点连线向量
        Eigen::Vector3d de = last_point_a - last_point_b;
        //三个方向残差
        residuals[0] = nu.x() / de.norm();
        residuals[1] = nu.y() / de.norm();
        residuals[2] = nu.z() / de.norm();
        if(jacobians != NULL)
        {
            if(jacobians[0] != NULL)
            {
                Eigen::Vector3d re = last_point_b - last_point_a;
                // Eigen::Matrix3d skew_re = skew(re);
                //求解last_point_b - last_point_a的反对陈矩阵
                Eigen::Matrix3d skew_re;
                skew_re(0,0)=0;skew_re(0,1)=-re(2);skew_re(0,2)=re(1);
                skew_re(1,0)=re(2);skew_re(1,1)=0;skew_re(1,2)=-re(0);
                skew_re(2,0)=-re(1);skew_re(2,1)=re(0);skew_re(2,2)=0;
                //  旋转矩阵求导
                //Eigen::Matrix3d skew_lp_r = skew(lp_r);
                Eigen::Matrix3d skew_lp_r;
                //
                skew_lp_r(0,0)=0;skew_lp_r(0,1)=-lp(2);skew_lp_r(0,2)=lp(1);
                skew_lp_r(1,0)=lp(2);skew_lp_r(1,1)=0;skew_lp_r(1,2)=-lp(0);
                skew_lp_r(2,0)=-lp(1);skew_lp_r(2,1)=lp(0);skew_lp_r(2,2)=0;

                Eigen::Matrix<double, 3, 4> dp_by_dq;

                double lX = lp.x(), lY = lp.y(), lZ = lp.z();
                dp_by_dq << q_last_curr.y() * lY + q_last_curr.z() * lZ, q_last_curr.w() * lZ + q_last_curr.x() * lY - 2 * q_last_curr.y() * lX, -q_last_curr.w() * lY + q_last_curr.x() * lZ - 2 * q_last_curr.z() * lX, q_last_curr.y() * lZ - q_last_curr.z() * lY,
                        -q_last_curr.w() * lZ - 2 * q_last_curr.x() * lY + q_last_curr.y() * lX, q_last_curr.x() * lX + q_last_curr.z() * lZ, q_last_curr.w() * lX + q_last_curr.y() * lZ - 2 * q_last_curr.z() * lY, -q_last_curr.x() * lZ + q_last_curr.z() * lX,
                        q_last_curr.w() * lY - 2 * q_last_curr.x() * lZ + q_last_curr.z() * lX, - q_last_curr.w() * lX - 2 * q_last_curr.y() * lZ + q_last_curr.z() * lY, q_last_curr.x() * lX + q_last_curr.y() * lY, q_last_curr.x() * lY - q_last_curr.y() * lX;

                dp_by_dq = 2 * dp_by_dq;

                Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > J_so3_r(jacobians[0]);
                J_so3_r.setZero();

                J_so3_r = skew_re * dp_by_dq / de.norm();

                // 平移矩阵求导
                Eigen::Matrix<double, 3, 3> dp_by_dt;
                //初始化成单位矩阵
                (dp_by_dt.block<3,3>(0,0)).setIdentity();
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > J_so3_t(jacobians[1]);
                J_so3_t.setZero();
                J_so3_t.block<3,3>(0,0) = skew_re * dp_by_dt / de.norm();
            }
        }
        return true;

    }
protected:
    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
};

//
//struct LidarPlaneFactor
//{
//	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
//					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
//		: curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
//		  last_point_m(last_point_m_), s(s_)
//	{
//		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
//		ljm_norm.normalize();
//	}
//
//	template <typename T>
//	bool operator()(const T *q, const T *t, T *residual) const
//	{
//
//		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
//		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
//		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
//		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
//		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};
//
//		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
//		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
//		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
//		q_last_curr = q_identity.slerp(T(s), q_last_curr);
//		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};
//
//		Eigen::Matrix<T, 3, 1> lp;
//		lp = q_last_curr * cp + t_last_curr;
//
//		residual[0] = (lp - lpj).dot(ljm);
//
//		return true;
//	}
//
//	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
//									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
//									   const double s_)
//	{
//		return (new ceres::AutoDiffCostFunction<
//				LidarPlaneFactor, 1, 4, 3>(
//			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
//	}
//
//	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
//	Eigen::Vector3d ljm_norm;
//	double s;
//};

class LidarPlaneFactor : public ceres::SizedCostFunction<1, 4, 3> {
public:
    LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
                                   Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
            : curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_), last_point_m(last_point_m_), s(s_) {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const
    {
//        Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
//        Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[1]);

        Eigen::Quaterniond q_last_curr(parameters[0][3],parameters[0][0],parameters[0][1],parameters[0][2]);
        q_last_curr.normalize();
        Eigen::Vector3d t_last_curr(parameters[1][0],parameters[1][1],parameters[1][2]);

        Eigen::Vector3d lp;
        Eigen::Vector3d lp_r;
        lp_r = q_last_curr * curr_point; // for computing Jacobian of Rotation: dp_dr
        lp = q_last_curr * curr_point + t_last_curr; //new point
        Eigen::Vector3d de = (last_point_l-last_point_j).cross(last_point_m-last_point_j);
        double nu = (lp-last_point_j).dot(de);

        residuals[0] = nu / de.norm();
        if(jacobians != NULL)
        {
            if(jacobians[0] != NULL)
            {
                Eigen::Vector3d dX_dp = de / de.norm();
                double X = nu / de.norm();
                Eigen::Vector3d ddh_dp = X * dX_dp / std::abs(X);
                //  Rotation
                //Eigen::Matrix3d skew_lp_r = skew(lp_r);
                Eigen::Matrix3d skew_lp_r;
                skew_lp_r(0,0)=0;skew_lp_r(0,1)=-lp(2);skew_lp_r(0,2)=lp(1);
                skew_lp_r(1,0)=lp(2);skew_lp_r(1,1)=0;skew_lp_r(1,2)=-lp(0);
                skew_lp_r(2,0)=-lp(1);skew_lp_r(2,1)=lp(0);skew_lp_r(2,2)=0;


                Eigen::Matrix<double, 3, 4> dp_by_dq;

                double lX = lp.x(), lY = lp.y(), lZ = lp.z();
                dp_by_dq << q_last_curr.y() * lY + q_last_curr.z() * lZ, q_last_curr.w() * lZ + q_last_curr.x() * lY - 2 * q_last_curr.y() * lX, -q_last_curr.w() * lY + q_last_curr.x() * lZ - 2 * q_last_curr.z() * lX, q_last_curr.y() * lZ - q_last_curr.z() * lY,
                        -q_last_curr.w() * lZ - 2 * q_last_curr.x() * lY + q_last_curr.y() * lX, q_last_curr.x() * lX + q_last_curr.z() * lZ, q_last_curr.w() * lX + q_last_curr.y() * lZ - 2 * q_last_curr.z() * lY, -q_last_curr.x() * lZ + q_last_curr.z() * lX,
                        q_last_curr.w() * lY - 2 * q_last_curr.x() * lZ + q_last_curr.z() * lX, - q_last_curr.w() * lX - 2 * q_last_curr.y() * lZ + q_last_curr.z() * lY, q_last_curr.x() * lX + q_last_curr.y() * lY, q_last_curr.x() * lY - q_last_curr.y() * lX;

                dp_by_dq = 2 * dp_by_dq;
                Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor> > J_so3_r(jacobians[0]);
                J_so3_r.setZero();
                J_so3_r = ddh_dp.transpose() * dp_by_dq;
                // Translation
                Eigen::Matrix<double, 3, 3> dp_dt;
                (dp_dt.block<3,3>(0,0)).setIdentity();
                Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor> > J_so3_t(jacobians[1]);
                J_so3_t.setZero();
                J_so3_t.block<1,3>(0,0) = ddh_dp.transpose() * dp_dt;
            }
        }
        return true;

    }
protected:
    Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
    double s;
};



struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};


struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<
				LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};
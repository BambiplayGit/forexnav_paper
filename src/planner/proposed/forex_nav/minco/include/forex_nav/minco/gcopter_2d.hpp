/*
 * GCOPTER 2D - Full GCOPTER for 2D ground robot navigation
 * Based on: Zhepei Wang's GCOPTER (https://github.com/ZJU-FAST-Lab/GCOPTER)
 */

#ifndef GCOPTER_2D_HPP
#define GCOPTER_2D_HPP

#include "forex_nav/minco/minco.hpp"
#include "forex_nav/minco/trajectory.hpp"
#include "forex_nav/minco/lbfgs.hpp"
#include "forex_nav/minco/sfc_generator.hpp"

#include <Eigen/Eigen>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>

namespace forex_nav {

class GCopter2D {
public:
    GCopter2D() = default;
    
    struct OptConfig {
        double weight_time = 20.0;
        double weight_energy = 0.1;
        double weight_pos = 1000.0;
        double weight_vel = 100.0;
        double weight_acc = 100.0;
        double weight_jerk = 50.0;
        double weight_guide = 100.0;
        double smooth_eps = 0.01;
        int integral_resolution = 16;
        double max_vel = 4.0;
        double max_acc = 2.0;
        double max_jerk = 10.0;
        double alloc_speed = 3.0;
        double rel_cost_tol = 1e-4;
        double length_per_piece = 2.0;
    };
    
    void setConfig(const OptConfig& config) { config_ = config; }
    
    void setReferencePath(const std::vector<Eigen::Vector3d>& path) {
        refPath_ = path;
    }
    
    bool optimize(
        const std::vector<Eigen::Vector3d>& path,
        const std::vector<AABB2D>& corridors,
        const Eigen::Matrix3d& initPVA,
        const Eigen::Matrix3d& tailPVA,
        Trajectory<5>& traj) {
        
        if (path.size() < 2 || corridors.empty()) {
            std::cout << "[GCopter2D] Error: Invalid input" << std::endl;
            return false;
        }
        
        if (refPath_.empty()) {
            refPath_ = path;
        }
        
        headPVA_ = initPVA;
        tailPVA_ = tailPVA;
        
        corridors_ = corridors;
        hPolytopes_.clear();
        for (const auto& corr : corridors_) {
            hPolytopes_.push_back(corr.toHRep());
        }
        
        if (!setupFromPath(path)) {
            std::cout << "[GCopter2D] Setup failed" << std::endl;
            return false;
        }
        
        std::cout << "[GCopter2D] Optimizing: " << pieceN_ << " pieces, "
                  << "dimTau=" << temporalDim_ << ", dimXi=" << spatialDim_ << std::endl;
        
        minco_.setConditions(headPVA_, tailPVA_, pieceN_);
        
        points_.resize(3, pieceN_ - 1);
        times_.resize(pieceN_);
        gradByPoints_.resize(3, pieceN_ - 1);
        gradByTimes_.resize(pieceN_);
        partialGradByCoeffs_.resize(6 * pieceN_, 3);
        partialGradByTimes_.resize(pieceN_);
        
        Eigen::VectorXd x(temporalDim_ + spatialDim_);
        
        setInitialAllocation();
        backwardT(times_, x.head(temporalDim_));
        
        setInitialPoints();
        backwardP(points_, x.segment(temporalDim_, spatialDim_));
        
        lbfgs::lbfgs_parameter_t params;
        params.mem_size = 64;
        params.past = 3;
        params.delta = config_.rel_cost_tol;
        params.g_epsilon = 0.0;
        params.max_iterations = 200;
        
        double minCost;
        int ret = lbfgs::lbfgs_optimize(
            x, minCost,
            &GCopter2D::costFunctional,
            nullptr, nullptr,
            this, params);
        
        if (ret >= 0 || ret == lbfgs::LBFGSERR_MAXIMUMLINESEARCH || 
            ret == lbfgs::LBFGSERR_MAXIMUMITERATION) {
            forwardT(x.head(temporalDim_), times_);
            forwardP(x.segment(temporalDim_, spatialDim_), points_);
            
            minco_.setParameters(points_, times_);
            minco_.getTrajectory(traj);
            
            std::cout << "[GCopter2D] Optimization completed: cost=" << minCost 
                      << ", duration=" << traj.getTotalDuration() << "s, ret=" << ret << std::endl;
            return true;
        } else {
            std::cout << "[GCopter2D] Optimization failed: " 
                      << lbfgs::lbfgs_strerror(ret) << std::endl;
            
            minco_.setParameters(points_, times_);
            minco_.getTrajectory(traj);
            return traj.getPieceNum() > 0;
        }
    }

private:
    OptConfig config_;
    minco::MINCO_S3NU minco_;
    
    int pieceN_;
    int temporalDim_;
    int spatialDim_;
    
    Eigen::Matrix3d headPVA_;
    Eigen::Matrix3d tailPVA_;
    std::vector<AABB2D> corridors_;
    std::vector<Eigen::MatrixX4d> hPolytopes_;
    
    Eigen::VectorXi pieceIdx_;
    Eigen::VectorXi hPolyIdx_;
    
    Eigen::Matrix3Xd shortPath_;
    std::vector<Eigen::Vector3d> refPath_;
    
    Eigen::Matrix3Xd points_;
    Eigen::VectorXd times_;
    Eigen::Matrix3Xd gradByPoints_;
    Eigen::VectorXd gradByTimes_;
    Eigen::MatrixX3d partialGradByCoeffs_;
    Eigen::VectorXd partialGradByTimes_;
    
    bool setupFromPath(const std::vector<Eigen::Vector3d>& path) {
        int polyN = corridors_.size();
        
        shortPath_.resize(3, polyN + 1);
        shortPath_.col(0) = headPVA_.col(0);
        for (int i = 0; i < polyN - 1 && i + 1 < static_cast<int>(path.size()); ++i) {
            shortPath_.col(i + 1) = path[i + 1];
        }
        shortPath_.col(polyN) = tailPVA_.col(0);
        
        pieceIdx_.resize(polyN);
        for (int i = 0; i < polyN; ++i) {
            Eigen::Vector3d delta = shortPath_.col(i + 1) - shortPath_.col(i);
            double len = delta.norm();
            int numPieces = std::max(1, static_cast<int>(std::ceil(len / config_.length_per_piece)));
            pieceIdx_(i) = numPieces;
        }
        
        pieceN_ = pieceIdx_.sum();
        temporalDim_ = pieceN_;
        
        hPolyIdx_.resize(pieceN_);
        int pieceCount = 0;
        for (int i = 0; i < polyN; ++i) {
            for (int j = 0; j < pieceIdx_(i); ++j) {
                hPolyIdx_(pieceCount++) = i;
            }
        }
        
        spatialDim_ = 3 * (pieceN_ - 1);
        
        return pieceN_ >= 1;
    }
    
    void setInitialAllocation() {
        int polyN = corridors_.size();
        int pieceCount = 0;
        
        for (int i = 0; i < polyN; ++i) {
            Eigen::Vector3d start_pt = shortPath_.col(i);
            Eigen::Vector3d end_pt = shortPath_.col(i + 1);
            double segLen = (end_pt - start_pt).norm();
            double segTime = segLen / config_.alloc_speed;
            double pieceTime = std::max(segTime / pieceIdx_(i), 0.1);
            
            for (int j = 0; j < pieceIdx_(i); ++j) {
                times_(pieceCount++) = pieceTime;
            }
        }
        
        std::cout << "[GCopter2D] Initial time: " << times_.sum() << "s" << std::endl;
    }
    
    void setInitialPoints() {
        int polyN = corridors_.size();
        int pointCount = 0;
        
        for (int i = 0; i < polyN; ++i) {
            Eigen::Vector3d start_pt = shortPath_.col(i);
            Eigen::Vector3d end_pt = shortPath_.col(i + 1);
            int numPieces = pieceIdx_(i);
            
            for (int j = 0; j < numPieces - 1; ++j) {
                double alpha = (j + 1.0) / numPieces;
                points_.col(pointCount++) = start_pt + alpha * (end_pt - start_pt);
            }
            
            if (i < polyN - 1 && pointCount < pieceN_ - 1) {
                points_.col(pointCount++) = end_pt;
            }
        }
        
        while (pointCount < pieceN_ - 1) {
            double alpha = (pointCount + 1.0) / pieceN_;
            points_.col(pointCount) = headPVA_.col(0) + alpha * (tailPVA_.col(0) - headPVA_.col(0));
            pointCount++;
        }
    }
    
    static void forwardT(const Eigen::VectorXd& tau, Eigen::VectorXd& T) {
        const int n = tau.size();
        T.resize(n);
        for (int i = 0; i < n; ++i) {
            T(i) = tau(i) > 0.0
                ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
        }
    }
    
    template <typename EIGENVEC>
    static void backwardT(const Eigen::VectorXd& T, EIGENVEC tau) {
        const int n = T.size();
        for (int i = 0; i < n; ++i) {
            tau(i) = T(i) > 1.0
                ? (sqrt(2.0 * T(i) - 1.0) - 1.0)
                : (1.0 - sqrt(2.0 / T(i) - 1.0));
        }
    }
    
    template <typename EIGENVEC>
    static void backwardGradT(const Eigen::VectorXd& tau,
                              const Eigen::VectorXd& gradT,
                              EIGENVEC gradTau) {
        const int n = tau.size();
        for (int i = 0; i < n; ++i) {
            if (tau(i) > 0) {
                gradTau(i) = gradT(i) * (tau(i) + 1.0);
            } else {
                double den = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
                gradTau(i) = gradT(i) * (1.0 - tau(i)) / (den * den);
            }
        }
    }
    
    void forwardP(const Eigen::VectorXd& xi, Eigen::Matrix3Xd& P) {
        const int numPoints = pieceN_ - 1;
        P.resize(3, numPoints);
        
        for (int i = 0; i < numPoints; ++i) {
            P(0, i) = xi(3 * i + 0);
            P(1, i) = xi(3 * i + 1);
            P(2, i) = xi(3 * i + 2);
        }
    }
    
    template <typename EIGENVEC>
    void backwardP(const Eigen::Matrix3Xd& P, EIGENVEC xi) {
        const int numPoints = P.cols();
        
        for (int i = 0; i < numPoints; ++i) {
            xi(3 * i + 0) = P(0, i);
            xi(3 * i + 1) = P(1, i);
            xi(3 * i + 2) = P(2, i);
        }
    }
    
    template <typename EIGENVEC>
    void backwardGradP(const Eigen::Matrix3Xd& gradP, EIGENVEC gradXi) {
        const int numPoints = gradP.cols();
        for (int i = 0; i < numPoints; ++i) {
            gradXi(3 * i + 0) = gradP(0, i);
            gradXi(3 * i + 1) = gradP(1, i);
            gradXi(3 * i + 2) = gradP(2, i);
        }
    }
    
    static bool smoothedL1(double x, double mu, double& f, double& df) {
        if (x < 0.0) {
            return false;
        } else if (x > mu) {
            f = x - 0.5 * mu;
            df = 1.0;
            return true;
        } else {
            double xdmu = x / mu;
            double sqrxdmu = xdmu * xdmu;
            double mumxd2 = mu - 0.5 * x;
            f = mumxd2 * sqrxdmu * xdmu;
            df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
            return true;
        }
    }
    
    Eigen::Vector3d closestPointOnRefPath(const Eigen::Vector3d& pt) const {
        Eigen::Vector3d best = refPath_[0];
        double bestDistSq = (pt - best).squaredNorm();
        for (size_t i = 0; i + 1 < refPath_.size(); ++i) {
            const Eigen::Vector3d& a = refPath_[i];
            const Eigen::Vector3d& b = refPath_[i + 1];
            Eigen::Vector3d ab = b - a;
            double lenSq = ab.squaredNorm();
            double t = (lenSq > 1e-12) ? std::max(0.0, std::min(1.0, (pt - a).dot(ab) / lenSq)) : 0.0;
            Eigen::Vector3d proj = a + t * ab;
            double dSq = (pt - proj).squaredNorm();
            if (dSq < bestDistSq) {
                bestDistSq = dSq;
                best = proj;
            }
        }
        return best;
    }
    
    void addPenalty(double& cost, Eigen::VectorXd& gradT, Eigen::MatrixX3d& gradC) {
        const double velSqrMax = config_.max_vel * config_.max_vel;
        const double accSqrMax = config_.max_acc * config_.max_acc;
        const double jerkSqrMax = config_.max_jerk * config_.max_jerk;
        
        Eigen::Vector3d pos, vel, acc, jer, sna;
        Eigen::Vector3d gradPos, gradVel, gradAcc, gradJerk;
        
        double step, alpha;
        double s1, s2, s3, s4, s5;
        Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;
        Eigen::Vector3d outerNormal;
        double node, pena;
        double violaPos, violaVel, violaAcc, violaJerk;
        double violaPosPena, violaVelPena, violaAccPena, violaJerkPena;
        double violaPosPenaD, violaVelPenaD, violaAccPenaD, violaJerkPenaD;
        
        const Eigen::MatrixX3d& coeffs = minco_.getCoeffs();
        const double integralFrac = 1.0 / config_.integral_resolution;
        
        for (int i = 0; i < pieceN_; ++i) {
            const Eigen::Matrix<double, 6, 3>& c = coeffs.block<6, 3>(i * 6, 0);
            step = times_(i) * integralFrac;
            
            int corridorIdx = hPolyIdx_(i);
            const Eigen::MatrixX4d& hPoly = hPolytopes_[corridorIdx];
            int K = hPoly.rows();
            
            for (int j = 0; j <= config_.integral_resolution; ++j) {
                s1 = j * step;
                s2 = s1 * s1;
                s3 = s2 * s1;
                s4 = s2 * s2;
                s5 = s4 * s1;
                
                beta0 << 1.0, s1, s2, s3, s4, s5;
                beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
                beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
                beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
                beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1;
                
                pos = c.transpose() * beta0;
                vel = c.transpose() * beta1;
                acc = c.transpose() * beta2;
                jer = c.transpose() * beta3;
                sna = c.transpose() * beta4;
                
                violaVel = vel.squaredNorm() - velSqrMax;
                violaAcc = acc.squaredNorm() - accSqrMax;
                violaJerk = jer.squaredNorm() - jerkSqrMax;
                
                gradPos.setZero();
                gradVel.setZero();
                gradAcc.setZero();
                gradJerk.setZero();
                pena = 0.0;
                
                for (int k = 0; k < K; ++k) {
                    outerNormal = hPoly.block<1, 3>(k, 0).transpose();
                    violaPos = outerNormal.dot(pos) + hPoly(k, 3);
                    if (smoothedL1(violaPos, config_.smooth_eps, violaPosPena, violaPosPenaD)) {
                        gradPos += config_.weight_pos * violaPosPenaD * outerNormal;
                        pena += config_.weight_pos * violaPosPena;
                    }
                }
                
                if (config_.weight_guide > 0.0 && refPath_.size() >= 2) {
                    Eigen::Vector3d closest = closestPointOnRefPath(pos);
                    Eigen::Vector3d diff = pos - closest;
                    double distSq = diff.squaredNorm();
                    gradPos += config_.weight_guide * 2.0 * diff;
                    pena += config_.weight_guide * distSq;
                }
                
                if (smoothedL1(violaVel, config_.smooth_eps, violaVelPena, violaVelPenaD)) {
                    gradVel += config_.weight_vel * violaVelPenaD * 2.0 * vel;
                    pena += config_.weight_vel * violaVelPena;
                }
                
                if (smoothedL1(violaAcc, config_.smooth_eps, violaAccPena, violaAccPenaD)) {
                    gradAcc += config_.weight_acc * violaAccPenaD * 2.0 * acc;
                    pena += config_.weight_acc * violaAccPena;
                }
                
                if (smoothedL1(violaJerk, config_.smooth_eps, violaJerkPena, violaJerkPenaD)) {
                    gradJerk += config_.weight_jerk * violaJerkPenaD * 2.0 * jer;
                    pena += config_.weight_jerk * violaJerkPena;
                }
                
                node = (j == 0 || j == config_.integral_resolution) ? 0.5 : 1.0;
                alpha = j * integralFrac;
                
                gradC.block<6, 3>(i * 6, 0) += (beta0 * gradPos.transpose() +
                                                 beta1 * gradVel.transpose() +
                                                 beta2 * gradAcc.transpose() +
                                                 beta3 * gradJerk.transpose()) *
                                                node * step;
                
                gradT(i) += (gradPos.dot(vel) +
                            gradVel.dot(acc) +
                            gradAcc.dot(jer) +
                            gradJerk.dot(sna)) *
                            alpha * node * step +
                            node * integralFrac * pena;
                
                cost += node * step * pena;
            }
        }
    }
    
    static double costFunctional(void* ptr,
                                 const Eigen::VectorXd& x,
                                 Eigen::VectorXd& g) {
        GCopter2D& obj = *(GCopter2D*)ptr;
        
        const int dimTau = obj.temporalDim_;
        const int dimXi = obj.spatialDim_;
        
        Eigen::VectorXd tau = x.head(dimTau);
        Eigen::VectorXd xi = x.segment(dimTau, dimXi);
        
        forwardT(tau, obj.times_);
        obj.forwardP(xi, obj.points_);
        
        obj.minco_.setParameters(obj.points_, obj.times_);
        
        double cost;
        obj.minco_.getEnergy(cost);
        obj.minco_.getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs_);
        obj.minco_.getEnergyPartialGradByTimes(obj.partialGradByTimes_);
        
        cost *= obj.config_.weight_energy;
        obj.partialGradByCoeffs_ *= obj.config_.weight_energy;
        obj.partialGradByTimes_ *= obj.config_.weight_energy;
        
        obj.addPenalty(cost, obj.partialGradByTimes_, obj.partialGradByCoeffs_);
        
        obj.minco_.propogateGrad(obj.partialGradByCoeffs_, obj.partialGradByTimes_,
                                  obj.gradByPoints_, obj.gradByTimes_);
        
        cost += obj.config_.weight_time * obj.times_.sum();
        obj.gradByTimes_.array() += obj.config_.weight_time;
        
        g.resize(dimTau + dimXi);
        backwardGradT(tau, obj.gradByTimes_, g.head(dimTau));
        obj.backwardGradP(obj.gradByPoints_, g.segment(dimTau, dimXi));
        
        return cost;
    }
};

}  // namespace forex_nav

#endif  // GCOPTER_2D_HPP

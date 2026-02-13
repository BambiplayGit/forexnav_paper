#pragma once

#include "Highs.h"
#include "Eigen/Dense"

using namespace std;

class SDPsolver
{
public:
    SDPsolver() { gradient.resize(6); }
    ~SDPsolver() = default;

    int buildSDP(const string &envname, Eigen::MatrixXd &A, Eigen::VectorXd &b);
    Eigen::VectorXd solveSDP(const string &envname, Eigen::MatrixXd &A, Eigen::VectorXd &b, 
                              Eigen::MatrixXd &F, Eigen::VectorXd &g, 
                              Eigen::VectorXd &q, Eigen::VectorXd &c);
    
    Eigen::VectorXd getGradient() { return gradient; }
    
    double getAlpha() { return alpha_value; }

private:
    Highs highs;
    Eigen::VectorXd gradient;
    double alpha_value;
};

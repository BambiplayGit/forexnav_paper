#include "SDPsolver/SDPsolver.h"
#include <chrono>
using namespace std;

int SDPsolver::buildSDP(const string &envname, Eigen::MatrixXd &A, Eigen::VectorXd &b)
{
    return 0;
}

// envname: name of the environment
// A: matrix A of robot in robot's frame
// b: vector b of robot in robot's frame
// F: matrix F of free_region in world's frame
// g: vector g of free_region in world's frame
// q: vector q of robot's pose in world's frame
// c: vector c of free_region's center in world's frame
// return: vector of gradient of the cofficient subject to the q and alpha
Eigen::VectorXd SDPsolver::solveSDP(const string &envname, Eigen::MatrixXd &A, Eigen::VectorXd &b, 
                                     Eigen::MatrixXd &F, Eigen::VectorXd &g, 
                                     Eigen::VectorXd &q, Eigen::VectorXd &c)
{
    // Input validation
    if (b.size() == 0 || g.size() == 0 || q.size() < 6 || c.size() < 6 ||
        A.rows() == 0 || A.cols() < 3 || F.rows() == 0 || F.cols() < 3) {
        // Invalid input, return safe default
        gradient.setZero();
        alpha_value = 1.0;
        Eigen::VectorXd result(7);
        result << gradient, alpha_value;
        return result;
    }

    // Disable HiGHS output
    highs.setOptionValue("output_flag", false);
    
    int n_robot_faces = b.size();
    int n_free_faces = g.size();
    
    // q is x y z roll pitch yaw
    // Get rotation matrix and translation vector
    Eigen::Vector3d w_p_free;
    w_p_free << c[0], c[1], c[2];
    Eigen::Vector3d w_p_robot;
    w_p_robot << q[0], q[1], q[2];
    Eigen::Matrix3d w_R_free;
    w_R_free = Eigen::AngleAxisd(c[5], Eigen::Vector3d::UnitZ())
                * Eigen::AngleAxisd(c[4], Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(c[3], Eigen::Vector3d::UnitX());
    Eigen::Matrix3d w_R_robot;
    w_R_robot = Eigen::AngleAxisd(q[5], Eigen::Vector3d::UnitZ())
                * Eigen::AngleAxisd(q[4], Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(q[3], Eigen::Vector3d::UnitX());
    Eigen::Matrix3d free_R_robot;
    free_R_robot = w_R_free.transpose() * w_R_robot;
    Eigen::Vector3d free_p_robot;
    free_p_robot = w_R_free.transpose() * (w_p_robot - w_p_free);

    // Transform F and g to free region's frame
    Eigen::MatrixXd F_free = F * w_R_free;
    Eigen::VectorXd g_free = g - F * (-w_R_free * (-w_R_free.transpose() * w_p_free));

    // F_body = -F_free * free_R_robot
    Eigen::MatrixXd F_body = -F_free * free_R_robot;

    // Variables: alpha (index 0), then vars (sigma variables)
    // Total variables: 1 + n_free_faces * (1 + n_robot_faces)
    int n_vars = 1 + n_free_faces * (1 + n_robot_faces);
    int n_constraints = n_free_faces * 4;

    // Build LP model
    HighsLp lp;
    lp.num_col_ = n_vars;
    lp.num_row_ = n_constraints;
    lp.sense_ = ObjSense::kMinimize;

    // Variable bounds: all >= 0
    lp.col_lower_.resize(n_vars, 0.0);
    lp.col_upper_.resize(n_vars, kHighsInf);

    // Objective: minimize alpha (only alpha has coefficient 1)
    lp.col_cost_.resize(n_vars, 0.0);
    lp.col_cost_[0] = 1.0;  // alpha

    // Build constraint matrix in row-wise format first, then convert
    // Constraint format: A*x = b (equality constraints)
    vector<double> row_lower(n_constraints);
    vector<double> row_upper(n_constraints);
    
    // Using triplet format to build sparse matrix
    vector<int> row_indices;
    vector<int> col_indices;
    vector<double> values;

    for (int i = 0; i < n_free_faces; i++)
    {
        int constraint_base = i * 4;
        int var_base = 1 + i * (n_robot_faces + 1);  // +1 for alpha

        // Constraint 0: alpha * g_free[i] - vars[var_base] - sum(b[k] * vars[var_base+k+1]) = F_free.row(i) * free_p_robot
        double rhs0 = F_free.row(i).dot(free_p_robot);
        row_lower[constraint_base] = rhs0;
        row_upper[constraint_base] = rhs0;
        
        // alpha coefficient
        row_indices.push_back(constraint_base);
        col_indices.push_back(0);
        values.push_back(g_free[i]);
        
        // vars[var_base] coefficient: -1
        row_indices.push_back(constraint_base);
        col_indices.push_back(var_base);
        values.push_back(-1.0);
        
        // vars[var_base+k+1] coefficient: -b[k]
        for (int k = 0; k < n_robot_faces; k++)
        {
            row_indices.push_back(constraint_base);
            col_indices.push_back(var_base + k + 1);
            values.push_back(-b[k]);
        }

        // Constraints 1,2,3: sum(A(k,j) * vars[var_base+k+1]) = -F_body(i,j)
        for (int j = 0; j < 3; j++)
        {
            double rhs_j = -F_body(i, j);
            row_lower[constraint_base + j + 1] = rhs_j;
            row_upper[constraint_base + j + 1] = rhs_j;
            
            for (int k = 0; k < n_robot_faces; k++)
            {
                row_indices.push_back(constraint_base + j + 1);
                col_indices.push_back(var_base + k + 1);
                values.push_back(A(k, j));
            }
        }
    }

    lp.row_lower_ = row_lower;
    lp.row_upper_ = row_upper;

    // Convert triplet format to CSC (column-wise sparse) format
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.start_.resize(n_vars + 1, 0);
    
    // Count entries per column
    for (size_t i = 0; i < col_indices.size(); i++)
    {
        lp.a_matrix_.start_[col_indices[i] + 1]++;
    }
    // Cumulative sum
    for (int i = 1; i <= n_vars; i++)
    {
        lp.a_matrix_.start_[i] += lp.a_matrix_.start_[i - 1];
    }

    // Fill values
    lp.a_matrix_.index_.resize(values.size());
    lp.a_matrix_.value_.resize(values.size());
    vector<int> col_pos = lp.a_matrix_.start_;
    
    for (size_t i = 0; i < values.size(); i++)
    {
        int col = col_indices[i];
        int pos = col_pos[col]++;
        lp.a_matrix_.index_[pos] = row_indices[i];
        lp.a_matrix_.value_[pos] = values[i];
    }

    // Pass model and solve
    highs.passModel(lp);
    HighsStatus status = highs.run();

    // Check solve status
    if (status != HighsStatus::kOk) {
        // Solve failed, return safe default values
        gradient.setZero();
        alpha_value = 1.0;  // Safe default (constraint satisfied)
        highs.clear();
        Eigen::VectorXd result(7);
        result << gradient, alpha_value;
        return result;
    }

    // Check if solution is valid
    const HighsSolution& solution = highs.getSolution();
    if (solution.row_dual.empty() || solution.col_value.empty() ||
        (int)solution.row_dual.size() < n_constraints || 
        (int)solution.col_value.size() < 1) {
        // Invalid solution, return safe default values
        gradient.setZero();
        alpha_value = 1.0;
        highs.clear();
        Eigen::VectorXd result(7);
        result << gradient, alpha_value;
        return result;
    }

    Eigen::VectorXd dual_value(n_constraints);
    for (int i = 0; i < n_constraints; i++)
    {
        dual_value[i] = solution.row_dual[i];
    }

    // Calculate gradient (same as original)
    gradient.setZero();
    for (int i = 0; i < n_free_faces; i++)
    {
        gradient(0) += (F_free(i,0)) * dual_value[i*4+0] * w_R_free(0, 0) + (F_free(i,1)) * dual_value[i*4+0] * w_R_free(0, 1) + (F_free(i,2)) * dual_value[i*4+0]* w_R_free(0, 2);
        gradient(1) += (F_free(i,0)) * dual_value[i*4+0] * w_R_free(1, 0) + (F_free(i,1)) * dual_value[i*4+0] * w_R_free(1, 1) + (F_free(i,2)) * dual_value[i*4+0]* w_R_free(1, 2);
        gradient(2) += (F_free(i,0)) * dual_value[i*4+0] * w_R_free(2, 0) + (F_free(i,1)) * dual_value[i*4+0] * w_R_free(2, 1) + (F_free(i,2)) * dual_value[i*4+0]* w_R_free(2, 2);
        
        gradient(3) += (F_free(i,0)) * dual_value[i*4+2] * (w_R_free(0, 0) * (sin(q[5])*sin(q[4]) + cos(q[5])*cos(q[4])*sin(q[3])) - w_R_free(1, 0) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5])) + w_R_free(2, 0) * cos(q[3])*cos(q[4])) + 
                       (F_free(i,1)) * dual_value[i*4+2] * (w_R_free(0, 1) * (sin(q[5])*sin(q[4]) + cos(q[5])*cos(q[4])*sin(q[3])) - w_R_free(1, 1) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5])) + w_R_free(2, 1) * cos(q[3])*cos(q[4])) + 
                       (F_free(i,2)) * dual_value[i*4+2] * (w_R_free(0, 2) * (sin(q[5])*sin(q[4]) + cos(q[5])*cos(q[4])*sin(q[3])) - w_R_free(1, 2) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5])) + w_R_free(2, 2) * cos(q[3])*cos(q[4])) - 
                       (F_free(i,0)) * dual_value[i*4+3] * (w_R_free(1, 0) * (cos(q[4])*cos(q[5]) + sin(q[3])*sin(q[4])*sin(q[5])) - w_R_free(0, 0) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5])) + w_R_free(2, 0) * cos(q[3])*sin(q[4])) - 
                       (F_free(i,1)) * dual_value[i*4+3] * (w_R_free(1, 1) * (cos(q[4])*cos(q[5]) + sin(q[3])*sin(q[4])*sin(q[5])) - w_R_free(0, 1) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5])) + w_R_free(2, 1) * cos(q[3])*sin(q[4])) -
                       (F_free(i,2)) * dual_value[i*4+3] * (w_R_free(1, 2) * (cos(q[4])*cos(q[5]) + sin(q[3])*sin(q[4])*sin(q[5])) - w_R_free(0, 2) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5])) + w_R_free(2, 2) * cos(q[3])*sin(q[4]));
        
        gradient(4) += (F_free(i,0)) * dual_value[i*4+3] * (w_R_free(0, 0) * cos(q[4])*cos(q[3])*cos(q[5]) - w_R_free(2, 0) * cos(q[3])*sin(q[4]) + w_R_free(1, 0) * cos(q[4])*cos(q[3])*sin(q[5])) - 
                       (F_free(i,1)) * dual_value[i*4+1] * (w_R_free(2, 1) * cos(q[4]) + w_R_free(0, 1) * cos(q[5])*sin(q[4]) + w_R_free(1, 1) * sin(q[4])*sin(q[5])) - 
                       (F_free(i,2)) * dual_value[i*4+1] * (w_R_free(2, 2) * cos(q[4]) + w_R_free(0, 2) * cos(q[5])*sin(q[4]) + w_R_free(1, 2) * sin(q[4])*sin(q[5])) - 
                       (F_free(i,0)) * dual_value[i*4+1] * (w_R_free(2, 0) * cos(q[4]) + w_R_free(0, 0) * cos(q[5])*sin(q[4]) + w_R_free(1, 0) * sin(q[4])*sin(q[5])) + 
                       (F_free(i,1)) * dual_value[i*4+3] * (w_R_free(0, 1) * cos(q[4])*cos(q[3])*cos(q[5]) - w_R_free(2, 1) * cos(q[3])*sin(q[4]) + w_R_free(1, 1) * cos(q[4])*cos(q[3])*sin(q[5])) + 
                       (F_free(i,2)) * dual_value[i*4+3] * (w_R_free(0, 2) * cos(q[4])*cos(q[3])*cos(q[5]) - w_R_free(2, 2) * cos(q[3])*sin(q[4]) + w_R_free(1, 2) * cos(q[4])*cos(q[3])*sin(q[5])) + 
                       (F_free(i,0)) * dual_value[i*4+2] * (w_R_free(0, 0) * cos(q[4])*cos(q[3])*cos(q[5]) - w_R_free(2, 0) * cos(q[3])*sin(q[4]) + w_R_free(1, 0) * cos(q[4])*cos(q[3])*sin(q[5])) +
                       (F_free(i,1)) * dual_value[i*4+2] * (w_R_free(0, 1) * cos(q[4])*cos(q[3])*cos(q[5]) - w_R_free(2, 1) * cos(q[3])*sin(q[4]) + w_R_free(1, 1) * cos(q[4])*cos(q[3])*sin(q[5])) +
                       (F_free(i,2)) * dual_value[i*4+2] * (w_R_free(0, 2) * cos(q[4])*cos(q[3])*cos(q[5]) - w_R_free(2, 2) * cos(q[3])*sin(q[4]) + w_R_free(1, 2) * cos(q[4])*cos(q[3])*sin(q[5]));
        
        gradient(5) += (F_free(i,0)) * dual_value[i*4+3] * (w_R_free(0, 0) * (cos(q[5])*sin(q[3]) - cos(q[3])*sin(q[4])*sin(q[5])) + w_R_free(1, 0) * (sin(q[5])*sin(q[4]) + cos(q[5])*cos(q[4])*sin(q[3]))) - 
                       (F_free(i,0)) * dual_value[i*4+2] * (w_R_free(0, 0) * (cos(q[4])*cos(q[5]) + sin(q[3])*sin(q[4])*sin(q[5])) + w_R_free(1, 0) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5]))) - 
                       (F_free(i,1)) * dual_value[i*4+2] * (w_R_free(0, 1) * (cos(q[4])*cos(q[5]) + sin(q[3])*sin(q[4])*sin(q[5])) + w_R_free(1, 1) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5]))) - 
                       (F_free(i,1)) * dual_value[i*4+3] * (w_R_free(0, 1) * (cos(q[5])*sin(q[3]) - cos(q[3])*sin(q[4])*sin(q[5])) + w_R_free(1, 1) * (sin(q[5])*sin(q[4]) + cos(q[5])*cos(q[4])*sin(q[3]))) - 
                       (F_free(i,2)) * dual_value[i*4+2] * (w_R_free(0, 2) * (cos(q[4])*cos(q[5]) + sin(q[3])*sin(q[4])*sin(q[5])) + w_R_free(1, 2) * (cos(q[5])*sin(q[4]) - cos(q[4])*sin(q[3])*sin(q[5]))) - 
                       (F_free(i,2)) * dual_value[i*4+3] * (w_R_free(0, 2) * (cos(q[5])*sin(q[3]) - cos(q[3])*sin(q[4])*sin(q[5])) + w_R_free(1, 2) * (sin(q[5])*sin(q[4]) + cos(q[5])*cos(q[4])*sin(q[3]))) - 
                       (F_free(i,0)) * dual_value[i*4+1] * (w_R_free(1, 0) * cos(q[4])*cos(q[3]) - w_R_free(0, 0) * cos(q[3]*sin(q[5]))) -
                       (F_free(i,1)) * dual_value[i*4+1] * (w_R_free(1, 1) * cos(q[4])*cos(q[3]) - w_R_free(0, 1) * cos(q[3]*sin(q[5]))) -
                       (F_free(i,2)) * dual_value[i*4+1] * (w_R_free(1, 2) * cos(q[4])*cos(q[3]) - w_R_free(0, 2) * cos(q[3]*sin(q[5])));
    }

    // Get alpha value
    alpha_value = solution.col_value[0];

    // Clear model for next solve
    highs.clear();

    Eigen::VectorXd result(7);
    result << gradient, alpha_value;
    return result;
}

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace Eigen;
using namespace std;

// 参数设置
const double PI = 3.1415926;
const double a = 0.024;
const double b = 0.024;
const int n_l = 11;
const int n_c = 6;
const int num_eigen = 30;
const double th[11] = {0.0008, 0.0009, 0.0012, 0.0013, 0.00131, 
                       0.00206, 0.00221, 0.00521, 0.00531, 0.00731, 0.01031};
const double layer_starts[11] = {0, 0.0008, 0.0009, 0.0012, 0.0013, 
                                 0.00131, 0.00206, 0.00221, 0.00521, 0.00531, 0.00731};
//各项异性热导率
const double ka_anisotropic[11][3] = {
    {2, 2, 0.4},
    {29.75, 29.75, 35.36},
    {102, 102, 61.50},
    {10, 10, 80.275},
    {1.5, 1.5, 1.5},
    {140, 140, 140},
    {30, 30, 30},
    {400, 400, 400},
    {10, 10, 10},
    {400, 400, 400},
    {400, 400, 400}
};

double kx[11], ky[11], kz[11];

// 热源数据信息
struct HeatSourceData {
    vector<double> x_starts, x_ends, y_starts, y_ends, power_densities;
};
//COMSOL 对比数据信息
struct CoMSOLData {
    vector<double> x, y, z, temp;
};

inline double compute_integral_segment(double start, double end, double L, int mode) {
    if (mode == 0) {
        return end - start;
    }
    double k = mode * PI / L;
    return (L / (mode * PI)) * (sin(k * end) - sin(k * start));
}

// 导入热源信息
HeatSourceData load_thermal_data() {
    HeatSourceData data;
    ifstream file("E:/hot/multi-layer/shuju/s8_pd_test_S.txt");
    string line;
    
    getline(file, line);
    
    while (getline(file, line)) {
        stringstream ss(line);
        string item;
        vector<double> row;
        
        while (getline(ss, item, ',')) {
            row.push_back(stod(item));
        }
        
        if (row.size() >= 7) {
            data.x_starts.push_back(row[2]);
            data.x_ends.push_back(row[3]);
            data.y_starts.push_back(row[4]);
            data.y_ends.push_back(row[5]);
            data.power_densities.push_back(row[6]);
        }
    }
    
    file.close();
    return data;
}

// 导入顶面温度信息
MatrixXd load_boundary_data() {
    ifstream file("E:/hot/multi-layer/shuju/combined_fd3.txt");
    vector<vector<double>> temp_data;
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        double val;
        
        while (ss >> val) {
            row.push_back(val);
        }
        temp_data.push_back(row);
    }
    
    file.close();
    
    int rows = temp_data.size();
    int cols = temp_data[0].size();
    MatrixXd data_f(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data_f(i, j) = temp_data[i][j];
        }
    }
    
    return data_f;
}

// 导入COMSOL数据
CoMSOLData load_comsol_data() {
    CoMSOLData data;
    ifstream file("E:/hot/multi-layer/V2/sr2.7.txt");
    double x, y, z, temp;
    
    while (file >> x >> y >> z >> temp) {
        data.x.push_back(x);
        data.y.push_back(y);
        data.z.push_back(z);
        data.temp.push_back(temp);
    }
    
    file.close();
    cout << "Number of COMSOL data points: " << data.x.size() << endl;
    return data;
}

// 计算gmn
MatrixXd calculate_gmn(const HeatSourceData& heat_data) {
    MatrixXd gmn = MatrixXd::Zero(num_eigen, num_eigen);
    int num_sources = heat_data.power_densities.size();
    
    MatrixXd int_x_all(num_sources, num_eigen);
    MatrixXd int_y_all(num_sources, num_eigen);
    
    for (int idx = 0; idx < num_sources; idx++) {
        for (int p = 0; p < num_eigen; p++) {
            int_x_all(idx, p) = compute_integral_segment(
                heat_data.x_starts[idx], heat_data.x_ends[idx], a, p);
            int_y_all(idx, p) = compute_integral_segment(
                heat_data.y_starts[idx], heat_data.y_ends[idx], b, p);
        }
    }
    
    VectorXd norm_x(num_eigen), norm_y(num_eigen);
    for (int i = 0; i < num_eigen; i++) {
        norm_x(i) = (i == 0) ? 1.0/a : 2.0/a;
        norm_y(i) = (i == 0) ? 1.0/b : 2.0/b;
    }
    
    for (int idx = 0; idx < num_sources; idx++) {
        double q = heat_data.power_densities[idx];
        for (int m = 0; m < num_eigen; m++) {
            for (int n = 0; n < num_eigen; n++) {
                gmn(m, n) += q * int_x_all(idx, m) * norm_x(m) * 
                             int_y_all(idx, n) * norm_y(n);
            }
        }
    }
    
    return gmn;
}

// 计算fmn
MatrixXd calculate_fmn(const MatrixXd& data_f) {
    const int num_l = 400;
    VectorXd x(num_l), y(num_l);
    
    for (int i = 0; i < num_l; i++) {
        x(i) = (i + 0.5) * a / num_l;
        y(i) = (i + 0.5) * b / num_l;
    }
    
    MatrixXd cos_mx(num_eigen, num_l);
    MatrixXd cos_ny(num_eigen, num_l);
    
    for (int m = 0; m < num_eigen; m++) {
        for (int i = 0; i < num_l; i++) {
            cos_mx(m, i) = cos(m * PI / a * x(i));
            cos_ny(m, i) = cos(m * PI / b * y(i));
        }
    }
    
    MatrixXd fx(num_eigen, num_l);
    for (int m = 0; m < num_eigen; m++) {
        double norm = (m > 0) ? 2.0/a : 1.0/a;
        for (int j = 0; j < num_l; j++) {
            double sum = 0;
            for (int i = 0; i < num_l; i++) {
                sum += data_f(j, i) * cos_mx(m, i);
            }
            fx(m, j) = sum * (a / num_l) * norm;
        }
    }
    
    MatrixXd fmn = MatrixXd::Zero(num_eigen, num_eigen);
    for (int m = 0; m < num_eigen; m++) {
        for (int n = 0; n < num_eigen; n++) {
            double norm = (n > 0) ? 2.0/b : 1.0/b;
            double sum = 0;
            for (int i = 0; i < num_l; i++) {
                sum += fx(m, i) * cos_ny(n, i);
            }
            fmn(m, n) = sum * (b / num_l) * norm;
        }
    }
    
    return fmn;
}

// 计算lambda_mn
double calculate_lambda_mn(int m, int n, int layer_idx) {
    double kx_layer = kx[layer_idx];
    double ky_layer = ky[layer_idx];
    double kz_layer = kz[layer_idx];
    
    double lambda_squared = (kx_layer / kz_layer) * pow(m * PI / a, 2) +
                           (ky_layer / kz_layer) * pow(n * PI / b, 2);
    return sqrt(lambda_squared);
}

// 求解系数矩阵
VectorXd solve_system(int m, int n, const MatrixXd& gmn, const MatrixXd& fmn) {
    SparseMatrix<double> amatrix(n_l * 2, n_l * 2);
    VectorXd bvector = VectorXd::Zero(n_l * 2);
    
    vector<Triplet<double>> triplets;
    
    if (m == 0 && n == 0) {
        triplets.push_back(Triplet<double>(0, 1, 1));
        bvector(0) = 0;
        
        for (int i = 1; i < n_l; i++) {
            triplets.push_back(Triplet<double>(2*i-1, 2*i-1, kz[i-1]));
            triplets.push_back(Triplet<double>(2*i-1, 2*i+1, -kz[i]));
            
            triplets.push_back(Triplet<double>(2*i, 2*i-2, 1));
            triplets.push_back(Triplet<double>(2*i, 2*i-1, th[i-1]));
            triplets.push_back(Triplet<double>(2*i, 2*i, -1));
            triplets.push_back(Triplet<double>(2*i, 2*i+1, -th[i-1]));
        }
        
        triplets.push_back(Triplet<double>(2*n_l-1, 2*n_l-2, 1));
        triplets.push_back(Triplet<double>(2*n_l-1, 2*n_l-1, th[n_l-1]));
        bvector(2*n_l-1) = fmn(0, 0);
        
        bvector(2*(n_c-1)-1) = -gmn(0, 0) * th[n_c-2];
        bvector(2*(n_c-1)) = -0.5 * gmn(0, 0) / kz[n_c-1] * pow(th[n_c-2], 2);
        bvector(2*n_c-1) = gmn(0, 0) * th[n_c-1];
        bvector(2*n_c) = 0.5 * gmn(0, 0) / kz[n_c-1] * pow(th[n_c-1], 2);
        
    } else {
        triplets.push_back(Triplet<double>(0, 0, 1));
        triplets.push_back(Triplet<double>(0, 1, -1));
        bvector(0) = 0;
        
        for (int i = 1; i < n_l; i++) {
            double lamb_prev = calculate_lambda_mn(m, n, i-1);
            double lamb_curr = calculate_lambda_mn(m, n, i);
            
            triplets.push_back(Triplet<double>(2*i-1, 2*i-2, 
                -kz[i-1] * lamb_prev * exp(-lamb_prev * th[i-1])));
            triplets.push_back(Triplet<double>(2*i-1, 2*i-1, 
                kz[i-1] * lamb_prev * exp(lamb_prev * th[i-1])));
            triplets.push_back(Triplet<double>(2*i-1, 2*i, 
                kz[i] * lamb_curr * exp(-lamb_curr * th[i-1])));
            triplets.push_back(Triplet<double>(2*i-1, 2*i+1, 
                -kz[i] * lamb_curr * exp(lamb_curr * th[i-1])));
            
            triplets.push_back(Triplet<double>(2*i, 2*i-2, exp(-lamb_prev * th[i-1])));
            triplets.push_back(Triplet<double>(2*i, 2*i-1, exp(lamb_prev * th[i-1])));
            triplets.push_back(Triplet<double>(2*i, 2*i, -exp(-lamb_curr * th[i-1])));
            triplets.push_back(Triplet<double>(2*i, 2*i+1, -exp(lamb_curr * th[i-1])));
        }
        
        double lamb_top = calculate_lambda_mn(m, n, n_l-1);
        triplets.push_back(Triplet<double>(2*n_l-1, 2*n_l-2, exp(-lamb_top * th[n_l-1])));
        triplets.push_back(Triplet<double>(2*n_l-1, 2*n_l-1, exp(lamb_top * th[n_l-1])));
        bvector(2*n_l-1) = fmn(m, n);
        
        double lamb_source = calculate_lambda_mn(m, n, n_c-1);
        bvector(2*(n_c-1)) = gmn(m, n) / (kz[n_c-1] * pow(lamb_source, 2));
        bvector(2*n_c) = -gmn(m, n) / (kz[n_c-1] * pow(lamb_source, 2));
    }
    
    amatrix.setFromTriplets(triplets.begin(), triplets.end());
    
    SparseLU<SparseMatrix<double>> solver;
    solver.compute(amatrix);
    
    if (solver.info() != Success) {
        cerr << "Decomposition failed for m=" << m << ", n=" << n << endl;
        return VectorXd::Zero(n_l * 2);
    }
    
    VectorXd solution = solver.solve(bvector);
    return solution;
}

// 计算全场温度
VectorXd calculate_temperature(const vector<MatrixXd>& AB, const MatrixXd& gmn,
                               const CoMSOLData& comsol_data) {
    int num_points = comsol_data.x.size();
    VectorXd temp = VectorXd::Zero(num_points);
    
    for (int pt = 0; pt < num_points; pt++) {
        double x = comsol_data.x[pt];
        double y = comsol_data.y[pt];
        double z = comsol_data.z[pt];
        
        int layer_idx = 0;
        for (int i = 0; i < n_l; i++) {
            if (z >= layer_starts[i] && (i == n_l-1 || z < layer_starts[i+1])) {
                layer_idx = i;
                break;
            }
        }
        
        double zeta = z;
        
        double A_0 = AB[0](0, 2 * layer_idx);
        double B_0 = AB[0](0, 2 * layer_idx + 1);
        temp(pt) = A_0 + B_0 * zeta;
        
        if (layer_idx == n_c - 1) {
            temp(pt) -= 0.5 * gmn(0, 0) / kz[layer_idx] * pow(zeta, 2);
        }
        
        for (int m = 0; m < num_eigen; m++) {
            for (int n = 0; n < num_eigen; n++) {
                if (m == 0 && n == 0) continue;
                
                int idx = m * num_eigen + n;
                double lamb = calculate_lambda_mn(m, n, layer_idx);
                double A = AB[idx](0, 2 * layer_idx);
                double B = AB[idx](0, 2 * layer_idx + 1);
                
                double exp_neg = exp(-lamb * zeta);
                double exp_pos = exp(lamb * zeta);
                double cos_mx = cos(m * PI / a * x);
                double cos_ny = cos(n * PI / b * y);
                
                temp(pt) += (A * exp_neg + B * exp_pos) * cos_mx * cos_ny;
                
                if (layer_idx == n_c - 1) {
                    temp(pt) += gmn(m, n) / (kz[layer_idx] * pow(lamb, 2)) * cos_mx * cos_ny;
                }
            }
        }
    }
    
    return temp;
}

// 输出结果
void export_results(const CoMSOLData& comsol_data, const VectorXd& calculated_temp,
                   const VectorXd& error) {
    ofstream outfile("temperature_results.csv");
    outfile << "x,y,z,temp_calculated,temp_comsol,error\n";
    
    for (size_t i = 0; i < comsol_data.x.size(); i++) {
        outfile << comsol_data.x[i] << "," << comsol_data.y[i] << "," 
                << comsol_data.z[i] << "," << calculated_temp(i) << "," 
                << comsol_data.temp[i] << "," << error(i) << "\n";
    }
    
    outfile.close();
    cout << "Results exported to temperature_results.csv" << endl;
}

int main() {
    auto start_time = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < n_l; i++) {
        kx[i] = ka_anisotropic[i][0];
        ky[i] = ka_anisotropic[i][1];
        kz[i] = ka_anisotropic[i][2];
    }
    
    cout << "Loading data..." << endl;
    HeatSourceData heat_data = load_thermal_data();
    MatrixXd data_f = load_boundary_data();
    CoMSOLData comsol_data = load_comsol_data();
    
    cout << "Calculating gmn and fmn..." << endl;
    MatrixXd gmn = calculate_gmn(heat_data);
    MatrixXd fmn = calculate_fmn(data_f);
    
    cout << "Solving all modes..." << endl;
    vector<MatrixXd> AB(num_eigen * num_eigen);
    
    for (int m = 0; m < num_eigen; m++) {
        for (int n = 0; n < num_eigen; n++) {
            int idx = m * num_eigen + n;
            VectorXd solution = solve_system(m, n, gmn, fmn);
            AB[idx] = MatrixXd(1, n_l * 2);
            AB[idx].row(0) = solution.transpose();
            
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    double seconds = chrono::duration_cast<chrono::duration<double>>(end_time - start_time).count();
    cout << fixed << setprecision(2);
    cout << "Calculation time: " << seconds << " seconds" << endl;
    // 计算误差
    cout << "Calculating error compared to COMSOL..." << endl;
    VectorXd temp = calculate_temperature(AB, gmn, comsol_data);
    
    VectorXd error(comsol_data.temp.size());
    for (size_t i = 0; i < comsol_data.temp.size(); i++) {
        error(i) = abs(temp(i) - comsol_data.temp[i]);
    }
    
    double mean_error = error.mean();
    double max_error = error.maxCoeff();
    
    cout << fixed << setprecision(6);
    cout << "Mean absolute error: " << mean_error << " K" << endl;
    cout << "Maximum absolute error: " << max_error << " K" << endl;
    
    return 0;
}

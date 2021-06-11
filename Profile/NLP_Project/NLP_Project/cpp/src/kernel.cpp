#include "kernel.hpp"

PolyKernel::PolyKernel(int _c, int _deg) {
    this->_c = _c;
    this->_deg = _deg;
}
double PolyKernel::val(const VectorXd &x, const VectorXd &y) {
    return pow(x.dot(y) + _c,_deg);
}
GaussianKernel::GaussianKernel(double _bandwidth) {
    this->_bandwidth = _bandwidth;
    this->_bandwidth_2 = 1.0/ pow(_bandwidth,2);
}

double GaussianKernel::val(const VectorXd &x, const VectorXd &y) {
    return exp(-0.5*_bandwidth_2 * (x-y).dot(x-y));
}
#include <string>

#ifndef MYLIB_H
#define MYLIB_H
#include <cmath>
#include <math.h>
#include <Eigen/Dense>
#include "kernel.hpp"
#include <set>
#include "ConfusionMatrix.hpp"

using namespace Eigen;
class svm {

public:
    svm(double c, std::string kernel_name);
    ~svm();

    void fit(const MatrixXd& x,const VectorXd& y);
    VectorXd predict(const MatrixXd& x);
    double score(const MatrixXd& x,const VectorXd& y);
    double evaluate(const MatrixXd& x,const VectorXd& y,ConfusionMatrix& matrix);


private:
    // f(x) = sign(sum(_ai * yi * K(x,xi) )+ _b)
    VectorXd _a;
    double _b ;
    int max_iteration = 100000;
    // _n: number of trained data, _d dim
    int _n;
    int _d;
    double _c;

    double _epsilon = 0.00000001;
    // tolerrate value on alpha change
    double _toler = 0.00000001;

    MatrixXd _train_x;
    VectorXd _train_y;
    MatrixXd _kernal_val;
    Kernel* _kernal;

    // member function
    void _init(const MatrixXd& x,const VectorXd& y);
    // SMO sovler
    VectorXd _E;
    std::set<int> _zero_a,_sup_vec;
    int _find_first();
    int _find_second(int i);
    double _clip_a(int i,int j,double a_j_new);
    void _update_b_E(int i,int j,double ainew,double ajnew);
    void _update_a(int i,double val);
    void _smo();
    bool _check_stop();
    void _simplified_smo();
    double _helper_smo(int i);
    int _find_random_j(int i);

};



#endif

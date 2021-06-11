#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include "ConfusionMatrix.hpp"
using namespace Eigen;
using namespace std;




class LogisticRegression{
public:
    LogisticRegression(int max_iter, double mu, double alpha);
    ~LogisticRegression()=default;

    void fit(const MatrixXd& x,const VectorXd& y);
    VectorXd predict(const MatrixXd& x);
    double score(const MatrixXd& x,const VectorXd& y);
    double evaluate(const MatrixXd& x,const VectorXd& y,ConfusionMatrix& matrix);

private:
    MatrixXd _train_x;
    VectorXd _train_y;

    MatrixXd _beta;
    MatrixXd _B;
    MatrixXd _train_x_bar;

    const double eps = 0.0001;   // 用于浮点数比较大小

    int _max_iter;
    double _mu; // step of learning
    double _alpha; // L2 regularization
    int _sampleNum;
    int _attrNum;

    void _init(const MatrixXd& x,const VectorXd& y);
    MatrixXd _theta_gradient();

    double _helper(const VectorXd& x,const VectorXd& y);

};

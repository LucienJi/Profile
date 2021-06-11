#include<Eigen/Dense>
#include<cmath>
using namespace Eigen;

class Kernel {
public:
    Kernel()= default;;
    ~Kernel()= default;;
    virtual double val(const VectorXd& x,const VectorXd& y)=0;

};

class LinearKernel:public Kernel{
public:
    LinearKernel()= default;;
    double val(const VectorXd& x,const VectorXd& y) override{
        return x.dot(y);
    }
};

class PolyKernel:public Kernel{
public:
    PolyKernel(int _c,int _deg);
    ~PolyKernel()= default;;
    double val(const VectorXd& x,const VectorXd& y) override;

private:
    int _c;
    int _deg;

};

class GaussianKernel:public Kernel{
public:

    GaussianKernel(double _bandwidth);
    ~GaussianKernel()= default;;
    double val(const VectorXd& x,const VectorXd& y) override;

private:
    double _bandwidth;
    double _bandwidth_2;

};
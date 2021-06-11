#include <iostream>
#include <iostream>
#include <mylib>

using namespace Eigen;
using namespace std;

void generate_data(MatrixXd& X, VectorXd& Y, int n){
    double k = 0.6;
    double b = -15.7;

    double x,y;
    for(int i = 0;i<n;i++){
        x = rand()*10.0;
        y = rand()*10.0;
        X(i,0) = x;
        X(i,1) = y;
        Y(i) = (x*k + b)>y ? 1 : 0;
    }

}

int main() {

    int n_train = 10000;
    int n_test = 100;
    MatrixXd train_x(n_train,2);
    VectorXd train_y(n_train);
    generate_data(train_x,train_y,n_train);

    MatrixXd test_x(n_test,2);
    VectorXd test_y(n_test);
    generate_data(test_x,test_y,n_test);

    //svm s = svm(1.0,"Poly");
    //GaussianKernel kernel(0.5);
    LogisticRegression lr(100,0.001,0.0);
    lr.fit(train_x,train_y);
    double e_rate = lr.score(test_x,test_y);
    std::cout<< e_rate << std::endl;


    return 0;
}
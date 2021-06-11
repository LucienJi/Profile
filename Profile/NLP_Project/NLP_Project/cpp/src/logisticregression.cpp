#include "../include/mylib_bits/logisticregression.hpp"



void LogisticRegression::_init(const MatrixXd& x,const VectorXd& y){
    _train_x = x;
    _train_y = y;

    _sampleNum = x.rows();
    _attrNum = x.cols();

    _B = MatrixXd::Ones(_sampleNum,1);
    _beta = VectorXd::Zero(_attrNum+1);
    _train_x_bar = MatrixXd::Ones(x.rows(),x.cols() + _B.cols());
    for(int i = 0;i<_sampleNum;i++){
        for(int j =0;j<_attrNum;j++){
            _train_x_bar(i,j) = x(i,j);
        }
    }

}

void LogisticRegression::fit(const MatrixXd& x,const VectorXd& y){
    _init(x,y);
    for(int i = 0;i<_max_iter;i++){
        _beta -= _mu*_theta_gradient();
    }


}

MatrixXd LogisticRegression::_theta_gradient(){
    MatrixXd ans = _alpha * _beta;
    // ans.fill(0);
    for(int i=0;i<_sampleNum;i++){
        // cout << X_bar.row(i).rows() << "\t" << X_bar.row(i).cols() <<endl;
        double tmp = exp((_train_x_bar.row(i)*_beta).sum());
        // cout << X_bar.row(i).transpose() << endl;
        ans +=_train_x_bar.row(i).transpose()*(_train_y(i)-tmp/(1+tmp));
    }
    return -ans;
}

VectorXd LogisticRegression::predict(const MatrixXd& x){
    int n = x.rows();
    int m = x.cols();
    
    MatrixXd x_bar = MatrixXd::Ones(n,m + 1);
    _B = MatrixXd::Ones(_sampleNum,1);
    for(int i = 0;i<n;i++){
        for(int j = 0;j<m;j++){
            x_bar(i,j) = x(i,j);
        }
    }
    VectorXd y_pred = VectorXd::Zero(n);
    for(int i = 0 ;i<n;i++){
        double tmp = exp((x_bar.row(i)*_beta).sum());
        tmp = 1/(tmp + 1);
        // cout << tmp << endl;
        if(tmp <0.5)
           y_pred(i)= 1; // 否则默认是0
    }
    return y_pred;

}

double LogisticRegression::score(const MatrixXd& x,const VectorXd& y){
    VectorXd y_preds = predict(x);
    assert(y_preds.rows() == y.rows());
    double error = 0.0;
    for(int i = 0 ;i<y_preds.rows();i++){
        error += std::abs(y_preds(i) - y(i));
    }

    return error/y.rows();
}

double LogisticRegression::evaluate(const MatrixXd& x,const VectorXd& y,ConfusionMatrix& matrix){
    VectorXd y_preds = predict(x);
    assert(y_preds.rows() == y.rows());
    double error = 0.0;
    for(int i = 0 ;i<y_preds.rows();i++){
        matrix.AddPrediction(int(y(i)),int(y_preds(i)));
        error += std::abs(y_preds(i) - y(i));
    }

    return error/y.rows();
}

LogisticRegression::LogisticRegression(int max_iter, double mu, double alpha) {
    this->_max_iter = max_iter;
    this->_mu = mu;
    this->_alpha = alpha;

}

double LogisticRegression::_helper(const VectorXd &x, const VectorXd &y) {
    return exp(x.dot(y));
}

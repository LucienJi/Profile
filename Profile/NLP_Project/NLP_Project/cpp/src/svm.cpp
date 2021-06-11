#include "../include/mylib_bits/svm.hpp"

#include <iostream>
#include <utility>

svm::svm(double c, std::string kernel_name){
    this->_c = c;
    if(kernel_name == "LinearKernel"){
        this->_kernal = new LinearKernel();

    }else if (kernel_name == "Poly"){
        this->_kernal = new PolyKernel(1,2);
    }else if(kernel_name == "Gaussian"){
        this->_kernal = new GaussianKernel(1);
    }else{
        this->_kernal = new LinearKernel();
    }
    
}




svm::~svm(){
}

void svm::_init(const MatrixXd& x, const VectorXd& y) {
    this->_n = x.rows();
    this->_d = x.cols();
    this->_train_x = x;
    this->_train_y = y;

    this->_a = VectorXd::Zero(x.rows());
    this->_b = 0.0;
    // kernel values precalculated, symmetric positive
    this->_kernal_val = MatrixXd(this->_n,this->_n);
    //std::cout<< this->_kernal->val(_kernal_val.row(1),_kernal_val.row(0)) << "  " << x.col(0)<<std::endl;

    for(int i = 0;i<this->_n;i++){
        for(int j = i;j<this->_n;j++){
            this->_kernal_val(i,j) = this->_kernal->val(x.row(i),x.row(j));

            this->_kernal_val(j,i) = this->_kernal_val(i,j);
        }
    }

    // smo
    // E
    this->_E = VectorXd(this->_n);
    for(int i = 0;i<this->_n;i++){
        this->_E(i) = -1.0*this->_train_y(i);
    }
    // at initial time, a equals all zero, no support vect
    for(int i = 0;i<this->_n;i++){
        this->_zero_a.insert(i);
    }


}
double svm::_helper_smo(int i) {
    double res;
    for(int j = 0;j<_a.size();j++){
        res+= _a(j)*_train_y(j)*_kernal_val(j,i);
    }
    res += this->_b;
    return res - _train_y(i);
}
int svm::_find_random_j(int i){
    std::srand(std::time(nullptr)); // use current time as seed for random generator
    int j = std::rand()%_a.size();
    while( j == i){
        j = std::rand()%_a.size();
    }
    return j;
}
void svm::_simplified_smo() {
    //std::cout<<"Fitting"<<std::endl;
    int max_ite = 10000;
    int ct = 0;
    while (ct < max_ite){
        int num_changed_alpha = 0;
        for(int i = 0;i<this->_a.size();i++){
            double E_i = this->_helper_smo(i);
            if((_train_y(i)*E_i < _toler && _a(i) < _c)|| (_train_y(i)*E_i > _toler && _a(i) > 0) ){

                int j = _find_second(i);
                if(j == -1){
                j = (i+1)%_a.size();
                }
                //int j = this->_find_random_j(i);

                double E_j = this->_helper_smo(j);
                double a_i_old = _a(i);
                double a_j_old = _a(j);
                double L,H;
                if(_train_y(i) == _train_y(j)){
                    L = std::max(0.0,a_i_old + a_j_old - _c);
                    H = std::min(_c,a_i_old + a_j_old);
                }else{
                    L = std::max(0.0,a_j_old - a_i_old);
                    H = std::min(_c,_c + a_j_old - a_i_old);
                }

                if(L == H){
                    continue;
                }
                double eta = 2*_kernal_val(i,j) - _kernal_val(i,i) - _kernal_val(j,j);
                if(eta >= 0){
                    continue;
                }
                double a_j = a_j_old - _train_y(j)*(E_i - E_j)/eta;
                a_j = std::max(a_j,L);
                a_j = std::min(a_j,H);
                if(fabs(a_j - a_j_old)<_toler){
                    continue;
                }
                double a_i = a_i_old + _train_y(i)*_train_y(j)*(a_j_old - a_j);
                double bi = _b - E_i - _train_y(i)*(a_i - a_i_old)*_kernal_val(i,i) - _train_y(j)*(a_j - a_j_old)*_kernal_val(i,j);
                double bj = _b - E_j - _train_y(i)*(a_i - a_i_old)*_kernal_val(i,j) - _train_y(j)*(a_j - a_j_old)*_kernal_val(j,j);
                if(a_i>0 && a_i < _c){
                    _b = bi;
                }else if(a_j>0 && a_j < _c){
                    _b = bj;
                }else{
                    _b = (bi + bj)/2;
                }
                _a(i) = a_i;
                _a(j) = a_j;
                num_changed_alpha++;
            }

        }

        if(num_changed_alpha == 0){
            ct++;
        }else{
            ct = 0;
        }
    }
}


int svm::_find_first() {
    // find the index of a which violate the kkt condition most
    // satisfy the KKT condition one of them:
    // ai =0 && yi * g(xi) >= 1, normal
    // 0<ai < C && yi*g(xi) = 1 , support vec
    // ai = C && yi*g(xi) <=1, between the two boundary
    // g(xi) = _E(i) + yi

    // violate:
    double gi;
    double ai;
    double yi;
    /*
    for(int i = 0 ;i<this->_n;i++){
        gi = this->_E(i) + this->_train_y(i);
        ai = this->_a(i);
        yi = this->_train_y(i);
        if(((yi*gi <= 1.0)&&ai<this->_c)|| (yi*gi >=1 && ai > 0) || (abs(yi*gi - 1)<1e-5 && ((abs(ai - this->_c)<1e-5)||(abs(ai)<1e-5)))){
            return i;
        }
    }
     */
    for(auto it = _sup_vec.begin();it!=_sup_vec.end();it++){
        // support a, a!=0
        int i = *it;
        gi = this->_E(i) + this->_train_y(i);
        ai = this->_a(i);
        yi = this->_train_y(i);
        if((ai >= _c - _epsilon && yi*gi > 1) || (ai < _c -_epsilon && abs(yi*gi - 1)>_epsilon)){
            return i;
        }

    }

    for(auto it = _zero_a.begin();it!=_zero_a.end();it++){
        int i = *it;
        gi = this->_E(i) + this->_train_y(i);
        ai = this->_a(i);
        yi = this->_train_y(i);
        if(yi*gi <= 1){
            return i;
        }
    }

    return -1;
}

int svm::_find_second(int i) {
    double Ei = this->_E(i);
    double max = 0.0;
    int res = -1;
    for(int j =0;j<this->_n;j++){
        if(abs(this->_E(j) - Ei) > max){
            max = abs(this->_E(j) - Ei);
            res = j;
        }
    }
    return res;

}

double svm::_clip_a(int i, int j, double ajnew) {
    double L = 0.0,H = 0.0;
    if(abs(_train_y(i)-_train_y(j))>_epsilon){
        // yi != yj
        L = std::max(0.0,_a(j)-_a(i));
        H = std::min(_c,_c+_a(j)-_a(i));
    }else{
        L = std::max(0.0,_a(j)+_a(i) - _c);
        H = std::min(_c,_a(i) + _a(j));
    }
    ajnew = std::min(H,ajnew);
    ajnew = std::max(L,ajnew);
    return ajnew;
}

void svm::_update_b_E(int i, int j, double ainew, double ajnew) {
    _update_a(i,ainew);
    _update_a(j,ajnew);

    double bi = -1 * _E(i) - _train_y(i)*_kernal_val(i,i)*(ainew - _a(i)) - _train_y(j)*_kernal_val(j,i)*(ajnew - _a(j)) + _b;
    double bj = -1 * _E(j) - _train_y(i)*_kernal_val(i,j)*(ainew - _a(i)) - _train_y(j)*_kernal_val(j,j)*(ajnew - _a(j)) + _b;
    double newb = bi + (bj - bi)/2;
    //_E(i) = _E(i) + (ainew - _a(i))*_train_y(i)*_kernal_val(i,i) + (ajnew - _a(j))*_train_y(j)*_kernal_val(i,j) + newb - _b;
    //_E(j) = _E(j) + (ainew - _a(i))*_train_y(j)*_kernal_val(j,i) + (ajnew - _a(j))*_train_y(j)*_kernal_val(j,j) + newb - _b;
    // update _E(i) with support
    double ei = 0.0;
    double ej = 0.0;
    for(auto it = _sup_vec.begin();it!=_sup_vec.end();it++){
        ei+=_train_y(*it)*_a(*it)*_kernal_val(i,*it);
        ej+=_train_y(*it)*_a(*it)*_kernal_val(j,*it);
    }
    _E(i) = ei + newb - _train_y(i);
    _E(j) = ej + newb - _train_y(j);
    _b = newb;
}

void svm::_update_a(int i, double val) {
    if(_zero_a.find(i) != _zero_a.end() && val > 0 + _toler){
        _zero_a.erase(i);
        _sup_vec.insert(i);
    }else if(_sup_vec.find(i) != _sup_vec.end() && val <= 0+_toler){
        _zero_a.insert(i);
        _sup_vec.erase(i);
    }
}

void svm::_smo(){
    int ct = 0;
    while(true){
        int i = _find_first();
        if(i<0){
            break;
        }
        int j = _find_second(i);
        if(j<0){
            break;
        }
        double eta = _kernal_val(i,i) + _kernal_val(j,j) - 2*_kernal_val(i,j);
        if(eta == 0){
            continue;
        }
        double ajnew = _a(j) + _train_y(j)*(_E(i)-_E(j))/eta;
        ajnew = _clip_a(i,j,ajnew);
        double ainew = _a(i) + _train_y(i)*_train_y(j)*(ajnew - _a(j));

        double diff  = abs(ainew - _a(i)) + abs(ajnew - _a(j));
        _update_b_E(i,j,ainew,ajnew);
        _a(i) = ainew;
        _a(j) = ajnew;

        if(_check_stop()){
            break;
        }
        if(ct>max_iteration){
            break;
        }else{
            ct++;
        }
    }
}

bool svm::_check_stop() {
    double sum;
    for(auto it=_sup_vec.begin();it!=_sup_vec.end();it++){
        sum+=_train_y(*it);
    }
    if(abs(sum) > _epsilon){
        return false;
    }
    return true;
}

void svm::fit(const MatrixXd& x, const VectorXd& y) {
    _init(x,y);
    _simplified_smo();
}

VectorXd svm::predict(const MatrixXd& x) {
    assert(x.cols() == this->_d);
    int nums = x.rows();
    VectorXd y_preds(nums);
    for(int i = 0;i<nums;i++){
        double y = 0.0;
        for(int j = 0;j< this->_n;j++){
            y+=_a(j) * _train_y(j) * _kernal->val(x.row(i),_train_x.row(j));
        }
        y_preds(i) = (y+_b)>0 ? 1 : 0;
    }
    return y_preds;
}

double svm::score(const MatrixXd &x, const VectorXd &y) {
    VectorXd y_preds = predict(x);
    assert(y_preds.rows() == y.rows());
    double error = 0.0;
    for(int i = 0 ;i<y_preds.rows();i++){
        error += std::abs(y_preds(i) - y(i));
    }

    return error/y.rows();

}

double svm::evaluate(const MatrixXd &x, const VectorXd &y, ConfusionMatrix &matrix) {
    VectorXd y_preds = predict(x);
    assert(y_preds.rows() == y.rows());
    double error = 0.0;
    for(int i = 0 ;i<y_preds.rows();i++){
        matrix.AddPrediction(int(y(i)),int(y_preds(i)));
        error += std::abs(y_preds(i) - y(i));
    }

    return error/y.rows();

}
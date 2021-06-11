//
// Created by 季经天 on 2021/5/9.
//

#include "../include/mylib_bits/ConfusionMatrix.hpp"
#include <iostream>

using namespace std;

ConfusionMatrix::ConfusionMatrix() {
    // TODO Exercise 2.1
    // Populate 2x2 matrix with 0s
    m_confusion_matrix = new int*[2];
    for(int i = 0 ; i<2;i++){
        m_confusion_matrix[i] = new int [2]{0,0};
    }
}

ConfusionMatrix::~ConfusionMatrix() {
    // Attribute m_confusion_matrix is deleted automatically
    for(int i = 0;i<2;i++){
        delete m_confusion_matrix[i];
    }
    delete[] m_confusion_matrix;
}

void ConfusionMatrix::AddPrediction(int true_label, int predicted_label) {
    // TODO Exercise 2.1
    m_confusion_matrix[true_label][predicted_label]++;
}

void ConfusionMatrix::PrintEvaluation() const{
    // Prints the confusion matrix
    cout <<"\t\tPredicted\n";
    cout <<"\t\t0\t1\n";
    cout <<"Actual\t0\t"
         <<GetTN() <<"\t"
         <<GetFP() <<endl;
    cout <<"\t1\t"
         <<GetFN() <<"\t"
         <<GetTP() <<endl <<endl;
    // Prints the estimators
    cout <<"Error rate\t\t"
         <<error_rate() <<endl;
    cout <<"False alarm rate\t"
         <<false_alarm_rate() <<endl;
    cout <<"Detection rate\t\t"
         <<detection_rate() <<endl;
    cout <<"F-score\t\t\t"
         <<f_score() <<endl;
    cout <<"Precision\t\t"
         <<precision() <<endl;
}

int ConfusionMatrix::GetTP() const {
    return m_confusion_matrix[1][1];
}

int ConfusionMatrix::GetTN() const {
    return m_confusion_matrix[0][0];
}

int ConfusionMatrix::GetFP() const {
    return m_confusion_matrix[0][1];
}

int ConfusionMatrix::GetFN() const {
    return m_confusion_matrix[1][0];
}

double ConfusionMatrix::f_score() const {
    // 2*Precision*Recall/(Precision + Reca)
    return 2*precision()*detection_rate()/(precision() + detection_rate());
}

double ConfusionMatrix::precision() const {
    // TP/(TP+FP)
    return double(m_confusion_matrix[1][1])/(double(m_confusion_matrix[0][1]) + double (m_confusion_matrix[1][1]));
}

double ConfusionMatrix::error_rate() const {
    return double (m_confusion_matrix[0][1]+ m_confusion_matrix[1][0])/(double(m_confusion_matrix[1][0]) + double (m_confusion_matrix[1][1]) + double(m_confusion_matrix[0][1]) + double(m_confusion_matrix[0][0]));

}

double ConfusionMatrix::detection_rate() const {
    // TP/(TP+FN)
    return double(m_confusion_matrix[1][1])/(double(m_confusion_matrix[1][0]) + double (m_confusion_matrix[1][1]));

}

double ConfusionMatrix::false_alarm_rate() const {
    // FP/(FP+TN)
    return double(m_confusion_matrix[0][1])/(double(m_confusion_matrix[0][1]) + double (m_confusion_matrix[0][0]));
}
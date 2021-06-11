//
// Created by 季经天 on 2021/5/9.
//

#include "../cpp/include/mylib_bits/ConfusionMatrix.hpp"

#include <pybind11/stl.h>
#include<pybind11/eigen.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_ConfusionMatrix(py::module &m){
    py::class_<ConfusionMatrix>(m,"ConfusionMatrix")
            .def(py::init<>())
            .def("AddPrediction",&ConfusionMatrix::AddPrediction)
            .def("GetTP",&ConfusionMatrix::GetTP)
            .def("GetTN",&ConfusionMatrix::GetTN)
            .def("GetFP",&ConfusionMatrix::GetFP)
            .def("GetFN",&ConfusionMatrix::GetFN)
            .def("f_score",&ConfusionMatrix::f_score)
            .def("precision",&ConfusionMatrix::precision)
            .def("error_rate",&ConfusionMatrix::error_rate)
            .def("detection_rate",&ConfusionMatrix::detection_rate)
            .def("false_alarm_rate",&ConfusionMatrix::false_alarm_rate)
            .def("PrintEvaluation",&ConfusionMatrix::PrintEvaluation);
}
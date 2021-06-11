#include "../cpp/include/mylib_bits/logisticregression.hpp"

#include <pybind11/stl.h>
#include<pybind11/eigen.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_LR(py::module &m){
    py::class_<LogisticRegression>(m,"LogisticRegression")
    .def(py::init<int,double,double>())
    .def("fit",&LogisticRegression::fit)
    .def("predict",&LogisticRegression::predict)
    .def("score",&LogisticRegression::score)
    .def("evaluate",&LogisticRegression::evaluate);
}
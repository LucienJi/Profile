#include "../cpp/include/mylib_bits/svm.hpp"

#include <pybind11/stl.h>
#include<pybind11/eigen.h>

#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_svm(py::module &m) {
    
    py::class_<svm>(m, "svm")
    .def(py::init<double,std::string>())
    .def("fit",&svm::fit)
    .def("predict",&svm::predict)
    .def("score",&svm::score)
    .def("evaluate",&svm::evaluate);

    py::class_<LinearKernel>(m,"LinearKernel")
    .def(py::init<>())
    .def("val",&LinearKernel::val);
    
    
}

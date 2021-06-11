#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_svm(py::module &);
void init_LR(py::module &);
void init_ConfusionMatrix(py::module &);
namespace mcl {

PYBIND11_MODULE(mylib, m) {
    // Optional docstring
    m.doc() = "My library";
    
    init_svm(m);
    init_LR(m);
    init_ConfusionMatrix(m);
}
}

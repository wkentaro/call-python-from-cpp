#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};

    py::object scipy_misc = py::module::import("scipy.misc");
    py::object plt = py::module::import("matplotlib.pyplot");

    py::array_t<uint8_t> img = scipy_misc.attr("face")()
      .cast<py::array_t<uint8_t> >();
    std::cout << "shape: " << img.shape()[0] << " " << img.shape()[1] << " " << img.shape()[2] << std::endl;
    std::cout << "ndim: " << img.ndim() << std::endl;

    py::object img_py = py::cast<py::object>(img);
    std::cout << "==> Saved to: image.jpg" << std::endl;
    plt.attr("imsave")("image.jpg", img_py);
}

#include <iostream>
#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;

int main(int argc, char** argv)
{
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  py::object sys = py::module::import("sys");
  std::string sys_version = sys.attr("version").cast<std::string>();
  std::cout << "sys version: " << sys_version << std::endl;

  py::object scipy = py::module::import("scipy");
  std::string scipy_version = scipy.attr("__version__").cast<std::string>();
  std::cout << "scipy version: " << scipy_version << std::endl;

  return 0;
}

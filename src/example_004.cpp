#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
namespace py = pybind11;
using namespace py::literals;

int main() {
    py::scoped_interpreter guard{};

    auto locals = py::dict();
    py::exec(R"(
        import chainer
        from chainercv.datasets import coco_instance_segmentation_label_names
        from chainercv.links import MaskRCNNFPNResNet50
        from chainercv import utils
        import imgviz
        import numpy as np

        gpu = 0
        image_file = '../data/33823288584_1d21cf0a26_k.jpg'

        chainer.config.cv_resize_backend = 'cv2'

        label_names = coco_instance_segmentation_label_names
        model = MaskRCNNFPNResNet50(
            n_fg_class=len(label_names),
            pretrained_model='coco',
        )

        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

        img = imgviz.io.imread(image_file)
        img_input = img.transpose(2, 0, 1).astype(np.float32)

        masks, labels, scores = model.predict([img_input])
        masks = masks[0]
        labels = labels[0]
        scores = scores[0]
        class_ids = labels + 1
        captions = []
        for c, s in zip(class_ids, scores):
            captions.append('{:s}: {:.2%}'.format(label_names[c - 1], s))
        viz = imgviz.instances2rgb(
            image=img,
            labels=class_ids,
            masks=masks,
            captions=captions,
            font_size=15,
        )
    )", py::globals(), locals);

    std::string image_file = locals["image_file"].cast<std::string>();
    py::array_t<uint8_t> viz = locals["viz"].cast<py::array_t<uint8_t> >();
    std::cout << "ndim: " << viz.ndim() << std::endl;
    std::cout << "shape: (";
    for (size_t i = 0; i < viz.ndim(); i++) {
      std::cout << viz.shape()[i] << ", ";
    }
    std::cout << ")" << std::endl;

    // pybind11::array_t -> cv::Mat
    // TODO(unknown): more efficient implementation
    unsigned int rows = viz.shape(0);
    unsigned int cols = viz.shape(1);
    cv::Mat viz_mat = cv::Mat::zeros(/*rows=*/rows, /*cols=*/cols, CV_8UC3);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        // RGB -> BGR
        viz_mat.at<cv::Vec3b>(/*row=*/i, /*col=*/j) = cv::Vec3b(
          *viz.data(i, j, 2), *viz.data(i, j, 1), *viz.data(i, j, 0)
        );
      }
    }

    cv::imshow(image_file, viz_mat);
    cv::waitKey(0);
}

#include "eskf.hpp"
#include <ceres/rotation.h>

struct Process {
    template<typename T>
    bool operator()(const T * x1, T * x2) const {
        T gt[3] = {T(gyr[0] * dt), T(gyr[1] * dt), T(gyr[2] * dt)};
        Eigen::Matrix<T, 3, 3> expG;
        ceres::AngleAxisToRotationMatrix(gt, expG.data());
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> R(x1);
        Eigen::Map<Eigen::Matrix<T, 3, 3>> Re(x2);
        Re = expG.transpose() * R;
        return true;
    }

    Eigen::Vector3d gyr;
    double dt;
};

struct Measure {
    template<typename T>
    bool operator()(const T *x, T * y) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> R(x);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> me(y);
        me = R * gravity.cast<T>();
        return true;
    }

    inline static const Eigen::Vector3d gravity{0, 0, 1};
};

struct SO3Manifold : eskf::ManifoldBase<SO3Manifold, 9, 3> {

    template<typename T>
    void Plus(const T *x, const T *delta, T *x_plus_delta) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> map_x(x);
        Eigen::Map<Eigen::Matrix<T, 3, 3>> map_x_plus_delta(x_plus_delta);
        Eigen::Matrix<T, 3, 3> mat_delta; 
        ceres::AngleAxisToRotationMatrix(delta, mat_delta.data());
        map_x_plus_delta = (mat_delta * map_x).eval();
    }

    template<typename T>
    void Minus(const T *y, const T *x, T *y_minus_x) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> map_y(y);
        Eigen::Map<const Eigen::Matrix<T, 3, 3>> map_x(x);
        Eigen::Matrix<T, 3, 3> mat_y_minus_x = map_y * map_x.transpose();
        ceres::RotationMatrixToAngleAxis(mat_y_minus_x.data(), y_minus_x);
    }    
};

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


PYBIND11_MODULE(ahrs, m) {
    namespace py = pybind11;

    using ManifoldType = eskf::ProductManifold<SO3Manifold>;

    using ahrs = eskf::ESKF<ManifoldType>;

    py::class_<ahrs>(m, "ahrs")
            .def(py::init([](){ return ahrs{ManifoldType{SO3Manifold{}}}; }))
            .def("process", [](ahrs *obj, Eigen::Vector3d gyr, double dt) {
                                obj->process(Process{gyr, dt});
                            }, py::arg("gyr"), py::arg("dt"))
            .def("update",  [](ahrs *obj, Eigen::Vector3d acc, Eigen::Matrix3d V) {
                               obj->update(Measure{}, acc, V);
                            }, py::arg("acc"), py::arg("V"))
            .def_property("x", [](ahrs *c)->decltype(ahrs::x)& { return c->x; }, 
                               [](ahrs *c, const decltype(ahrs::x) &x) { c->x = x; }, 
                               py::return_value_policy::reference_internal)
            .def_property("P", [](ahrs *c)->decltype(ahrs::P)& { return c->P; }, 
                               [](ahrs *c, const decltype(ahrs::P) &P) { c->P = P; }, 
                               py::return_value_policy::reference_internal)
            .def_property("Q", [](ahrs *c)->decltype(ahrs::Q)& { return c->Q; }, 
                               [](ahrs *c, const decltype(ahrs::Q) &Q) { c->Q = Q; }, 
                               py::return_value_policy::reference_internal)
            ;
}

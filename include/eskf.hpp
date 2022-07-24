#ifndef ESKF_HPP
#define ESKF_HPP

#include <ceres/ceres.h>

namespace eskf {

    namespace internal {
        // 内部函数，用于实现自动求导
        template<typename F, typename MX, typename MY, typename MJ>
        void compute_with_jacobian(F &&func, const Eigen::MatrixBase<MX> &X, Eigen::MatrixBase<MY> &Y, Eigen::MatrixBase<MJ> &J) {
            constexpr int N = MX::RowsAtCompileTime;
            constexpr int M = MY::RowsAtCompileTime;

            ceres::Jet<double, N> X_auto_jet[N];
            for (int i = 0; i < N; i++) {
                X_auto_jet[i].a = X[i];
                X_auto_jet[i].v[i] = 1;
            }
            ceres::Jet<double, N> Y_auto_jet[M];
            func(X_auto_jet, Y_auto_jet);
            for (int i = 0; i < M; i++) {
                Y[i] = Y_auto_jet[i].a;
                J.template block<1, N>(i, 0) = Y_auto_jet[i].v.transpose();
            }
        }
    }

    // ESKF滤波器部分
    template<typename ManifoldX>
    struct ESKF {
        inline static constexpr int Nx =  ManifoldX::NxSize;
        inline static constexpr int Nd =  ManifoldX::NdSize;

        ESKF(ManifoldX m) : manifold(std::move(m)) {
            x = Eigen::Matrix<double, Nx, 1>::Zero();
            P = Eigen::Matrix<double, Nd, Nd>::Identity();
            Q = Eigen::Matrix<double, Nd, Nd>::Identity();
        }
       
        // ESKF预测过程
        // func: 运动方程对象
        template<typename Func>
        void process(Func &&func) {
            // 计算广义加法雅克比矩阵
            Eigen::Matrix<double, Nx, Nd> pJ;
            manifold.PlusJacobian(x.data(), pJ.data());
            // 计算状态转移结果和雅克比矩阵
            Eigen::Matrix<double, Nx, Nx> fJ;
            internal::compute_with_jacobian(std::forward<Func>(func), x, x, fJ);
            // 计算广义减法雅克比矩阵
            Eigen::Matrix<double, Nd, Nx> mJ;
            manifold.MinusJacobian(x.data(), mJ.data());
            // 更新误差量的协方差
            Eigen::Matrix<double, Nd, Nd> F = mJ * fJ * pJ;
            P = F * P * F.transpose() + Q;
        }

        // ESKF更新和修正
        // func: 观测方程对象
        // z:    实际观测值
        // V:    观测噪声协方差
        template<typename Func, int Ny>
        void update(Func &&func, const Eigen::Matrix<double, Ny, 1> &z, const Eigen::Matrix<double, Ny, Ny> &V) {
            // 计算广义加法雅克比矩阵
            Eigen::Matrix<double, Nx, Nd> pJ;
            manifold.PlusJacobian(x.data(), pJ.data());
            // 计算观测结果和雅克比矩阵
            Eigen::Vector3d y;
            Eigen::Matrix<double, Ny, Nx> hJ;
            internal::compute_with_jacobian(std::forward<Func>(func), x, y, hJ);
            // 更新后验概率
            Eigen::Matrix<double, Ny, Nd> H = hJ * pJ;
            Eigen::Matrix<double, Nd, Ny> K = P * H.transpose() * (H * P * H.transpose() + V).inverse();
            Eigen::Matrix<double, Nd, 1> dx = K * (z - y);
            P = (Eigen::Matrix<double, Nd, Nd>::Identity() - K * H) * P;
            // 状态修正
            manifold.PlusWithJacobian(x.data(), dx.data(), x.data(), pJ.data());
            Eigen::Matrix<double, Nd, Nx> mJ;
            manifold.MinusJacobian(x.data(), mJ.data());
            Eigen::Matrix<double, Nd, Nd> J = mJ * pJ;
            P = J * P * J.transpose(); 
        }

        Eigen::Matrix<double, Nx, 1> x;  // 系统状态量

        Eigen::Matrix<double, Nd, Nd> P; // 系统误差量协方差
        Eigen::Matrix<double, Nd, Nd> Q; // 系统转移噪声协方差

        ManifoldX manifold;              // 系统状态流型
    };
    

    // 流型基类，实现广义加减法的雅克比矩阵的计算
    template<typename Derived, int Nx, int Nd>
    struct ManifoldBase {
        inline static constexpr int NxSize = Nx;
        inline static constexpr int NdSize = Nd;

        struct PlusFactor {
            template<typename T>
            void operator()(const T *_delta, T *_x_plus_delta) const {
                auto _x = Eigen::Map<const Eigen::Matrix<double, Nx, 1>>(x).template cast<T>().eval();
                ptr->Plus(_x.data(), _delta, _x_plus_delta);
            };
            const Derived *ptr;
            const double *x;
        };

        struct MinusFactor {
            template<typename T>
            void operator()(const T *_y, T *_y_minus_x) const {
                auto _x = Eigen::Map<const Eigen::Matrix<double, Nx, 1>>(x).template cast<T>().eval();
                ptr->Minus(_y, _x.data(), _y_minus_x);
            };
            const Derived *ptr;
            const double *x;
        };
        
        // 计算加法结果和雅克比矩阵
        void PlusWithJacobian(const double *x, const double* delta, double *x_plus_delta, double *jacobian) const {
            Eigen::Map<const Eigen::Matrix<double, Nd, 1>> map_delta(delta);
            Eigen::Map<Eigen::Matrix<double, Nx, 1>> map_x_plus_delta(x_plus_delta);
            Eigen::Map<Eigen::Matrix<double, Nx, Nd>> map_jacobian(jacobian);
            internal::compute_with_jacobian(PlusFactor{static_cast<const Derived*>(this), x}, map_delta, map_x_plus_delta, map_jacobian);
        }

        // 计算加法雅克比矩阵
        void PlusJacobian(const double *x, const double* delta, double *jacobian) const {
            double x_plus_delta[NxSize];
            PlusWithJacobian(x, delta, x_plus_delta, jacobian);
        }

        // 计算加法雅克比矩阵（在delta=0处）
        void PlusJacobian(const double *x, double *jacobian) const {
            const double delta[NdSize] = {0};
            double x_plus_delta[NxSize];
            PlusWithJacobian(x, delta, x_plus_delta, jacobian);
        }

        // 计算减法结果和雅克比矩阵
        void MinusWithJacobian(const double *y, const double* x, double *y_minus_x, double *jacobian) const {
            Eigen::Map<const Eigen::Matrix<double, Nx, 1>> map_y(y);
            Eigen::Map<Eigen::Matrix<double, Nd, 1>> map_y_minus_x(y_minus_x);
            Eigen::Map<Eigen::Matrix<double, Nd, Nx>> map_jacobian(jacobian);
            internal::compute_with_jacobian(MinusFactor{static_cast<const Derived*>(this), x}, map_y, map_y_minus_x, map_jacobian);
        }
        
        // 计算减法雅克比矩阵
        void MinusJacobian(const double *y, const double* x, double *jacobian) const {
            double y_minus_x[NdSize];
            MinusWithJacobian(y, x, y_minus_x, jacobian);
        }

        // 计算减法雅克比矩阵（在y=x处）
        void MinusJacobian(const double* x, double *jacobian) const {
            const double *y = x;
            double y_minus_x[NdSize];
            MinusWithJacobian(y, x, y_minus_x, jacobian);
        }
    };

    // 普通的N维向量流型
    // 通常用于在ProuductManifold里进行组合
    template<int N>
    struct EuclideanManifold : ManifoldBase<EuclideanManifold<N>, N, N> {
        using Base = ManifoldBase<EuclideanManifold<N>, N, N>;

        template<typename T>
        void Plus(const T *x, const T *delta, T *x_plus_delta) const {
            Eigen::Map<const Eigen::Matrix<T, N, 1>> map_x(x);
            Eigen::Map<const Eigen::Matrix<T, N, 1>> map_delta(delta);
            Eigen::Map<Eigen::Matrix<T, N, 1>> map_x_plus_delta(x_plus_delta);
            map_x_plus_delta = map_x + map_delta;
        }

        template<typename T>
        void Minus(const T *y, const T *x, T *y_minus_x) const {
            Eigen::Map<const Eigen::Matrix<T, N, 1>> map_y(y);
            Eigen::Map<const Eigen::Matrix<T, N, 1>> map_x(x);
            Eigen::Map<Eigen::Matrix<T, N, 1>> map_y_minus_x(y_minus_x);
            map_y_minus_x = map_y - map_x;
        }    
    };

    // 组合不同的流型
    template<typename ...Ts>
    struct ProductManifold : ManifoldBase<ProductManifold<Ts...>, (Ts::NxSize + ...), (Ts::NdSize + ...)> {

        ProductManifold(Ts ... others) : other(std::make_tuple(std::move(others)...)) {}

        template<typename T>
        void Plus(const T *x, const T *delta, T *x_plus_delta) const {
            PlusImpl(x, delta, x_plus_delta, std::make_integer_sequence<int, sizeof...(Ts)>());
        }

        template<typename T>
        void Minus(const T *y, const T *x, T *y_minus_x) const {
            MinusImpl(y, x, y_minus_x, std::make_integer_sequence<int, sizeof...(Ts)>());
        }

    private:
        template<int Index, int N, int ...Ns>
        static constexpr int prefix_sum() {
            if constexpr (Index == 0) return 0;
            else return N + prefix_sum<Index-1, Ns...>();
        }

        template<typename T, int ...Ns>
        void PlusImpl(const T *x, const T *delta, T *x_plus_delta, std::integer_sequence<int, Ns...> seq) const {
            int NxIndex[sizeof...(Ts)];
            ((NxIndex[Ns] = prefix_sum<Ns, Ts::NxSize...>()), ...);
            int NdIndex[sizeof...(Ts)];
            ((NdIndex[Ns] = prefix_sum<Ns, Ts::NdSize...>()), ...);

            (std::get<Ns>(other).Plus(x + NxIndex[Ns], delta + NdIndex[Ns], x_plus_delta + NxIndex[Ns]), ...);
        }

        template<typename T, int ...Ns>
        void MinusImpl(const T *y, const T *x, T *y_minus_x, std::integer_sequence<int, Ns...> seq) const {
            int NxIndex[sizeof...(Ts)];
            ((NxIndex[Ns] = prefix_sum<Ns, Ts::NxSize...>()), ...);
            int NdIndex[sizeof...(Ts)];
            ((NdIndex[Ns] = prefix_sum<Ns, Ts::NdSize...>()), ...);

            (std::get<Ns>(other).Minus(y + NxIndex[Ns], x + NxIndex[Ns], y_minus_x + NdIndex[Ns]), ...);
        }

        std::tuple<Ts...> other;
    };

}

#endif /* ESKF_HPP */

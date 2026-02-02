#include <cmath>
#include <functional>

// test_function over square
inline double square_field(double x, double y, double t) {
    constexpr double pi = std::numbers::pi;
    constexpr double tf = 1.0;
    // coefficient function
    auto coe = [](double x, double y) -> double {
        return 0.5 * std::sin(5.0 * std::numbers::pi * x) * std::exp(-x * x) + 1.0;
    };

    double a = 9.0 / 5.0 * t / tf - 2.0;
    double b = coe(y, 1.0) * x * std::cos(a + pi / 2.0);
    double c = coe(x, 1.0) * y * std::sin(a * pi / 2.0);

    double d = 2.0 * pi * (coe(y, 1.0) * x * std::cos(a) - y * std::sin(a));
    double e = 2.0 * pi * (b + c);
    return std::sin(d) * std::cos(e);
}

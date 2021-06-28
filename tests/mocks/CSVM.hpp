#pragma once
#include "plssvm/CSVM.hpp"
#include "plssvm/typedef.hpp"
#include "gmock/gmock.h"

class MockCSVM : public plssvm::CSVM {
  public:
    MockCSVM(real_t cost_, real_t epsilon_, unsigned kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_) : plssvm::CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    MOCK_METHOD(void, learn, (), (override));
    MOCK_METHOD(void, loadDataDevice, (), (override));
    MOCK_METHOD(std::vector<real_t>, CG, (const std::vector<real_t> &b, const int, const real_t), (override));

    const real_t get_num_data_points() const { return num_data_points; }
    const real_t get_num_features() const { return num_features; }
    std::vector<std::vector<real_t>> get_data() const { return data; }
};
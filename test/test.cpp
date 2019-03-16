#include <string>
#include <utility.h>
#include <gtest/gtest.h>
#include <cmath>

const double pos_comp_eps = 1e-10;

class EgoToMapTransformationFixture : public ::testing::Test
{

    protected:

    void SetEgoPose(const double ego_x, const double ego_y, const double ego_theta)
    {
        ego_x_ = ego_x;
        ego_y_ = ego_y;
        ego_theta_ = ego_theta;
    }

    void SetObstaclePoseInEgoFrame(const double obs_x_ego, const double obs_y_ego)
    {
        obs_x_ego_ = obs_x_ego;
        obs_y_ego_ = obs_y_ego;
    }

    void SetExpectationForObstaclePoseInMapFrame(const double expected_obs_x_map, const double expected_obs_y_map)
    {
        expected_obs_x_map_ = expected_obs_x_map;
        expected_obs_y_map_ = expected_obs_y_map;
    }

    void CheckExpectation()
    {
        double obs_x_map, obs_y_map;

        utility::TransformObsFromEgoFrameToMapFrame(ego_x_, ego_y_, ego_theta_, obs_x_ego_, obs_y_ego_, obs_x_map, obs_y_map);

        EXPECT_NEAR(obs_x_map, expected_obs_x_map_, pos_comp_eps);
        EXPECT_NEAR(obs_y_map, expected_obs_y_map_, pos_comp_eps);
    }

    private:

    double 
        ego_x_,
        ego_y_,
        ego_theta_,
        obs_x_ego_,
        obs_y_ego_,
        expected_obs_x_map_,
        expected_obs_y_map_;

};

TEST_F(EgoToMapTransformationFixture, Test1)
{
    SetEgoPose(0.0,0.0,0.0);
    SetObstaclePoseInEgoFrame(1.0, 0.0);
    SetExpectationForObstaclePoseInMapFrame(1.0, 0.0);
    CheckExpectation();
}

TEST_F(EgoToMapTransformationFixture, Test2)
{
    SetEgoPose(5.0,0.0,0.0);
    SetObstaclePoseInEgoFrame(1.0, 0.0);
    SetExpectationForObstaclePoseInMapFrame(6.0, 0.0);
    CheckExpectation();
}

TEST_F(EgoToMapTransformationFixture, Test3)
{
    SetEgoPose(5.0,0.0,M_PI/2.0);
    SetObstaclePoseInEgoFrame(1.0, 0.0);
    SetExpectationForObstaclePoseInMapFrame(5.0, 1.0);
    CheckExpectation();
}

TEST_F(EgoToMapTransformationFixture, Test4)
{
    SetEgoPose(5.0,2.0,M_PI/2.0);
    SetObstaclePoseInEgoFrame(1.0, -2.0);
    SetExpectationForObstaclePoseInMapFrame(7.0,3.0);
    CheckExpectation();
}

TEST_F(EgoToMapTransformationFixture, Test5)
{
    SetEgoPose(5.0,2.0,-M_PI/2.0);
    SetObstaclePoseInEgoFrame(1.0, -2.0);
    SetExpectationForObstaclePoseInMapFrame(3.0,1.0);
    CheckExpectation();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
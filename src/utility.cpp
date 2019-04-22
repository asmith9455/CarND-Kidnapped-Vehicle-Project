#include "utility.h"

#include <cmath>

namespace utility
{
    void TransformObsFromEgoFrameToMapFrame(const double ego_x, const double ego_y, const double ego_theta, const double obs_x_ego, const double obs_y_ego, double& obs_x_map, double& obs_y_map)
    {
        //first rotate by -theta then translate by ego position
        obs_x_map = std::cos(ego_theta) * obs_x_ego - std::sin(ego_theta) * obs_y_ego + ego_x;
        obs_y_map = std::sin(ego_theta) * obs_x_ego + std::cos(ego_theta) * obs_y_ego + ego_y;
    }
}
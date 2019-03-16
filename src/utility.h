#ifndef UTILITY_H_
#define UTILITY_H_

namespace utility
{

void TransformObsFromEgoFrameToMapFrame(const double ego_x, const double ego_y, const double ego_theta, const double obs_x_ego, const double obs_y_ego, double& obs_x_map, double& obs_y_map);
}

#endif
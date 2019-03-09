/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  num_particles = 100; // TODO: Set the number of particles

  using Distribution = std::normal_distribution<double>;

  Distribution dist_x_{x, std[0]};
  Distribution dist_y_{y, std[1]};
  Distribution dist_theta_{theta, std[2]};

  gen = std::mt19937{rd()};

  for (int i = 0; i < num_particles; ++i)
  {
    Particle p;
    p.id = i;
    p.x = dist_x_(gen);
    p.y = dist_y_(gen);
    p.theta = dist_theta_(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
}

void BicycleModel(const double dt, const double speed, const double yaw_rate, const double speed_sd, const double yaw_rate_sd, Particle &particle)
{
  const double r = speed / yaw_rate;
  using ::std::cos;
  using ::std::sin;
  const double new_theta = particle.theta + dt * yaw_rate;

  // std::normal_distribution<double> dist_x{};
  // std::normal_distribution<double> dist_y{};
  // std::normal_distribution<double> dist_theta{};

  particle.x = r * (sin(new_theta) - sin(particle.theta));
  particle.y = r * (cos(particle.theta) - cos(new_theta));
  particle.theta = new_theta;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  for (auto &particle : particles)
  {
    BicycleModel(delta_t, velocity, yaw_rate, std_pos[0], std_pos[1], particle);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
}

double NormalDist(const double mux, const double muy, const double sdx, const double sdy, const double x, const double y)
{
  using std::exp;
  using std::pow;

  return 1 / 2 / M_PI / sdx / sdy * exp(0.5 * (pow(x - mux, 2) / pow(sdx, 2) + pow(y - muy, 2) / pow(sdy, 2)));
}

void TransformEgoToMap(const double ego_x, const double ego_y, const double ego_theta, const double x, const double y, double &xt, double &yt)
{
  //first rotate by -theta
  xt = std::cos(-ego_theta) * x - std::sin(ego_theta) * y;
  yt = std::sin(ego_theta) * x + std::cos(ego_theta) * y;

  //now account for translation

  xt += ego_x;
  yt += ego_y;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  const auto best_particle_it = std::max_element(particles.begin(), particles.end(), [](const Particle &p1, const Particle &p2) { return p1.weight < p2.weight; });

  for (auto &p : particles)
  {
    //we need to calculate the weight for this particle
    //we can do this by determining which landmarks are associated with each of our observations.
    //once we have the associations, we can
    double weight = 1.0;

    for (const auto &obs : observations)
    {
      Map::single_landmark_s best_landmark;
      std::double_t best_landmark_distance = std::numeric_limits<std::double_t>::max();

      //transform the observation into the map frame
      double obs_x_map, obs_y_map;
      TransformEgoToMap(best_particle_it->x, best_particle_it->y, best_particle_it->theta, obs.x, obs.y, obs_x_map, obs_y_map);

      for (const auto &landmark : map_landmarks.landmark_list)
      {
        //@todo: should also skip the landmark if it is out of range of the sensor
        const auto distance = dist(landmark.x_f, landmark.y_f, obs_x_map, obs_y_map);
        if (distance < best_landmark_distance)
        {
          best_landmark_distance = distance;
          best_landmark = landmark;
        }
      }

      //calculate the weight given the best landmark for this observation
      //@todo: what if 2 observations associate with the same landmark?

      p.weight *= NormalDist(best_landmark.x_f, best_landmark.y_f, std_landmark[0], std_landmark[1], obs_x_map, obs_y_map);
    }
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  using std::back_inserter;
  using std::discrete_distribution;
  using std::transform;
  using std::vector;

  vector<double> weights;

  transform(particles.begin(), particles.end(), back_inserter(weights), [](const Particle &p) { return p.weight; });

  discrete_distribution<int> index_dist{weights.begin(), weights.end()};

  vector<Particle> new_particles;

  transform(particles.begin(), particles.end(), back_inserter(new_particles),
            [this, &index_dist](const Particle &p) { return particles[index_dist(gen)]; });

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
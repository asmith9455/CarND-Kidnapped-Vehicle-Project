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
#include "utility.h"

#include <matplotlibcpp.h>

using std::string;
using std::vector;

namespace mpl = matplotlibcpp;

template<typename T>
T clamp(const T value, const T min, const T max)
{
  return ::std::min(::std::max(value, min), max);
}

void ParticleFilter::PlotParticles(
  const ::std::string& title,
  const ::std::vector<Particle>& particles,
  const double mean_x, 
  const double mean_y, 
  const double mean_theta,
  const double mean_weight,
  const bool scale_by_weights)
{
  if (enable_visualization)
  {
    static const auto quiver_scalar{1e-2};

    ::std::vector<double> 
      init_xs, 
      init_ys, 
      init_us, 
      init_ws, 
      mean_xs{mean_x}, 
      mean_ys{mean_y}, 
      mean_us{quiver_scalar * ::std::cos(mean_theta)}, 
      mean_ws{quiver_scalar * ::std::sin(mean_theta)};

    for (const auto& p : particles)
    {
      init_xs.push_back(p.x);
      init_ys.push_back(p.y);
      
      auto size_scalar = quiver_scalar;
      
      if(scale_by_weights)
      {
        size_scalar *= p.weight;
      }

      size_scalar = clamp(size_scalar, 1e-6, 1.0);

      init_us.push_back(size_scalar * ::std::cos(mean_theta));
      init_ws.push_back(size_scalar * ::std::sin(mean_theta));
    }

    if(scale_by_weights)
    {
      const auto mean_scalar = clamp(quiver_scalar * mean_weight, 1e-6, 1.0); 
      init_us.at(0) *= mean_scalar;
      init_ws.at(0) *= mean_scalar;
    }
    else
    {
      const auto mean_scalar = clamp(quiver_scalar, 1e-6, 1.0); 
      init_us.at(0) *= mean_scalar;
      init_ws.at(0) *= mean_scalar;
    }

    mpl::quiver(init_xs, init_ys, init_us, init_ws);
    mpl::quiver(mean_xs, mean_ys, mean_us, mean_ws, ::std::map<::std::string, ::std::string>{{"facecolors", "r"}});

    // ::std::vector<double> landmark_xs, landmark_ys;

    // for(const auto& landmark : map_landmarks.landmark_list)
    // {
    //   landmark_xs.push_back(landmark.x_f);
    //   landmark_ys.push_back(landmark.y_f);
    // }
    // mpl::scatter(landmark_xs, landmark_ys, 2.0);

    mpl::title(title);

    mpl::show();
  }
}

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

  PlotParticles("Initialization", particles, x, y, theta);

  is_initialized = true;
}

void ParticleFilter::BicycleModel(const double dt, const double speed, const double yaw_rate, const double x_sd, const double y_sd, const double theta_sd, Particle &particle)
{
  using ::std::cos;
  using ::std::sin;

  if (std::fabs(yaw_rate) < 1e-10)
  {
    particle.x += speed * cos(particle.theta) * dt;
    particle.y += speed * sin(particle.theta) * dt;
  }
  else
  {
    const double new_theta = particle.theta + dt * yaw_rate;
    const double r = speed / yaw_rate;
    particle.x += r * (sin(new_theta) - sin(particle.theta));
    particle.y += r * (cos(particle.theta) - cos(new_theta));
    particle.theta = new_theta;
  }

  using Distribution = std::normal_distribution<double>;

  Distribution x_dist{0.0, x_sd};
  Distribution y_dist{0.0, y_sd};
  Distribution theta_dist{0.0, theta_sd};

  particle.x += x_dist(gen);
  particle.y += y_dist(gen);
  particle.theta += theta_dist(gen);
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

  ::std::cout << "-------------------------------" << ::std::endl;
  ::std::cout << "Prediction Step" << ::std::endl;
  ::std::cout << "delta_t" << delta_t << ::std::endl;
  ::std::cout << "velocity" << velocity << ::std::endl;
  ::std::cout << "yaw_rate" << yaw_rate << ::std::endl;
  ::std::cout << "x std dev" << std_pos[0] << ::std::endl;
  ::std::cout << "y std dev" << std_pos[1] << ::std::endl;
  ::std::cout << "theta std dev" << std_pos[2] << ::std::endl;

  for (auto &particle : particles)
  {
    BicycleModel(delta_t, velocity, yaw_rate, std_pos[0], std_pos[1], std_pos[2], particle);
  }

  const auto best_particle_it2 = std::max_element(particles.begin(), particles.end(), [](const Particle &p1, const Particle &p2) { return p1.weight < p2.weight; });

  PlotParticles(
    "After prediction with noise",
    particles, 
    {best_particle_it2->x}, 
    {best_particle_it2->y}, 
    {best_particle_it2->theta}, 
    {best_particle_it2->weight},
    true);
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

  const auto x_term_1 = (x-mux)*(x-mux)/sdx/sdx;
  const auto y_term_1 = (y-muy)*(y-muy)/sdy/sdy;
  const auto scalar = 1.0/2.0/M_PI/sdx/sdy;
  
  //return 1.0 / 2.0 / M_PI / sdx / sdy * exp(0.5 * (pow(x - mux, 2.0) / pow(sdx, 2.0) + pow(y - muy, 2.0) / pow(sdy, 2.0)));
  return scalar*exp(-0.5*(x_term_1+y_term_1));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations)
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

  // std::cout << "best particle before weight update: " << best_particle_it->weight << std::endl;

  std::cout << "STARTING UPDATE-------------- (with " << particles.size() << " particles)" << std::endl;

  for (int i = 0; i < particles.size(); ++i)
  {
    Particle& p = particles[i];
    //we need to calculate the weight for this particle
    //we can do this by determining which landmarks are associated with each of our observations.
    //once we have the associations, we can
    // double weight = 1.0; //todo: should the particle weight start at 1.0 again?

    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    p.best_dists.clear();
    p.weight = 1.0;

    // std::cout << "(p: " << p.id << "; ";

    // std::cout << "number  of observations is: " << observations.size() << "; ";

    for (const auto &obs : observations)
    {
      Map::single_landmark_s best_landmark;
      bool found_landmark{false};
      std::double_t best_landmark_distance = std::numeric_limits<std::double_t>::max();

      //transform the observation into the map frame
      double obs_x_map, obs_y_map;
      utility::TransformObsFromEgoFrameToMapFrame(p.x, p.y, p.theta, obs.x, obs.y, obs_x_map, obs_y_map);

      for (const auto &landmark : map_landmarks.landmark_list)
      {
        // const double range_to_landmark = dist(p.x, p.y, landmark.x_f, landmark.y_f);

        // const bool out_of_range = range_to_landmark > sensor_range + 1.0;

        // if (out_of_range)
        //   continue;

        const auto distance = dist(landmark.x_f, landmark.y_f, obs_x_map, obs_y_map);
        if (distance < best_landmark_distance)
        {
          best_landmark_distance = distance;
          best_landmark = landmark;
          found_landmark = true;
        }
      }

      ::std::cout << best_landmark_distance << ", ";

      assert(found_landmark);

      //calculate the weight given the best landmark for this observation
      const double new_weight = NormalDist(best_landmark.x_f, best_landmark.y_f, std_landmark[0], std_landmark[1], obs_x_map, obs_y_map);
      p.weight *= new_weight;
      // std::cout << "new_weight: " << std::to_string(new_weight) << ", " << std::endl;
      p.associations.push_back(best_landmark.id_i);
      p.sense_x.push_back(best_landmark.x_f);
      p.sense_y.push_back(best_landmark.y_f);
      p.best_dists.push_back(best_landmark_distance);
    }

    ::std::cout << ::std::endl;
    // std::cout << "w: " << std::to_string(p.weight) << "), ";
  }

  std::cout << std::endl;

  const auto best_particle_it2 = std::max_element(particles.begin(), particles.end(), [](const Particle &p1, const Particle &p2) { return p1.weight < p2.weight; });
  const auto worst_particle_it2 = std::min_element(particles.begin(), particles.end(), [](const Particle &p1, const Particle &p2) { return p1.weight > p2.weight; });

  PlotParticles(
    "After weight updates",
    particles, 
    {best_particle_it2->x}, 
    {best_particle_it2->y}, 
    {best_particle_it2->theta}, 
    {best_particle_it2->weight},
    true);

  std::cout << "best particle after weight update: " << std::to_string(best_particle_it2->weight) << std::endl;
  for(const double& dist : best_particle_it2->best_dists)
  {
    std::cout << dist << ", ";
  }
  std::cout << "worst particle after weight update: " << std::to_string(worst_particle_it2->weight) << std::endl;

  std::cout << "best particle: (" << std::to_string(best_particle_it2->x) << ", " << std::to_string(best_particle_it2->y) << ", " << std::to_string(best_particle_it2->theta) << ")" << std::endl;

  std::cout << "ENDING UPDATE-------------- (with " << particles.size() << " particles)" << std::endl;
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

  // ::std::cout << "randomly selected indices are: " << ::std::endl;

  for(std::size_t i = 0; i < particles.size(); ++i)
  {
    const std::size_t randomly_selected_index = index_dist(gen);
    // std::cout << randomly_selected_index << ", "; 
    new_particles.push_back(particles[randomly_selected_index]);
  }

  // ::std::cout << ::std::endl;

  particles = new_particles;

  const auto best_particle_it2 = std::max_element(particles.begin(), particles.end(), [](const Particle &p1, const Particle &p2) { return p1.weight < p2.weight; });

  PlotParticles(
    "After Resampling",
    particles, 
    {best_particle_it2->x}, 
    {best_particle_it2->y}, 
    {best_particle_it2->theta}, 
    {best_particle_it2->weight},
    true);
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
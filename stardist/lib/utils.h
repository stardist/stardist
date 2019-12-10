#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <string>
#include <cmath>


class ProgressBar {
  float curr_percentage;
  const float eps;
  const int width;
  const std::string label;
  
 public:
  ProgressBar(const std::string label,
              const int width = 50,
              const float eps = 1);
  
  void update(const float);
  void finish();
};

#endif /* UTILS_H */

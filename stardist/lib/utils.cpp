#include "utils.h"


ProgressBar::ProgressBar(const std::string label,const int width, const float eps): width(width),label(label),eps(eps), curr_percentage(0){};

void ProgressBar::finish(){
  printf("\n");
}

void ProgressBar::update(const float percentage){

  if (fabs(percentage-curr_percentage)>eps){
    int w = width*percentage/100;
    std::string s = std::string(w, '#') + std::string(width-w, ' ');
    printf("\r|%s| [%.0f %% %s]",s.c_str(), percentage, label.c_str());
    fflush(stdout);
    curr_percentage = percentage;
  }

}

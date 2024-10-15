#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <opencv2/opencv.hpp>

cv::Mat calculateSIFT(const cv::Mat& image);
cv::Mat calculateORB(const cv::Mat& image);
cv::Mat calculateColorHistogram(const cv::Mat& image);
cv::Mat calculateColorCorrelogram(const cv::Mat& image);

#endif // FEATURE_EXTRACTION_H

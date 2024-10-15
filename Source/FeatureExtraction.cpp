#include "FeatureExtraction.h"
#include <opencv2/opencv.hpp>

using namespace cv;

Mat calculateSIFT(const Mat& image) {
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(image, noArray(), keypoints, descriptors);
    return descriptors;
}

Mat calculateORB(const Mat& image) {
    Ptr<ORB> orb = ORB::create();
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detectAndCompute(image, noArray(), keypoints, descriptors);
    return descriptors;
}

Mat calculateColorHistogram(const Mat& image) {
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    int hBins = 50, sBins = 60;
    int histSize[] = { hBins, sBins };
    float hRanges[] = { 0, 180 };
    float sRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges, sRanges };
    MatND hist;

    int channels[] = { 0, 1 };
    calcHist(&hsvImage, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist.reshape(1, 1);
}

Mat calculateColorCorrelogram(const Mat& image) {
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    int hBins = 50, sBins = 60;
    int histSize[] = { hBins, sBins };
    float hRanges[] = { 0, 180 };
    float sRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges, sRanges };

    Mat correlogram = Mat::zeros(hBins * sBins, hBins * sBins, CV_32F);

    for (int i = 0; i < hsvImage.rows; i++) {
        for (int j = 0; j < hsvImage.cols; j++) {
            int h = hsvImage.at<Vec3b>(i, j)[0];
            int s = hsvImage.at<Vec3b>(i, j)[1];

            int bin1 = (h * hBins / 180) * sBins + (s * sBins / 256);

            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    int ni = i + di;
                    int nj = j + dj;

                    if (ni >= 0 && ni < hsvImage.rows && nj >= 0 && nj < hsvImage.cols) {
                        int nh = hsvImage.at<Vec3b>(ni, nj)[0];
                        int ns = hsvImage.at<Vec3b>(ni, nj)[1];

                        int bin2 = (nh * hBins / 180) * sBins + (ns * sBins / 256);

                        correlogram.at<float>(bin1, bin2)++;
                    }
                }
            }
        }
    }

    normalize(correlogram, correlogram, 0, 1, NORM_MINMAX, -1, Mat());

    return correlogram.reshape(1, 1);
}

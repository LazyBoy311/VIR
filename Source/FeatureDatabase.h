#ifndef FEATURE_DATABASE_H
#define FEATURE_DATABASE_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class FeatureDatabase {
public:
    FeatureDatabase(const string& featureFolder, const string& imageFolder); // Constructor khởi tạo đối tượng FeatureDatabase
    void initializeDatabase(); // Khởi tạo cơ sở dữ liệu đặc trưng
    vector<pair<string, double>> queryImage(const cv::Mat& queryImage, const string& featureType, int k); // Truy vấn hình ảnh dựa trên đặc trưng và số lượng kết quả trả về

private:
    void saveFeatures(); // Lưu các đặc trưng vào tệp tin
    void loadFeatures(); // Tải các đặc trưng từ tệp tin
    void saveAdditionalFeatures(const string& additionalImageFolder); // Lưu các đặc trưng bổ sung của ảnh vào tệp tin
    double calculateHistSimilarity(const cv::Mat& hist1, const cv::Mat& hist2); // Tính toán độ tương đồng của histogram màu
    double calculateDescriptorSimilarity(const cv::Mat& desc1, const cv::Mat& desc2); // Tính toán độ tương đồng của mô tả ORB hoặc SIFT

    // Biến thể cấu trúc chứa các đặc trưng của ảnh
    struct Features {
        cv::Mat colorHistogram;     // Histogram màu
        cv::Mat colorCorrelogram;   // Color Correlogram
        cv::Mat siftDescriptor;     // Mô tả SIFT
        cv::Mat orbDescriptor;      // Mô tả ORB
    };

    string featureFolder;   // Đường dẫn thư mục chứa các đặc trưng
    string imageFolder;     // Đường dẫn thư mục chứa các ảnh gốc
    string featureType;     // Loại đặc trưng hiện tại (có thể không cần thiết)
    unordered_map<string, Features> featureData; // Dữ liệu chứa các đặc trưng của từng ảnh
};
#endif
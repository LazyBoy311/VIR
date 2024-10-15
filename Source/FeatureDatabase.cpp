#include "FeatureDatabase.h"
#include "FeatureExtraction.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <queue>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// Khởi tạo đối tượng FeatureDatabase với đường dẫn thư mục chứa đặc trưng và ảnh
FeatureDatabase::FeatureDatabase(const string& featureFolder, const string& imageFolder)
    : featureFolder(featureFolder), imageFolder(imageFolder) {}

// Hàm khởi tạo cơ sở dữ liệu đặc trưng
void FeatureDatabase::initializeDatabase() {
    // Nếu thư mục đặc trưng không tồn tại hoặc trống, lưu các đặc trưng
    if (!fs::exists(featureFolder) || fs::is_empty(featureFolder)) {
        saveFeatures();
    }
    else {
        // Nếu đã có dữ liệu, tải các đặc trưng đã tính toán trước đó
        loadFeatures();
    }
}

// Hàm lưu đặc trưng vào tệp tin yml cho một ảnh
void FeatureDatabase::saveImageFeatures(const string& imagePath, const string& baseName) {
    // Đọc ảnh từ đường dẫn
    Mat image = imread(imagePath);
    // Bỏ qua nếu không thể đọc được ảnh
    if (image.empty()) {
        cout << "Khong the xu ly anh: " << baseName << endl;
        return;
    }

    cout << "Dang xu ly anh: " << baseName << endl;

    // Tính các đặc trưng của ảnh
    Mat colorHist = calculateColorHistogram(image);
    Mat colorCorrelogram = calculateColorCorrelogram(image);
    Mat siftDescriptor = calculateSIFT(image);
    Mat orbDescriptor = calculateORB(image);

    // Đường dẫn tệp tin chứa các đặc trưng
    string featureFile = featureFolder + "/" + baseName + "_features.yml";
    FileStorage fs(featureFile, FileStorage::WRITE);
    fs << "ColorHistogram" << colorHist;
    fs << "ColorCorrelogram" << colorCorrelogram;
    fs << "SIFT" << siftDescriptor;
    fs << "ORB" << orbDescriptor;
    fs.release();
}

// Hàm lưu các đặc trưng của ảnh từ thư mục chính
void FeatureDatabase::saveFeatures() {
    // Duyệt qua các tệp ảnh trong thư mục ảnh
    for (const auto& entry : fs::directory_iterator(imageFolder)) {
        string imagePath = entry.path().string();
        // Chỉ xử lý các tệp .jpg hoặc .png
        if (imagePath.find(".jpg") != string::npos || imagePath.find(".png") != string::npos) {
            // Tên cơ sở của tệp ảnh
            string baseName = entry.path().stem().string();
            // Lưu các đặc trưng
            saveImageFeatures(imagePath, baseName);
        }
    }
}

// Hàm lưu các đặc trưng bổ sung của ảnh từ thư mục bổ sung
void FeatureDatabase::saveAdditionalFeatures(const string& additionalImageFolder) {
    // Duyệt qua các tệp ảnh trong thư mục ảnh bổ sung
    for (const auto& entry : fs::directory_iterator(additionalImageFolder)) {
        string imagePath = entry.path().string();
        // Chỉ xử lý các tệp .jpg hoặc .png
        if (imagePath.find(".jpg") != string::npos || imagePath.find(".png") != string::npos) {
            // Tên cơ sở của tệp ảnh
            string baseName = entry.path().stem().string();
            // Lưu các đặc trưng
            saveImageFeatures(imagePath, baseName);
        }
    }
}


// Hàm truy vấn hình ảnh dựa trên đặc trưng được chỉ định
vector<pair<string, double>> FeatureDatabase::queryImage(const Mat& queryImage, const string& featureType, int k) {
    Mat queryFeature;

    // Tính toán đặc trưng của ảnh truy vấn dựa trên loại đặc trưng chỉ định
    if (featureType == "ColorHistogram") {
        queryFeature = calculateColorHistogram(queryImage);
    }
    else if (featureType == "ColorCorrelogram") {
        queryFeature = calculateColorCorrelogram(queryImage);
    }
    else if (featureType == "SIFT") {
        queryFeature = calculateSIFT(queryImage);
    }
    else if (featureType == "ORB") {
        queryFeature = calculateORB(queryImage);
    }
    else {
        cerr << "Loại đặc trưng không được hỗ trợ!" << endl;
        return {}; // Trả về vector rỗng nếu loại đặc trưng không được hỗ trợ
    }

    // Priority queue để lưu trữ các điểm tương đồng
    priority_queue<pair<double, string>, vector<pair<double, string>>, less<pair<double, string>>> pq;

    // Tính toán điểm tương đồng giữa ảnh truy vấn và các đặc trưng có sẵn
    for (const auto& pair : featureData) {
        const string& featureFileName = pair.first;
        const Features& features = pair.second;

        double similarityScore;

        // Tính toán điểm tương đồng dựa trên loại đặc trưng chỉ định
        if (featureType == "ColorHistogram") {
            similarityScore = calculateHistSimilarity(queryFeature, features.colorHistogram);
        }
        else if (featureType == "ColorCorrelogram") {
            similarityScore = calculateHistSimilarity(queryFeature, features.colorCorrelogram);
        }
        else if (featureType == "SIFT") {
            similarityScore = calculateDescriptorSimilarity(queryFeature, features.siftDescriptor);
        }
        else if (featureType == "ORB") {
            similarityScore = calculateDescriptorSimilarity(queryFeature, features.orbDescriptor);
        }

        // Thêm điểm tương đồng vào priority queue
        pq.push(make_pair(similarityScore, featureFileName + "_features.yml"));

        // Giới hạn kích thước của priority queue để chỉ giữ top K phần tử
        if (pq.size() > k) {
            pq.pop();
        }
    }

    // Lấy ra các tên ảnh có điểm tương đồng cao nhất từ priority queue
    vector<pair<string, double>> topKImages;
    while (!pq.empty()) {
        topKImages.push_back(make_pair(pq.top().second, pq.top().first));
        pq.pop();
    }
    return topKImages;
}


// Hàm tính toán điểm tương đồng dựa trên histogram màu
double FeatureDatabase::calculateHistSimilarity(const Mat& hist1, const Mat& hist2) {
    // Sử dụng khoảng cách Chi-square cho histogram màu
    return compareHist(hist1, hist2, HISTCMP_CHISQR_ALT);
}

// Hàm tính toán điểm tương đồng dựa trên mô tả ORB hoặc SIFT
double FeatureDatabase::calculateDescriptorSimilarity(const Mat& desc1, const Mat& desc2) {
    // Kiểm tra xem các descriptor có trống không
    if (desc1.empty() || desc2.empty()) {
        cerr << "Empty descriptors passed to calculateDescriptorSimilarity" << endl;
        return DBL_MAX;
    }

    // Tạo đối tượng BFMatcher với khoảng cách Euclidean (NORM_L2)
    BFMatcher matcher(NORM_L2);

    // Vector để lưu trữ các kết quả đối sánh
    vector<DMatch> matches;

    // Thực hiện đối sánh các descriptor
    matcher.match(desc1, desc2, matches);

    // Tính toán độ tương đồng dựa trên khoảng cách của các đối sánh
    double similarity = 0.0;
    for (const auto& match : matches) {
        similarity += match.distance;
    }

    // Trả về giá trị trung bình của khoảng cách các đối sánh
    return similarity / matches.size();
}



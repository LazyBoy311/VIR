#include "FeatureDatabase.h"
#include <fstream> 
#include <chrono>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;
using namespace std::chrono;

/**/
// Hàm hiển thị kết quả từ tập tin
void displayResults(const vector<pair<string, double>>& results, const string& imageFolder) {
    int cols = 5;
    int rows = (results.size() + cols - 1) / cols;
    int imgWidth = 200;
    int imgHeight = 200;

    Mat displayImg = Mat::zeros(rows * imgHeight, cols * imgWidth, CV_8UC3);

    for (int i = 0; i < results.size(); ++i) {
        string imageFile = imageFolder + "/" + results[i].first.substr(0, results[i].first.find_last_of('_')) + ".jpg";
        Mat img = imread(imageFile);

        // Kiểm tra xem ảnh có được tải thành công và kích thước hợp lệ hay không
        if (img.empty() || img.cols <= 0 || img.rows <= 0) {
            cerr << "Could not load image: " << imageFile << endl;
            continue;
        }

        resize(img, img, Size(imgWidth, imgHeight));

        int row = i / cols;
        int col = i % cols;

        img.copyTo(displayImg(Rect(col * imgWidth, row * imgHeight, imgWidth, imgHeight)));

        putText(displayImg, to_string(results[i].second), Point(col * imgWidth, row * imgHeight + 20),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    }

    if (!displayImg.empty() && displayImg.cols > 0 && displayImg.rows > 0) {
        imshow("Query Results", displayImg);
        waitKey(0);
    }
    else {
        cerr << "No valid images to display." << endl;
    }
}


// Hàm lưu trữ kết quả vào tập tin
void saveResultsToFile(const vector<pair<string, double>>& results, const string& resultFile) {
   ofstream outFile(resultFile);
    if (outFile.is_open()) {
        for (const auto& result : results) {
            outFile << result.first << " " << result.second << endl;
        }
        outFile.close();
    }
    else {
        cerr << "Could not open the file to write results!" <<endl;
    }
}

// Hàm đọc kết quả từ tập tin
vector<pair<string, double>> loadResultsFromFile(const string& resultFile) {
   vector<pair<string, double>> results;
    ifstream inFile(resultFile);
    if (inFile.is_open()) {
        string filename;
        double distance;
        while (inFile >> filename >> distance) {
            results.emplace_back(filename, distance);
        }
        inFile.close();
    }
    else {
        cerr << "Could not open the file to read results!" << endl;
    }
    return results;
}

/**/
int main() {
    string featureFolder = "../features_1"; // Đường dẫn thư mục chứa đặc trưng
    string imageFolder = "../images_1"; // Đường dẫn thư mục chứa ảnh gốc
    string queryImagePath = "../query/01.jpg"; // Đường dẫn tới ảnh truy vấn
    string resultFile = "../results.txt"; // Đường dẫn tới tệp kết quả


    // Chuẩn bị CSDL đặc trưng
    FeatureDatabase db(featureFolder, imageFolder);
    db.initializeDatabase();

    Mat queryImage = imread(queryImagePath);

    while (true) {
        string featureType; // Loại đặc trưng
        int numImages; // Số lượng ảnh muốn in ra
        cout << "Enter feature type (ColorHistogram/ColorCorrelogram/SIFT/ORB) or 0 to exit: ";
        cin >> featureType;
        if (featureType == "0") {
            break;
        }

        cout << "Enter number of similar images to display: ";
        cin >> numImages;

        // Bắt đầu đo thời gian
        auto start = high_resolution_clock::now();

        vector<pair<string, double>> similarImages = db.queryImage(queryImage, featureType, numImages);

        // Kết thúc đo thời gian
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);

        for (const auto& pair : similarImages) {
            cout << "Similar image: " << pair.first << ", similarity score: " << pair.second << endl;
        }

        // Lưu kết quả vào tệp tin
        saveResultsToFile(similarImages, resultFile);

        // Hiển thị kết quả
        vector<pair<string, double>> loadedResults = loadResultsFromFile(resultFile);
        displayResults(loadedResults, imageFolder);
        cout << "Time taken for the query: " << duration.count() << " milliseconds" << endl;
    }

    cout << "Image query completed." <<endl;
    return 0;
}


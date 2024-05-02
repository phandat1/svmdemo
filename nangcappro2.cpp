#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <graphics.h>
#include <algorithm>
#include <bits/stdc++.h>
using namespace std;

// cau truc luu tru cau truc diem du lieu gom dac trung va nhan.
struct DataPoint {
    vector<double> features; // cac dac trung cua du lieu.
    int label; // nhan cua diem du lieu 1 or -1.
};

// ham tinh tich vo huong cua hai vector.
double dotProduct(const vector<double>& v1, const vector<double>& v2) {
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// ham xao tron du lieu ngau nhien de tranh bias trong qua trinh huan luyen.
void shuffleData(vector<DataPoint>& data) {
    random_device rd; // thiet bi tao so ngau nhien.
    mt19937 g(rd()); // may tao so ngau nhien.
    shuffle(data.begin(), data.end(), g); // xao tron du lieu.
}

// ham huan luyen mo hinh SVM.
vector<double> trainSVM(vector<DataPoint>& data, double learningRate = 0.01, int epochs = 1000, double lambda = 0.01) {
    if (data.empty()) return {};

    int featureSize = data[0].features.size(); // kich thuoc cua dac trung.
    vector<double> w(featureSize, 0.0); // khoi tao vector trong so W.
    double b = 0.0; // khoi tao trong so b.

    // khoi tao trong so w ngau nhien.
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-0.1, 0.1);
    for (double& weight : w) {
        weight = dis(gen);
    }

    //vong lap huan luyen.
    for (int epoch = 0; epoch < epochs; ++epoch) {
        shuffleData(data); // xao tron du lieu truoc moi epoch.
        int correctCount = 0;
        for (auto& point : data) {
            double margin = point.label * (dotProduct(w, point.features) + b);
            if (margin < 1) {
                for (size_t i = 0; i < featureSize; ++i) {
                    w[i] -= learningRate * (lambda * w[i] - point.label * point.features[i]);
                }
                b -= learningRate * (-point.label);
            } else {
                correctCount++;
            }
        }
        learningRate *= 0.99; //giam toc do sau moi epoch.

        // tinh va in do chinh xac sau moi epoch.
        double accuracy = static_cast<double>(correctCount) / static_cast<double>(data.size());
        cout << "Epoch " << (epoch + 1) << " Accuracy: " << (accuracy * 100.0) << "%" << endl;
    }
    w.push_back(b); // them sai so b vao cuoi vector trong so w.
    return w;
}

// Ham ve do thi cac diem du lieu va sieu mat phang.
void drawGraph(const vector<DataPoint>& data, const vector<double>& w) {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, NULL); // khoi tao do hoa.

    // tim gia tri lon nhat va nho nhat cua cac dac trung de ve do thi.
    double maxX = data[0].features[0];
    double minX = data[0].features[0];
    double maxY = data[0].features[1];
    double minY = data[0].features[1];
    for (const auto& point : data) {
        if (point.features[0] > maxX) maxX = point.features[0];
        if (point.features[0] < minX) minX = point.features[0];
        if (point.features[1] > maxY) maxY = point.features[1];
        if (point.features[1] < minY) minY = point.features[1];
    }

    // Tinh ti le chuyen doi toa do thuc te sang toa do do hoa.
    double xScale = (getmaxx() - 200) / (maxX - minX);
    double yScale = (getmaxy() - 200) / (maxY - minY);

    // ve cac diem du lieu.
    for (const auto& point : data) {
        int x = static_cast<int>(200 + (point.features[0] - minX) * xScale);
        int y = static_cast<int>(200 + (maxY - point.features[1]) * yScale);
        int color = point.label == 1 ? GREEN : RED;
        setcolor(color);
        circle(x, y, 5);
        floodfill(x, y, color);
    }

    // ve sieu mat phang.
    if (w.size() >= 3) {
        setcolor(WHITE);
        int x1 = 0;
        int y1 = static_cast<int>(200 + (maxY - ((-w.back() - w[0] * minX) / w[1])) * yScale);
        int x2 = getmaxx();
        int y2 = static_cast<int>(200 + (maxY - ((-w.back() - w[0] * maxX) / w[1])) * yScale);
        line(x1, y1, x2, y2);
    }

    getch(); // doi nguoi dung nhap.
    closegraph(); // dong cua so do hoa.
}

int main() {
    // du lieu huan luyen.
    vector<DataPoint> data = {
        {{2, 3}, 1}, {{4, 1}, -1}, {{4, 2}, 1}, {{2, 3}, -1}, {{1, 8}, -1}
    };

    // huan luyen mo hinh va lay cac tham so.
    vector<double> svmParameters = trainSVM(data);
    
    // in ra cac tham so cua mo hinh.
    cout << "Model parameters:" << endl;
    for (size_t i = 0; i < svmParameters.size() - 1; ++i) {
        cout << "w" << i << " = " << svmParameters[i] << ", ";
    }
    cout << "b = " << svmParameters.back() << endl;

    // ve cac diem du lieu va sieu mat phang.
    drawGraph(data, svmParameters);

    return 0;
}

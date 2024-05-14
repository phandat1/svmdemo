#include <bits/stdc++.h>
#include <graphics.h>

using namespace std;

// cau truc luu tru cau truc diem du lieu gom dac trung va nhan.

struct DataPoint {
    vector<double> features; // cac dac trung cua du lieu.
    int label;// nhan cua diem du lieu 1 or -1.
};

// ham tinh tich vo huong cua hai vector.

double dotProduct(const vector<double>& v1, const vector<double>& v2) {
    double result = 0.0;
    for (int i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// ham xao tron du lieu ngau nhien de tranh bias trong qua trinh huan luyen.

void initializeWeights(vector<double>& w, int featureSize) {
    random_device rd; // thiet bi tao so ngau nhien.
    mt19937 gen(rd()); // may tao so ngau nhien.
    uniform_real_distribution<> dis(-0.1, 0.1);
    w.clear();
    
    //Duyet qua tung phan tu cua vector va gan moi phan tu bang mot giá tri ngau nhien duoc tao ra.
    
    for (int i = 0; i < featureSize; ++i) {
        w.push_back(dis(gen));
    }
}

// ham huan luyen mo hinh SVM.

vector<double> trainSVM(vector<DataPoint>& data, double learningRate = 0.01, int epochs = 10000, double lambda = 0.01) {
    if (data.empty()) return {};

    int featureSize = data[0].features.size(); // kich thuoc cua dac trung.
    vector<double> w(featureSize, 0.0); //khoi tao vector trong so W.
    double b = 0.0;// khoi tao trong so b.

    initializeWeights(w, featureSize); // khoi tao trong so w ngau nhien.
    
    //vong lap huan luyen.
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (auto& point : data) {
            if (point.label * (dotProduct(w, point.features) + b) < 1) {
                for (int i = 0; i < featureSize; ++i) {
                    w[i] -= learningRate * (lambda * w[i] - point.label * point.features[i]); 
                }
                b -= learningRate * (-point.label);
            }
        }
        learningRate *= 0.99; // xao tron du lieu truoc moi epoch.
    }
    w.push_back(b); // them sai so b vao cuoi vector trong so w.
    return w; 
}

void drawGraph(const vector<DataPoint>& data, const vector<double>& w) {
    int gd = DETECT, gm;
    initgraph(&gd, &gm, NULL); 

    for (const auto& point : data) {
        int x = 200 + 30 * point.features[0]; 
        int y = 200 - 30 * point.features[1]; 
        if (point.label == 1)
            setcolor(2); 
        else
            setcolor(4);
        circle(x, y, 5); 
        floodfill(x, y, getcolor());
    }

    if (w.size() >= 3) {
        setcolor(15); 
        for (int x = 0; x < getmaxx(); x++) {
            double x1 = (x - 200) / 30.0;
            double y1 = (-w[2] - w[0] * x1) / w[1];
            int vy = 200 - 30 * y1;
            if (vy >= 0 && vy <= getmaxy()) {
                putpixel(x, vy, WHITE);
            }
        }
    }

    getch(); 
    closegraph(); 
}
vector<DataPoint> readDataFromFile(const string& filename) {
    ifstream inFile(filename);
    vector<DataPoint> data;

    if (!inFile) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1);
    }

    int numDataPoints;
    if (!(inFile >> numDataPoints)) {
        cerr << "Error reading number of data points" << endl;
        exit(1);
    }

    for (int i = 0; i < numDataPoints; ++i) {
        DataPoint dp;
        double x1, x2;
        int label;
        if (!(inFile >> x1 >> x2 >> label)) {
            cerr << "Error reading data point " << i + 1 << endl;
            exit(1);
        }
        dp.features.push_back(x1);
        dp.features.push_back(x2);
        dp.label = label;
        data.push_back(dp);
    }
    inFile.close();
    return data;
}
void giaodien(){ 
    cout << "                        ____________________________________________________________________________\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |                          TRUONG DAI HOC BACH KHOA                          |\n";
	cout << "                       |                          KHOA CONG NGHE THONG TIN                          |\n";                             
    cout << "                       |                                                                            |\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |                          DO AN LAP TRINH TINH TOAN                         |\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |                  QUA TRINH TINH TOAN VA SU DUNG THUAT TOAN                 |\n";
    cout << "                       |                         SUPPORT VECTOR MACHINE(SVM)                        |\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |                            GIAO VIEN HUONG DAN                             |\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |                          PGS.TS. Nguyen Tan Khoi                           |\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |                            SINH VIEN THUC HIEN                             |\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |        1) Phan Phuoc Dat.......................... MSV: 102230339          |\n";
    cout << "                       |        2) Dang Hoang Huy.......................... MSV: 102230349          |\n";
    cout << "                       |                                                                            |\n";
    cout << "                       |                                    Nhom 26                                 |\n";
    cout << "                       |____________________________________________________________________________|\n\n";                      
}
int main() {
	 giaodien();
     string filename ;
     cout<<"nhap tep du diem du lieu can huan luyen : ";
     cin>>filename;
     cout<<"\n";
    vector<DataPoint> data = readDataFromFile(filename);
    vector<double> svmParameters = trainSVM(data);

    cout << "Model parameters: ";
    for (int i = 0; i < svmParameters.size() - 1; ++i) {
        cout << "w" << i << " = " << svmParameters[i] << ", ";
    }
    cout << "b = " << svmParameters.back() << endl;
    cout << "Final SVM Hyperplane: y = ";
    cout << svmParameters[0] << "*x" << 1 ;
    for (int i = 1; i < svmParameters.size() - 1; ++i){
    	if(svmParameters[i]>0)
        cout << " + " <<svmParameters[i]<< "*x" << i+1 ;
        else cout<<" - "<<abs(svmParameters[i])<<"*x"<<i+1;
        
    }
    cout << svmParameters.back() << endl;

    drawGraph(data, svmParameters);

    return 0;
}

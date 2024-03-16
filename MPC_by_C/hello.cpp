#include<iostream>
#include<iomanip>
#include<vector>
#include<math.h>
#include"Eigen/Dense"
using namespace std;
using namespace Eigen;


int main(){
    MatrixXd M(3,3);
    M << 1, 0, 1,
         0, 3, 0,
         0,0,2;
    
    cout << M << endl;

    cout << M.transpose() << endl;
    cout << M << endl;
    return 0;    
}
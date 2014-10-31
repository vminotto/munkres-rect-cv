#include "Munkres.h"
#include <fstream>

using namespace std;

int main(){


	cv::Mat_<float> mat2(5,0);
	cv::Mat_<float> mat(5,4);

	mat(0, 0) = 1;
	mat(0, 1) = 2;
	mat(0, 2) = 0;
	mat(0, 3) = 4;

	mat(1, 0) = 0;
	mat(1, 1) = 0;
	mat(1, 2) = 0;
	mat(1, 3) = 0;

	mat(2, 0) = 5;
	mat(2, 1) = 6;
	mat(2, 2) = 0;
	mat(2, 3) = 6;

	mat(3, 0) = 5;
	mat(3, 1) = std::numeric_limits<float>::infinity();
	mat(3, 2) = std::numeric_limits<float>::infinity();
	mat(3, 3) = std::numeric_limits<float>::infinity();

	mat(4, 0) = 2;
	mat(4, 1) = 3;
	mat(4, 2) = 0;
	mat(4, 3) = 22;

	cout << mat << endl;

	Munkres<float> m;
	m(mat);
	m.p();

	std::cin.get();
	return 0;

}
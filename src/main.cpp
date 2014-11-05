#include "Munkres.h"
#include <random>

using namespace std;

void testMunkres(Munkres<double> &mun, cv::Mat_<double> &mat, bool hugeMat = false){
	
	/*Runs the munkres algorithm and measure the elapsed processing time.*/
	size_t begin = cv::getTickCount();
	mun(mat);
	size_t end = cv::getTickCount();

	if (hugeMat)
		cout << "Input Cost Matrix: " << endl << "A " << mat.rows << " x " << mat.cols << " matrix" << endl;
	else
		cout << "Input Cost Matrix: " << endl << mat << endl << endl;

	cv::Mat_<int> uc(mun.getUnassignedRows(), false);
	cv::Mat_<int> ur(mun.getUnassignedCols(), false);
	cv::Mat_<double> costs(mun.getIndividualCosts(), false);

	cout << "Output assignment: " << mun.getAssignment() << endl;
	cout << "Output cost of assignment: " << mun.getCost() << endl;
	cout << "Output individual costs: " << (costs.empty() ? costs : costs.t()) << endl;
	cout << "Unassigned rows: " << (uc.empty() ? uc : uc.t()) << endl;
	cout << "Unassigned cols: " << (ur.empty() ? ur : ur.t()) << endl;
	cout << "Total elapsed time: " << (end - begin) / cv::getTickFrequency() << endl;
	cout << endl;
}

int main(){

	double inf = std::numeric_limits<double>::infinity();
	
	/*Works for either float or double data.*/
	Munkres<double> mun;

	/*Testing with a rectangular and incomplete assignment, and inf costs.*/
	cv::Mat_<double> mat(5, 4);	
	mat(0, 0) = 3; 		mat(0, 1) = 3;		mat(0, 2) = 1;		mat(0, 3) = 2;	
	mat(1, 0) = inf;	mat(1, 1) = inf;	mat(1, 2) = 6;		mat(1, 3) = 5;
	mat(2, 0) = 3;		mat(2, 1) = 2;		mat(2, 2) = 3;		mat(2, 3) = 4;
	mat(3, 0) = 5;		mat(3, 1) = 2.22;	mat(3, 2) = inf;	mat(3, 3) = inf;
	mat(4, 0) = 2.9;	mat(4, 1) = 3.12;	mat(4, 2) = 5.5;	mat(4, 3) = 2.3;
	testMunkres(mun, mat);
	
	/*Testing with a rectangular overcomplete assignment, with random data.*/
	std::random_device rDev;
	cv::RNG rng(rDev());
	mat.create(4,7);
	rng.fill(mat, cv::RNG::UNIFORM, 0.0, std::nextafter(1.0, 1));
	testMunkres(mun, mat);

	/*Testing a large square assignment, with random data.*/
	mat.create(400, 400);
	rng.fill(mat, cv::RNG::UNIFORM, 0.0, std::nextafter(1.0, 1));
	testMunkres(mun, mat, true);

	cout << "\a\a\a";
	std::cin.get();
	return 0;

}
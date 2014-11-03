#ifndef MUNKRES_H
#define MUNKRES_H

#include <opencv.hpp>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include "CvAuxFuncs.h"


template <class T> class Munkres
{
public:

	static_assert(std::is_floating_point<T>::value, "The Munkres algorithm implemented in 'Munkres.h' only accepts floating point data.");
	using uint8_t = std::uint8_t;

	Munkres(){}
	~Munkres(){}	

	/*Performs the Munkres Algorithm.
	Input: MxN matrix representing the costs of the assignment problem. Data must 
	be floating point. The element 'costMat(y,x)' indicates the cost to assign 'x' to 'y'.
	Return: 1xM integer matrix representing which 'x' was received by each 'y'. In cases
	M > N, a -1 is used to indicate that nothing was assigned to a given 'y'. Conversely,
	in cases where M < N, some 'x' will remain unassigned.*/
	cv::Mat_<int> operator()(cv::Mat_<T> costMat){
		this->costMat = costMat;
		initialize();
		step1();
		step2();
		
		bool done = std::count(starZ.begin(), starZ.end(), -1) == 0;
		int l0 = 0;
		while (!done){
			step3();
			step4();
			done = step5();
			l0++;
		}
		findCost();
		return assignment;
	}

	/*Gets the resulting assignment matrix of the last call to the algorithm.*/
	cv::Mat_<int> getAssignmentMatrix(){ return assignment; }
	/*Gest the total assignment cost of the last found solution.*/
	T getAssignmentCost(){	return cost; }

private:

	/*Output results of the algorithm*/
	cv::Mat_<int> assignment;
	T cost = -1;

	/*Auxiliary variables used for running the algorihtm. They are declared
	here so that data can be easely transfered among all steps (1 trhough 6).*/
	cv::Mat_<T> costMat, dMat, minR, minC;
	cv::Mat_<int> validRow, validCol, starZ, primeZ;
	cv::Mat_<uint8_t> validMat, invalMat, coverRow, coverCol, notCoverRow, notCoverCol;
	std::vector<int> rIdx, cIdx;
	int uZr, uZc, nRows, nCols;

	/*Intermediate functions used to solve the assignment problem. These are
	called internally by the algorith, and should not be altered.*/
	void initialize();
	void step1();
	void step2();
	void step3();
	void step4();
	bool step5();
	void step6();
	void findCost();

	T outerPlus(const cv::Mat_<T> &_mat, cv::Mat_<T> &x, cv::Mat_<T> &y, std::vector<int> &rIdx, std::vector<int> &cIdx);

};

/*Initialiazes all some of the auxiliary data strucutres 
that will be used during the algorithm's execution.*/
template <class T> void Munkres<T>::initialize(){
	
	assignment = cv::Mat_<int>(1, costMat.rows, -1);
	cost = 0;
	
	validMat = costMat < std::numeric_limits<T>::max();
	invalMat = ~validMat;

	/*Set all invalid values (inf and max) to a big value with which we can work
	without causing overflows.*/
	costMat.setTo(T(0), invalMat);
	T sum = std::accumulate(costMat.begin(), costMat.end(), T(0));
	double bigValue = std::pow(T(10), std::ceil(std::log10(sum)) + 1);	
	costMat.setTo(bigValue, invalMat);

	/*Set-up the distance matrix 'dMat'*/
	cv::reduce(validMat, validRow, 1, CV_REDUCE_SUM, cv::DataType<int>::depth);
	cv::reduce(validMat, validCol, 0, CV_REDUCE_SUM, cv::DataType<int>::depth);
	
	nRows = cv::countNonZero(validRow);
	nCols = cv::countNonZero(validCol);

	cv::minMaxIdx(costMat, nullptr, &bigValue, nullptr, nullptr, validMat);
	bigValue *= 10;
	int n = std::max(nRows, nCols);
	dMat = cv::Mat_<T>(n, n, (T)bigValue);
			
	getCloneL(costMat, validRow, validCol).copyTo(dMat(cv::Range(0, nRows), cv::Range(0, nCols)));

}

/*Subtract the row minimum from each row.*/
template <class T> void Munkres<T>::step1(){
	cv::reduce(dMat, minR, 1, CV_REDUCE_MIN);
	cv::reduce(dMat - repeat(minR, 1, dMat.cols), minC, 0, CV_REDUCE_MIN);
}

/*Find a zero of dMat. If there are no starred zeros in its
column or row, start the zero. Repeat for each zero.*/
template <class T> void Munkres<T>::step2(){
	cv::Mat_<T> sum = repeat(minR, 1, dMat.cols) + repeat(minC, dMat.rows, 1);

	cv::Mat_<uint8_t> zP = dMat == sum;

	starZ = cv::Mat_<int>(dMat.rows, 1, -1);
	cv::Point pos;

	while (findFirst(zP, uint8_t(255), pos)){
		starZ(pos.y) = pos.x;
		zP.row(pos.y) = 0;
		zP.col(pos.x) = 0;
	}

}

/*Cover each column with a starred zero. If all the columns are
 covered, then the matching is maximum.*/
template <class T> void Munkres<T>::step3(){
	
	coverCol = cv::Mat_<uint8_t>(1, starZ.rows, uint8_t(0));
	coverRow = cv::Mat_<uint8_t>(starZ.rows, 1, uint8_t(0));

	cv::Mat_<int> mask = getCloneL(starZ, starZ > int(-1), 1);
	assignI(cv::Mat_<uint8_t>(mask.cols, mask.rows, uint8_t(255)), coverCol, 0, mask);

	notCoverCol = ~coverCol;
	notCoverRow = ~coverRow;

	primeZ = cv::Mat_<int>(starZ.rows, 1, int(-1));

	cv::Mat_<T> minRR = getCloneL(minR, notCoverRow, 1);
	cv::Mat_<T> minCC = getCloneL(minC, 1, notCoverCol);

	cv::Mat_<T> sum = cv::repeat(minRR, 1, minCC.cols) + cv::repeat(minCC, minRR.rows, 1);
	cv::Mat_<T> dMatSub = getCloneL(dMat, notCoverRow, notCoverCol);
	getNonZeroInds(sum == dMatSub, rIdx, cIdx);
}

/*Find a noncovered zero and prime it. If there is no starred
zero in the row containing this primed zero, Go to Step 5.
Otherwise, cover this row and uncover the column containing
the starred zero. Continue in this manner until there are no
uncovered zeros left. Save the smallest uncovered value and
Go to Step 6.*/
template <class T> void Munkres<T>::step4(){	

	bool done = false;

	while (!done){

		std::vector<int> cC, cR;
		cR = getNonZeroInds(notCoverRow);
		cC = getNonZeroInds(notCoverCol);
		rIdx = getCloneI(cv::Mat_<int>(cR, false), rIdx, 0);
		cIdx = getCloneI(cv::Mat_<int>(cC, false), cIdx, 0);
		
		bool enterStep6 = true;
		while (!cIdx.empty()){
			uZr = rIdx.front();
			uZc = cIdx.front();
			primeZ(uZr) = uZc;
			int stz = starZ(uZr);
			
			if (stz < 0){
				enterStep6 = false;
				done = true;
				break;
			}

			coverRow(uZr) = 255;
			coverCol(stz) = 0;
			notCoverRow(uZr) = 0;
			notCoverCol(stz) = 255;

			cv::Mat_<uint8_t> z = cv::Mat_<int>(rIdx, false) == uZr;
			rIdx = getCloneL(cv::Mat_<int>(rIdx, false), ~z, 1);
			cIdx = getCloneL(cv::Mat_<int>(cIdx, false), ~z, 1);
			cR = getNonZeroInds(notCoverRow);
			z = getCloneI(dMat, getNonZeroInds(notCoverRow), stz) == (getCloneL(minR, notCoverRow, 1) + minC(stz));

			cv::Mat_<int> cRTemp = getCloneL(cv::Mat_<int>(cR, false), z, 1);
			std::copy(cRTemp.begin(), cRTemp.end(), back_inserter(rIdx));

			cv::Mat_<int> stzTemp(cv::countNonZero(z), 1, stz);
			std::copy(stzTemp.begin(), stzTemp.end(), back_inserter(cIdx));
		}

		if (enterStep6)
			step6();
	}
}

/*Add the minimum uncovered value to every element of each covered
row, and subtract it from every element of each uncovered column.
Return to Step 4 without altering any stars, primes, or covered lines.*/
template <class T> void Munkres<T>::step6(){
	cv::Mat_<T> dMatTmp = getCloneL(dMat, notCoverRow, notCoverCol);
	T minVal = outerPlus(dMatTmp, getCloneL(minR, notCoverRow, 1), getCloneL(minC, 1, notCoverCol), rIdx, cIdx);
	assignL<T>(getCloneL(minC, 1, notCoverCol) + minVal, minC, 1, notCoverCol);
	assignL<T>(getCloneL(minR, coverRow, 1) - minVal, minR, coverRow, 1);
}

/* Construct a series of alternating primed and starred zeros as follows:
Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0 (if any).
Let Z2 denote the primed zero in the row of Z1 (there will always
be one).  Continue until the series terminates at a primed zero
that has no starred zero in its column.  Unstar each starred
zero of the series, star each primed zero of the series, erase
all primes and uncover every line in the matrix. Return to Step 3.*/
template <class T> bool Munkres<T>::step5(){
	cv::Mat_<cv::Point> rowZ1;
	cv::findNonZero(starZ == uZc, rowZ1);
	starZ(uZr) = uZc;
	while (!rowZ1.empty()){		

		uZr = rowZ1(0).y;
		uZc = primeZ(uZr);
		starZ(uZr) = -1;
		cv::findNonZero(starZ == uZc, rowZ1);
		starZ(uZr) = uZc;
	}
	return std::count(starZ.begin(), starZ.end(), -1) == 0;
}

/*After step 1 through 6 have been executed, the final assignment
relationship and assignment cost are computed here.*/
template <class T> void Munkres<T>::findCost(){
	cv::Mat_<int> rowIdx(getNonZeroInds(validRow), true);
	cv::Mat_<int> colIdx(getNonZeroInds(validCol), true);
	colIdx = colIdx.t();

	starZ = starZ(cv::Range(0, nRows), cv::Range::all());
	cv::Mat_<uint8_t> vIdx = starZ <= nCols-1;

	cv::Mat_<int> rowIdxx = getCloneL(rowIdx, vIdx, 1);
	cv::Mat_<int> colIdxx = getCloneI(colIdx, 0, getCloneL(starZ, vIdx, 1));
	assignI(colIdxx, assignment, 0, rowIdxx);
	
	cv::Mat_<uint8_t> mask = assignment > -1;
	cv::Mat_<int> pass = getCloneL(assignment, 1, mask);

	cv::Mat_<uint8_t> tValidMask = getCloneI(validMat, getNonZeroInds(mask), pass);
	
	assignL(cv::Mat_<int>(1, (int)tValidMask.diag().total(), 0), pass, 1, ~(tValidMask.diag()));
	assignL(pass, assignment, 1, mask);

	cv::Mat_<T> finalCostMat = getCloneI(costMat, getNonZeroInds(mask), getCloneL(assignment, 1, mask));
	cost = cv::Scalar_<T>(cv::trace(finalCostMat))[0];
}

/*Auxiliary function*/
template <class T> T Munkres<T>::outerPlus(const cv::Mat_<T> &_mat, cv::Mat_<T> &x, cv::Mat_<T> &y, std::vector<int> &rIdx, std::vector<int> &cIdx){
	cv::Mat_<T> mat = _mat.clone();
	T minVal = std::numeric_limits<T>::max();
	for (int c = 0; c < mat.cols; ++c){
		cv::Mat_<T> col = mat.col(c);
		col -= x + y(0, c);
		minVal = std::min(minVal, *std::min_element(col.begin(), col.end()));
	}
	getNonZeroInds(mat == minVal, rIdx, cIdx);
	return minVal;
}

#endif
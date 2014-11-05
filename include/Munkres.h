#ifndef MUNKRES_H
#define MUNKRES_H

#include <numeric>
#include <algorithm>
#include <cstdint>
#include "CvAuxFuncs.h"


template <class T> class Munkres
{
public:

	static_assert(std::is_floating_point<T>::value, "The Munkres algorithm implemented in 'Munkres.h' only accepts floating point data.");
	using uint8_t = std::uint8_t;

	Munkres() = default;
	~Munkres() = default;

	/*Performs the Munkres Algorithm.
	Input: MxN matrix representing the costs of the assignment problem. Data must 
	be floating point. The element 'costMat(y,x)' indicates the cost to assign 'x' to 'y'.
	Return: 1xM integer matrix representing which 'x' was received by each 'y'. In cases
	M > N, a -1 is used to indicate that nothing was assigned to a given 'y'. Conversely,
	in cases where M < N, some 'x' will remain unassigned.*/
	cv::Mat_<int> operator()(cv::Mat_<T> costMat){
		this->costMat = costMat;
		prepare();
		step1();
		step2();
		bool done = std::count(starZ.begin(), starZ.end(), -1) == 0;
		while (!done){
			step3();
			step4();
			done = step5();
		}
		finish();
		return assignment;
	}

	cv::Mat_<int> getAssignment(){ return assignment; }
	T getCost(){ return cost; }
	std::vector<T> &getIndividualCosts(){ return individualCosts; }
	std::vector<int> &getUnassignedRows(){ return unassignedRows; }
	std::vector<int> &getUnassignedCols(){ return unassignedCols; }


private:

	/*Row vector containing the resulting assignment, that is, the integer of
	value 'assignment(r)' indicates which column was assignment to row 'r'.*/
	cv::Mat_<int> assignment;
	/*This vector holds the individual assignment cost of each correspondence 
	in the 'assignment' vector.*/
	std::vector<T> individualCosts;
	/*Total cost of the resulting assignment after a call to 'operator()'.*/
	T cost = -1;
	/*Remaining rows ans cols that were not included in the 'assignment' matrix. The
	rows vector will only have elements if 'costMat.rows > costMat.cols', and the cols
	one if costMat.cols > costMat.rows*/
	std::vector<int> unassignedRows, unassignedCols;


	/*Auxiliary variables used for running the algorihtm. They are declared
	here so that data can be easely transfered among all steps (1 trhough 6).*/
	cv::Mat_<T> costMat, dMat, minR, minC;
	cv::Mat_<int> validRow, validCol, starZ, primeZ, rIdx, cIdx;
	cv::Mat_<uint8_t> validMat, invalMat, coverRow, coverCol;
	int uZr, uZc, nRows, nCols;

	/*Intermediate functions used to solve the assignment problem. These are
	called internally by the algorithm, and should not be altered.*/
	void prepare();
	void step1();
	void step2();
	void step3();
	void step4();
	bool step5();
	void step6();
	void finish();

	template <class I> T outerPlus(const cv::Mat_<T> &_mat, cv::Mat_<T> &x, cv::Mat_<T> &y, cv::Mat_<I> &rIdx, cv::Mat_<I> &cIdx);

};

/*Initialiazes all some of the auxiliary data strucutres 
that will be used during the algorithm's execution.*/
template <class T> void Munkres<T>::prepare(){
	
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

	unassignedRows.clear();
	individualCosts.resize(costMat.rows, T(-1));
	unassignedCols  .resize(costMat.cols, T(-1));
	std::iota(unassignedCols.begin(), unassignedCols.end(), 0);
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

	cv::Mat_<int> maskI = getCloneL(starZ, starZ > int(-1), 1);
	assignI(cv::Mat_<uint8_t>(maskI.cols, maskI.rows, uint8_t(255)), coverCol, 0, maskI);

	primeZ = cv::Mat_<int>(starZ.rows, 1, int(-1));

	cv::Mat_<T> minRR = getCloneL(minR, coverRow, 1, true, false);
	cv::Mat_<T> minCC = getCloneL(minC, 1, coverCol, false, true);

	cv::Mat_<T> sum = cv::repeat(minRR, 1, minCC.cols) + cv::repeat(minCC, minRR.rows, 1);
	cv::Mat_<T> dMatSub = getCloneL(dMat, coverRow, coverCol, true, true);
	getIndsOfNonZeros2D(cv::Mat_<uint8_t>(sum == dMatSub), rIdx, cIdx);
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

		cv::Mat_<int> cC, cR;
		cR = getIndsOfNonZeros1D(coverRow, true);
		cC = getIndsOfNonZeros1D(coverCol, true);

		rIdx = getCloneI(cR, rIdx, 0);
		cIdx = getCloneI(cC, cIdx, 0);
		
		bool enterStep6 = true;
		while (!cIdx.empty()){
			uZr = rIdx(0);
			uZc = cIdx(0);
			primeZ(uZr) = uZc;
			int stz = starZ(uZr);
	
			if (stz < 0){
				enterStep6 = false;
				done = true;
				break;
			}

			coverRow(uZr) = 255;
			coverCol(stz) = 0;

			cv::Mat_<uint8_t> z = rIdx == uZr;
			rIdx = getCloneL(rIdx, z, 1, true);
			cIdx = getCloneL(cIdx, z, 1, true);
			cR = getIndsOfNonZeros1D(coverRow, true);
			z = getCloneI(dMat, getIndsOfNonZeros1D(coverRow, true), stz) == (getCloneL(minR, coverRow, 1, true, false) + minC(stz));
			
			cv::Mat_<int> cRTemp = getCloneL(cR, z, 1);
			std::copy(cRTemp.begin(), cRTemp.end(), std::back_inserter(rIdx));

			cv::Mat_<int> stzTemp(cv::countNonZero(z), 1, stz);
			std::copy(stzTemp.begin(), stzTemp.end(), std::back_inserter(cIdx));
		}

		if (enterStep6)
			step6();
	}
}

/*Add the minimum uncovered value to every element of each covered
row, and subtract it from every element of each uncovered column.
Return to Step 4 without altering any stars, primes, or covered lines.*/
template <class T> void Munkres<T>::step6(){
	cv::Mat_<T> dMatTmp = getCloneL(dMat, coverRow, coverCol, true, true);
	T minVal = outerPlus(dMatTmp, getCloneL(minR, coverRow, 1, true), getCloneL(minC, 1, coverCol, false, true), rIdx, cIdx);
	assignL<T>(getCloneL(minC, 1, coverCol, false, true) + minVal, minC, 1, coverCol, false, true);
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
template <class T> void Munkres<T>::finish(){
	cv::Mat_<int> rowIdx = getIndsOfNonZeros1D(validRow);
	cv::Mat_<int> colIdx = getIndsOfNonZeros1D(validCol);
	colIdx = colIdx.t();

	starZ = starZ(cv::Range(0, nRows), cv::Range::all());

	cv::Mat_<uint8_t> vIdx = starZ <= nCols - 1;

	cv::Mat_<int> rowIdxx = getCloneL(rowIdx, vIdx, 1);
	cv::Mat_<int> colIdxx = getCloneI(colIdx, 0, getCloneL(starZ, vIdx, 1));

	assignI(colIdxx, assignment, 0, rowIdxx);

	cv::Mat_<uint8_t> mask = assignment > -1;
	cv::Mat_<int> pass = getCloneL(assignment, 1, mask);

	cv::Mat_<uint8_t> tValidMask = getCloneI(validMat, getIndsOfNonZeros1D(mask), pass);
	cv::Mat_<uint8_t> d = tValidMask.diag();
	cv::Mat_<int> src = cv::Mat_<int>(1, (int)d.total(), 0);

	assignL(src, pass, 1, d, false, true);
	assignL(pass, assignment, 1, mask);


	for (int r = 0; r < individualCosts.size(); ++r){
		int c = assignment(r);
		if (c != -1)
			individualCosts[r] = costMat(r, c);
		else
			unassignedRows.push_back(r);
	}
	unassignedCols.erase(std::remove_if(unassignedCols.begin(), unassignedCols.end(), [&](int col){
		return std::find(assignment.begin(), assignment.end(), col) != assignment.end();
	}), unassignedCols.end());

	cv::Mat_<T> finalCostMat = getCloneI(costMat, getIndsOfNonZeros1D(mask), getCloneL(assignment, 1, mask));
	cost = cv::Scalar_<T>(cv::trace(finalCostMat))[0];
}

/*Auxiliary function*/
template <class T> template <class I> 
T Munkres<T>::outerPlus(const cv::Mat_<T> &_mat, cv::Mat_<T> &x, cv::Mat_<T> &y,
	cv::Mat_<I> &rIdx, cv::Mat_<I> &cIdx)
{
	cv::Mat_<T> costMat = _mat.clone();
	T minVal = std::numeric_limits<T>::max();
	for (int c = 0; c < costMat.cols; ++c){
		cv::Mat_<T> col = costMat.col(c);
		col -= x + y(c);
		minVal = std::min(minVal, *std::min_element(col.begin(), col.end()));
	}
	getIndsOfNonZeros2D(cv::Mat_ <uint8_t>(costMat == minVal), rIdx, cIdx);
	return minVal;
}

#endif
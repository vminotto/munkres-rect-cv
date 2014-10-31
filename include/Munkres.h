#ifndef MUNKRES_H
#define MUNKRES_H

#include <opencv.hpp>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include "CvAuxFuncs.h"

using std::uint8_t;

template <class T> class Munkres
{
public:
	Munkres(){}
	~Munkres(){}

	
	void operator()(cv::Mat_<T> costMat){
		this->costMat = costMat;
		prepareDistanceMatrix();
		step1and2();
		step3();
		findCost();
	}

	void p(){

		cout << endl << "Cost Matrix" << endl;
		cout << costMat << endl;

		cout << endl << "Dist Matrix" << endl;
		cout << dMat << endl;
	}

private:

	/*Output results of the algorithm*/
	cv::Mat_<int> assignment;
	T cost = -1;

	/*Auxiliary varibles used for running the algorihtm*/
	cv::Mat_<T> costMat, dMat, minR, minC;
	cv::Mat_<int> validRows, validCols, starZ, primeZ;
	cv::Mat_<uint8_t> validMask, invalMask;

	int uZr = -1, uZc = -1, nRows = -1, nCols = -1;

	/*Removes infinities and prepares a square distance matrix from
	the input matrix (which can be rectangular.*/
	void prepareDistanceMatrix();

	/*Subtract the row minimum from each row.*/
	void step1and2();
	void step3();
	void step5();
	void findCost();

	T outerPlus(const cv::Mat_<T> &_mat, cv::Mat_<T> &x, cv::Mat_<T> &y, std::vector<int> &rIdx, std::vector<int> &cIdx);

};

template <class T> void Munkres<T>::prepareDistanceMatrix(){	
	
	assignment = cv::Mat_<int>(1, costMat.rows, -1);
	cost = 0;
	
	validMask = costMat < std::numeric_limits<T>::max();
	invalMask = ~validMask;

	/*Set all invalid values (inf and max) to a big value with which we can work
	without causing overflows.*/
	costMat.setTo(T(0), invalMask);
	T sum = std::accumulate(costMat.begin(), costMat.end(), T(0));
	double bigValue = std::pow(T(10), std::ceil(std::log10(sum)) + 1);	
	costMat.setTo(bigValue, invalMask);

	/*Set-up the distance matrix 'dMat'*/
	cv::reduce(validMask, validRows, 1, CV_REDUCE_SUM, cv::DataType<int>::depth);
	cv::reduce(validMask, validCols, 0, CV_REDUCE_SUM, cv::DataType<int>::depth);
	
	nRows = cv::countNonZero(validRows);
	nCols = cv::countNonZero(validCols);

	cv::minMaxIdx(costMat, nullptr, &bigValue, nullptr, nullptr, validMask);
	bigValue *= 10;
	int n = std::max(nRows, nCols);
	dMat = cv::Mat_<T>(n, n, (T)bigValue);
			
	getCloneL(costMat, validRows, validCols).copyTo(dMat(cv::Range(0, nRows), cv::Range(0, nCols)));

}

template <class T> void Munkres<T>::step1and2(){

	
	/*Find row minima*/
	cv::reduce(dMat, minR, 1, CV_REDUCE_MIN);
	/*Find column minima of (dMat - minR)*/
	cv::reduce(dMat - repeat(minR, 1, dMat.cols), minC, 0, CV_REDUCE_MIN);

	cv::Mat_<T> sum = repeat(minR, 1, dMat.cols) + repeat(minC, dMat.rows, 1);
	cv::Mat_<uint8_t> zP = dMat == sum;

	starZ = cv::Mat_<int>(dMat.rows, 1, -1);
	cv::Point pos;
	while (findFirst(zP, uint8_t(255), pos)){
		starZ(pos.y, 0) = pos.x;
		zP.row(pos.y) = 0;
		zP.col(pos.x) = 0;
	}
}

template <class T> void Munkres<T>::step3(){
	
	while (true){
		
		if (std::count(starZ.begin(), starZ.end(), -1) == 0)
			break;

		cv::Mat_<uint8_t> coverCol(1, starZ.rows, uint8_t(0));
		cv::Mat_<uint8_t> mask = getCloneL(starZ, starZ >= 0, 1);
		cv::Mat_<uint8_t> trues(mask.cols, mask.rows, uint8_t(255));
		assignI(trues, coverCol, 0, mask);

		cv::Mat_<uint8_t> coverRow(starZ.rows, 1, uint8_t(0));
		primeZ = cv::Mat_<int>(starZ.rows, 1, int(-1));


 		cv::Mat_<T> minRR = getCloneL(minR, ~coverRow, 1);
		cv::Mat_<T> minCC = getCloneL(minC, 1, ~coverCol);

		cv::Mat_<T> sum = cv::repeat(minRR, 1, minCC.cols) + cv::repeat(minCC, minRR.rows, 1);
		cv::Mat_<T> dMatSub = getCloneL(dMat, ~coverRow, ~coverCol);

		std::vector<int> rIdx, cIdx;
		getNonZeroInds(sum == dMatSub, rIdx, cIdx);
		
		int step = -1;
		while (true){		
			std::vector<int> cC, cR;
			cR = getNonZeroInds(~coverRow);
			cC = getNonZeroInds(~coverCol);
			rIdx = getCloneI(cv::Mat_<int>(cR, false), rIdx, 0);
			cIdx = getCloneI(cv::Mat_<int>(cC, false), cIdx, 0);			
			step = 6;
			
			while (!cIdx.empty()){
				uZr = rIdx.front();
				uZc = cIdx.front();
				primeZ(uZr, 0) = uZc;
				cout << primeZ << endl;
				cout << starZ << endl;
				int stz = starZ(uZr);
				if (stz < 0){
					step = 5;
					break;
				}
				coverRow(uZr, 0) = 255;
				coverCol(0, stz) = 0;
				cout << coverRow << endl;
				cout << coverCol << endl;
				cv::Mat_<uint8_t> z = cv::Mat_<int>(rIdx, false) == uZr;
				cout << z << endl;
				rIdx = getCloneL(cv::Mat_<int>(rIdx, false), ~z, 1);
				cIdx = getCloneL(cv::Mat_<int>(cIdx, false), ~z, 1);
				cR = getNonZeroInds(~coverRow);
				z = getCloneI(dMat, getNonZeroInds(~coverRow), stz) == (getCloneL(minR, ~coverRow, 1) + minC(stz));
				cout << z << endl;
				

				cv::Mat_<int> cRTemp = getCloneL(cv::Mat_<int>(cR, false), z, 1);
				std::copy(cRTemp.begin(), cRTemp.end(), back_inserter(rIdx));
				
				cv::Mat_<int> stzTemp(cv::countNonZero(z), 1, stz);
				std::copy(stzTemp.begin(), stzTemp.end(), back_inserter(cIdx));
			}
			if (step == 6){
				cv::Mat_<T> dMatTmp = getCloneL(dMat, ~coverRow, ~coverCol);
				cout << dMatTmp << endl;
				T minVal = outerPlus(dMatTmp, getCloneL(minR, ~coverRow, 1), getCloneL(minC, 1, ~coverCol), rIdx, cIdx);
				assignL<T>(getCloneL(minC, 1, ~coverCol) + minVal, minC, 1, ~coverCol);
				assignL<T>(getCloneL(minR, coverRow, 1) - minVal, minR, coverRow, 1);

				cout << minC << endl;
				cout << minR << endl;
			}
			else{
				break;
			}
		}
		step5();	
	}
}

template <class T> void Munkres<T>::step5(){
	cv::Mat_<cv::Point> rowZ1;
	cv::findNonZero(starZ == uZc, rowZ1);
	starZ(uZr, 0) = uZc;
	while (!rowZ1.empty()){		
		uZr = rowZ1(0).y;
		uZc = primeZ(uZr);
		starZ(uZr) = -1;
		cv::findNonZero(starZ == uZc, rowZ1);
		starZ(uZr) = uZc;
	}
}

template <class T> void Munkres<T>::findCost(){
	cv::Mat_<int> rowIdx(getNonZeroInds(validRows), true);
	cv::Mat_<int> colIdx(getNonZeroInds(validCols), true);
	colIdx = colIdx.t();

	cout << validRows << endl;
	cout << validCols << endl;

	cout << rowIdx << endl;
	cout << colIdx << endl;
	cout << starZ << endl;

	starZ = starZ(cv::Range(0, nRows), cv::Range::all());
	cv::Mat_<uint8_t> vIdx = starZ <= nCols-1;

	cout << vIdx << endl;

	cv::Mat_<int> rowIdxx = getCloneL(rowIdx, vIdx, 1);
	cv::Mat_<int> colIdxx = getCloneI(colIdx, 0, getCloneL(starZ, vIdx, 1));
	assignI(colIdxx, assignment, 0, rowIdxx);
	
	cv::Mat_<uint8_t> mask = assignment > -1;
	cv::Mat_<int> pass = getCloneL(assignment, 1, mask);

	cv::Mat_<uint8_t> tValidMask = getCloneI(validMask, getNonZeroInds(mask), pass);
	cout << tValidMask << endl;
	cout << tValidMask.diag().total();
	assignL(cv::Mat_<int>(1, tValidMask.diag().total(), T(0)), pass, 1, ~(tValidMask.diag()));

	assignL(pass, assignment, 1, mask);

	cout << assignment << endl;
	cout << pass << endl;
	cout << validMask << endl;
	cv::Mat_<T> finalCostMat = getCloneI(costMat, getNonZeroInds(mask), getCloneL(assignment, 1, mask));
	cout << finalCostMat << endl;
	cost = cv::Scalar_<T>(cv::trace(finalCostMat))[0];
}


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
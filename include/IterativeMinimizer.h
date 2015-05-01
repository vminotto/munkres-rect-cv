#ifndef ITERATIVE_MINIMIZER_H
#define ITERATIVE_MINIMIZER_H

#include <vector>
#include <utility>
#include <opencv.hpp>
#include <numeric>
#include <algorithm>

/*This class implements a combinatorial optimization through iterative
minimization and dicard*/
template <class T> class IterativeMinimizer
{
public:

	static_assert(std::is_floating_point<T>::value, "The Munkres algorithm implemented in 'IterativeMinimizer.h' only accepts floating point data.");

	IterativeMinimizer() = default;
	~IterativeMinimizer() = default;
	
	std::vector<int> &operator()(cv::Mat_<T> &_costMat){
		this->set(_costMat);
		if (validMat){
			this->prepare();

			for (int i = 0; i < numIters; i++){
				double minCost;
				int minIdx[2];
				cv::minMaxIdx(costMat, &minCost, nullptr, minIdx, nullptr);
				int bestRow = minIdx[0];
				int bestCol = minIdx[1];

				bestCombination[bestRow] = bestCol;
				individualCosts[bestRow] = costMat(bestRow, bestCol);
				costMat.row(bestRow) = std::numeric_limits<T>::max();
				costMat.col(bestCol) = std::numeric_limits<T>::max();
				cost += T(minCost);
			}
			this->finish();
		}
		return bestCombination;
	}

	/*Gets the resulting assignment matrix of the last call to the algorithm.*/
	cv::Mat_<int> getAssignment(){ return cv::Mat_<int>(bestCombination, false).t(); }
	/*Gest the total assignment cost of the last found solution.*/
	T getCost(){ return cost; }
	/*Gets the individual cost of each index in the 'assignment' vector.*/
	std::vector<T> &getIndividualCosts(){ return individualCosts; }

	std::vector<int> &getUnassignedRows(){ return remainingRows; }
	std::vector<int> &getUnassignedCols(){ return remainingCols; }


private:

	bool validMat = false;
	/*Matrix that is used for computing the combinatorial optimization methods.
	Node this structure needs to be set through the 'set()' method first.*/
	cv::Mat_<T> costMat;
	/*Number of iterations to compute the best distance combinations, 
	that is, numIters = std::min(costMat.rows, costMat.cols) */
	int numIters;
	/*Best combinations returned by the last call to one of the combinatorial
	optimization algorithms that are implemented in this class.*/
	std::vector<int> bestCombination;
	/*Remaining rows that were not included in the 'bestCombination' vector. This 
	vector will only have elements if 'costMat.rows > costMat.cols'*/
	std::vector<int> remainingRows;
	/*Remaining columns that were not included in the 'bestCombination' vector. This
	vector will only have elements if 'costMat.cols > costMat.rows'*/
	std::vector<int> remainingCols;
	/*Total cost of the last outputed assignment result.*/
	T cost;
	std::vector<T> individualCosts;


	void set(cv::Mat_<T> costMat);
	void prepare();
	void finish();

};

template <class T> void IterativeMinimizer<T>::set(cv::Mat_<T> costMat)
{	
	if (!costMat.data || costMat.rows == 0 || costMat.cols == 0){
		std::cerr << "Warning at 'IterativeMinimizer::set()': the matrix passed is invalid."
			" It either has a null dimension or no data on it." << std::endl;
		validMat = false;
		return;
	}
	cost = T(0);
	numIters = std::min(costMat.rows, costMat.cols);
	remainingCols.resize(costMat.cols, -1);
	remainingRows.reserve(costMat.rows);
	bestCombination.resize(costMat.rows, -1);
	individualCosts.resize(costMat.rows, T(-1));
	this->costMat = costMat.clone();
	validMat = true;
}

template <class T> void IterativeMinimizer<T>::prepare(){
	std::iota(remainingCols.begin(), remainingCols.end(), 0);
	remainingRows.clear();
}

template <class T> void IterativeMinimizer<T>::finish(){
	
	/*In case more rows > cols*/
	for (int i = 0; i < bestCombination.size(); ++i){
		if (bestCombination[i] == -1){
			remainingRows.push_back(i);
		}
	}

	remainingCols.erase(std::remove_if(remainingCols.begin(), remainingCols.end(), 
		[&](int i){
		return std::find(bestCombination.begin(), bestCombination.end(), i) != bestCombination.end();
	}
	), remainingCols.end());

	/*remainingRows.erase(std::remove_if(remainingRows.begin(), remainingRows.end(),
		[&](int i)
	{
		return std::find_if(bestCombination.begin(), bestCombination.last(), [&](std::pair<int, int> p)
		{
			return i == p.first;
		}) != bestCombination.end();
	}
	), remainingRows.end());*/

}

#endif
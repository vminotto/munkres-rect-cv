#ifndef CV_AUX_FUNCS_H
#define CV_AUX_FUNCS_H

#include <opencv.hpp>

/*Returns a clone of src based on the indexes passed in through the arrays
'_rowLogicInds' and '_colLogicInds'. If both arrays are all ones, an identical 
clone of 'src' is returned. Columns and rows marked with zeros are not copied.
If the sizes of either vectors exceed the size of src's dimensiosn, an error 
is thrown. This functions simulates matlab's logic indexing.*/
template <class T, class I = unsigned char> cv::Mat_<T> getCloneL(const cv::Mat_<T> &src,
	cv::InputArray _rowLogicInds, cv::InputArray _colLogicInds, bool negateRowInds = false, bool negateColInds = false)
{

	static_assert(std::is_integral<I>::value, "Second template parameter on 'getCloneL(...)' must be an integral primitive type.");

	CV_Assert(src.dims <= 2);

	if (src.empty() || _rowLogicInds.empty() || _colLogicInds.empty())
		return cv::Mat_<T>();

	size_t nRows = cv::countNonZero(_rowLogicInds);
	size_t nCols = cv::countNonZero(_colLogicInds);
	if (negateRowInds) nRows = _rowLogicInds.total() - nRows;
	if (negateColInds) nCols = _colLogicInds.total() - nCols;

	if (nRows == src.rows && nCols == src.cols)
		return src.clone();
	if (_rowLogicInds.total() != src.rows || _colLogicInds.total() != src.cols){
		CV_Error(CV_StsVecLengthErr, std::string("Size of each dimension in the 'src' matrix must match the size of the respective logical indexing vectors."));
	}
	
	cv::Mat_<I> rowLogicInds, colLogicInds;
	rowLogicInds = _rowLogicInds.getMat();
	colLogicInds = _colLogicInds.getMat();

	cv::Mat_<T> dst(nRows, nCols);

	for (int rSrc = 0, rDst = 0; rSrc < src.rows; ++rSrc){
		if (bool(rowLogicInds(rSrc)) != negateRowInds){
			for (int cSrc = 0, cDst = 0; cSrc < src.cols; ++cSrc){
				if (bool(colLogicInds(cSrc)) != negateColInds){
					dst(rDst, cDst) = src(rSrc, cSrc);
					++cDst;
				}
			}
			++rDst;
		}
	}
	return dst;
}

/*Similar as getCloneL, except the values in '_rowInds' and '_colInds' 
represent the actual indexes to be copied. The order of the constructed 
clone is based on the order of appearence of each index in such arrays.
If any of the indexes is out of range, an error is thrown. Using this 
function the resulting matrix may be larger than 'src', which happens
when the same index is specified multiple times.*/
template <class T, class I = int> cv::Mat_<T> getCloneI(const cv::Mat_<T> &src,
	cv::InputArray _rowInds, cv::InputArray _colInds)
{

	static_assert(std::is_integral<I>::value, "Second template parameter on 'getCloneI(...)' must be an integral primitive type.");

	CV_Assert(src.dims <= 2);

	if (src.empty() || _colInds.empty() || _rowInds.empty())
		return cv::Mat_<T>();

	size_t nRows = _rowInds.total();
	size_t nCols = _colInds.total();


	cv::Mat_<I> rowInds, colInds;
	rowInds = _rowInds.getMat();
	colInds = _colInds.getMat();
	
#ifdef _DEBUG
	auto minMax = std::minmax_element(rowInds.begin(), rowInds.end());
	if (*minMax.first < 0 || *minMax.second >= src.rows)
		CV_Error(CV_StsVecLengthErr, std::string("One or more index in '_rowInds' exceed the horizontal dimension of 'src'."));

	minMax = std::minmax_element(colInds.begin(), colInds.end());
	if (*minMax.first < 0 || *minMax.second >= src.cols)
		CV_Error(CV_StsVecLengthErr, std::string("One or more index in '_colInds' exceed the vertical dimension of 'src'."));
#endif

	cv::Mat_<T> dst(nRows, nCols);

	for (int r = 0; r < nRows; ++r){
		for (int c = 0; c < nCols; ++c){
			dst(r, c) = src(rowInds(r), colInds(c));
		}
	}
	return dst;
}


/*Assigns the content of src into dst, using logical indexing for the columns
and rows. For this assignment, 'src' is accessed using the logical indexes
in the arrays. This is similar to the 'getCloneL' function, except it is used
for data insertion instead of retrieval. If the sizes of either vectors 
exceed the size of dst's dimensiosn, an error is thrown. This functions simulates 
the following in matlab:
C++:
--------------------------------------------
cv::Mat_<int> A = cv::Mat_<int>::zeros(3,3); 
std::vector<int> rowLogicInds = {1, 0, 1};
std::vector<int> colLogicInds = {1, 1, 0};
cv::Mat_<int> B(2, 2);
B(0,0) = 1; B(0,1) = 2;
B(1,0) = 3; B(1,1) = 4;
assignL(B, A, rowLogicInds, colLogicInds);

//Output
std::cout << A << std::endl;
//[1, 2, 0;
// 0, 0, 0;
// 3, 4, 0]
--------------------------------------------

MatLab:
--------------------------------------------
A = zeros(3);
rowLogicInds = logical([1 0 1]);
colLogicInds = logical([1 1 0]);
B = [1, 2; 3, 4];
A(rowLogicInds, colLogicInds) = B;

%Output:
disp(A);
%1     2     0
%0     0     0
%3     4     0
--------------------------------------------*/
template <class T, class I = unsigned char> void assignL(const cv::Mat_<T> &src, cv::Mat_<T> &dst,
	cv::InputArray _rowLogicInds, cv::InputArray _colLogicInds, bool negateRowInds = false, bool negateColInds = false)
{

	static_assert(std::is_integral<I>::value, "Second template parameter on 'assignL(...)' must be an integral primitive type.");

	CV_Assert(src.dims <= 2);

	if (src.empty() || _rowLogicInds.empty() || _colLogicInds.empty()){
		return;
	}
	if (_rowLogicInds.total() != dst.rows || _colLogicInds.total() != dst.cols){
		CV_Error(CV_StsVecLengthErr, std::string("Size of each dimension in the 'dst' matrix must match the number of non-zeros in the respective logical indexing arrays."));
	}

	size_t nRows = cv::countNonZero(_rowLogicInds);
	size_t nCols = cv::countNonZero(_colLogicInds);
	if (negateRowInds) nRows = _rowLogicInds.total() - nRows;
	if (negateColInds) nCols = _colLogicInds.total() - nCols;

	if (!nCols || !nRows){
		return;
	}

	if (nCols == dst.cols && nRows == dst.rows){
		dst = src;
		return;
	}
	//if (nRows != src.rows || nCols != src.cols){
	//	CV_Error(CV_StsVecLengthErr, std::string("Size of each dimension in the 'src' matrix must match the size of the respective logical indexing arrays."));
	//}
	
	cv::Mat_<I> rowLogicInds, colLogicInds;
	rowLogicInds = _rowLogicInds.getMat();
	colLogicInds = _colLogicInds.getMat();	

	for (int rDst = 0, rSrc = 0; rDst < dst.rows; ++rDst){
		if (bool(rowLogicInds(rDst)) != negateRowInds){
			for (int cDst = 0, cSrc = 0; cDst < dst.cols; ++cDst){
				if (bool(colLogicInds(cDst)) != negateColInds){
					dst(rDst, cDst) = src(rSrc, cSrc);
					++cSrc;
				}
			}
			++rSrc;
		}
	}
}

/*Same as assignL, except the index arrays specify which positions from 'dst'
should be used when receiving the data from 'src'. This is a common behave in
MatLab which is simulated through the following below:
C++:
--------------------------------------------
cv::Mat_<int> A = cv::Mat_<int>::zeros(3,3);
std::vector<int> rowInds = {2, 0};
std::vector<int> colInds = {2, 1};
cv::Mat_<int> B(2, 2);
B(0,0) = 1; B(0,1) = 2;
B(1,0) = 3; B(1,1) = 4;
assignI(B, A, rowInds, colInds);

//Output
std::cout << A << std::endl;
// [0, 4, 3;
//  0, 0, 0;
//  0, 2, 1]
--------------------------------------------

MatLab:
--------------------------------------------
A = zeros(3);
rowInds = [3 1];
colInds = [3 2];
B = [1, 2; 3, 4];
A(rowInds, colInds) = B;

%Output:
disp(A);
%0     4     3
%0     0     0
%0     2     1
--------------------------------------------*/
template <class T, class I = int> void assignI(const cv::Mat_<T> &src, cv::Mat_<T> &dst,
	cv::InputArray _rowInds, cv::InputArray _colInds)
{

	static_assert(std::is_integral<I>::value, "Second template parameter on 'assignI(...)' must be an integral primitive type.");
	
	CV_Assert(src.dims <= 2);

	if (src.empty() || _rowInds.empty() || _colInds.empty())
		return;

	size_t nRows = _rowInds.total();
	size_t nCols = _colInds.total();
	if (nRows != src.rows || nCols != src.cols){
		CV_Error(CV_StsVecLengthErr, std::string("Size of each dimension in the 'src' matrix must match the size of the respective logical indexing arrays."));
	}

	cv::Mat_<I> rowInds, colInds;
	rowInds = _rowInds.getMat();
	colInds = _colInds.getMat();

#ifdef _DEBUG
	auto minMax = std::minmax_element(rowInds.begin(), rowInds.end());
	if (*minMax.first < 0|| *minMax.second >= dst.rows)
		CV_Error(CV_StsVecLengthErr, std::string("One or more index in '_rowInds' exceed the horizontal dimension of 'src'."));

	minMax = std::minmax_element(colInds.begin(), colInds.end());
	if (*minMax.first < 0 || *minMax.second >= dst.cols)
		CV_Error(CV_StsVecLengthErr, std::string("One or more index in '_colInds' exceed the vertical dimension of 'src'."));
#endif

	for (int r = 0; r < nRows; ++r){
		for (int c = 0; c < nCols; ++c){
			dst(rowInds(r), colInds(c)) = src(r, c);
		}
	}

}

template <class T, class I = int> cv::Mat_<I> getIndsOfNonZeros1D(cv::Mat_<T> &in, bool negateInput = false){
	
	static_assert(std::is_integral<I>::value, "Second template parameter on 'getIndsOfNonZeros1D(...)' must be an integral primitive type.");


	if (in.empty())
		return cv::Mat_<I>();

	if (in.rows > 1 && in.cols > 1){
		CV_Error(CV_StsVecLengthErr, std::string("Input array must be one-dimensional."));
	}
	
	cv::Mat_<I> inds;

	for (int i = 0; i < in.total(); ++i){
		if (bool(in(i)) != negateInput)
			inds.push_back(i);
	}
	return inds;
}

template <class T, class I = int> void getIndsOfNonZeros2D(cv::Mat_<T> in, 
	cv::Mat_<I> &rowInds, cv::Mat_<I> &colInds){

	static_assert(std::is_integral<I>::value, "Second template parameter on 'getIndsOfNonZeros2D(...)' must be an integral primitive type.");

	if (in.empty())
		return;

	if (in.size().width < 1 || in.size().height < 1){
		CV_Error(CV_StsVecLengthErr, std::string("Input array must be two-dimensional."));
	}

	rowInds = cv::Mat_<I>();
	colInds = cv::Mat_<I>();
	for (int r = 0; r < in.rows; ++r){
		for (int c = 0; c < in.cols; ++c){
			if (in(r, c)){
				rowInds.push_back(r);
				colInds.push_back(c);
			}
		}
	}
}

/*Finds the first element in src that compares equal to 'val' and store its
position on 'pos'. The return indicates whether or not the element was found.
This method iterates through the vertical dimension first, corresponding to
a call to the matlab function '[r c] = find(src, 1)'*/
template <class T> bool findFirst(cv::Mat_<T> &src, T val, cv::Point2i &pos){
	bool found = false;
	for (int c = 0; c < src.cols && !found; ++c){
		for (int r = 0; r < src.rows && !found; ++r){
			if (src(r, c) == val){
				found = true;
				pos.x = c;
				pos.y = r;
			}
		}
	}
	return found;
}


template <class T> cv::Mat_<T> &push_rows(cv::Mat_<T> &mat, cv::Mat_<T> &rows){
	if (mat.cols != rows.cols){
		CV_Error(CV_StsUnmatchedSizes, std::string("Input matrices 'mat' and 'rows' must have the same width."));
	}
	if (rows.rows > 0){
		mat.push_back(rows);
	}
	return mat;
}
template <class T> cv::Mat_<T> &push_cols(cv::Mat_<T> &mat, cv::Mat_<T> &cols){
	if (mat.rows != cols.rows){
		CV_Error(CV_StsUnmatchedSizes, std::string("Input matrices 'mat' and 'cols' must have the same height."));
	}
	if (cols.cols > 0){
		mat = mat.t();
		mat.push_back(cols.t());
		mat = mat.t();
	}
	return mat;
}

#endif
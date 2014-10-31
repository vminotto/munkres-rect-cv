#include <opencv.hpp>

/*Returns a clone of src based on the indexes passed in through the arrays
'_rowLogicInds' and '_colLogicInds'. If both arrays are all ones, an identical 
clone of 'src' is returned. Columns and rows marked with zeros are not copied.
If the sizes of either vectors exceed the size of src's dimensiosn, an error 
is thrown. This functions simulates matlab's logic indexing.*/
template <class T> cv::Mat_<T> getCloneL(const cv::Mat_<T> src,
	cv::InputArray _rowLogicInds, cv::InputArray _colLogicInds)
{

	CV_Assert(src.dims <= 2);

	if (src.empty() || _rowLogicInds.empty() || _colLogicInds.empty())
		return src.clone();

	int nRows = cv::countNonZero(_rowLogicInds);
	int nCols = cv::countNonZero(_colLogicInds);

	std::vector<unsigned char> rowLogicInds, colLogicInds;
	_rowLogicInds.getMat().convertTo(rowLogicInds, cv::DataType<unsigned char>::type);
	_colLogicInds.getMat().convertTo(colLogicInds, cv::DataType<unsigned char>::type);

	if (rowLogicInds.size() != src.rows || colLogicInds.size() != src.cols){
		CV_Error(CV_StsVecLengthErr, std::string("Size of each dimension in the 'src' matrix must match the size of the respective logical indexing vectors."));
	}

	cv::Mat_<T> dst(nRows, nCols);

	for (int rSrc = 0, rDst = 0; rSrc < src.rows; ++rSrc){
		if (rowLogicInds[rSrc]){
			for (int cSrc = 0, cDst = 0; cSrc < src.cols; ++cSrc){
				if (colLogicInds[cSrc]){
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
template <class T> cv::Mat_<T> getCloneI(const cv::Mat_<T> src,
	cv::InputArray _rowInds, cv::InputArray _colInds)
{

	CV_Assert(src.dims <= 2);

	if (src.empty() || _rowInds.empty() || _colInds.empty())
		return src.clone();

	std::vector<int> rowInds, colInds;
	_rowInds.getMat().convertTo(rowInds, cv::DataType<int>::type);
	_colInds.getMat().convertTo(colInds, cv::DataType<int>::type);

	int nRows = rowInds.size();
	int nCols = colInds.size();

	auto minMax = std::minmax_element(rowInds.begin(), rowInds.end());
	if (*minMax.first < 0 || *minMax.second >= src.rows)
		CV_Error(CV_StsVecLengthErr, std::string("One or more index in '_rowInds' exceed the horizontal dimension of 'src'."));

	minMax = std::minmax_element(colInds.begin(), colInds.end());
	if (*minMax.first < 0 || *minMax.second >= src.cols)
		CV_Error(CV_StsVecLengthErr, std::string("One or more index in '_colInds' exceed the vertical dimension of 'src'."));

	cv::Mat_<T> dst(nRows, nCols);

	for (int r = 0; r < nRows; ++r){
		for (int c = 0; c < nCols; ++c){
			dst(r, c) = src(rowInds[r], colInds[c]);
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
template <class T> void assignL(const cv::Mat_<T> &src, cv::Mat_<T> &dst,
	cv::InputArray _rowLogicInds, cv::InputArray _colLogicInds)
{

	CV_Assert(src.dims <= 2);

	if (src.empty() || _rowLogicInds.empty() || _colLogicInds.empty())
		return;

	int nRows = cv::countNonZero(_rowLogicInds);
	int nCols = cv::countNonZero(_colLogicInds);

	if (!nCols || !nRows)
		return;

	std::vector<unsigned char> rowLogicInds, colLogicInds;
	_rowLogicInds.getMat().convertTo(rowLogicInds, cv::DataType<unsigned char>::type);
	_colLogicInds.getMat().convertTo(colLogicInds, cv::DataType<unsigned char>::type);

	//if (nRows != src.rows || nCols != src.cols){
	//	CV_Error(CV_StsVecLengthErr, std::string("Size of each dimension in the 'dst' matrix must match the number of non-zero elements on each logical indexing array."));
	//}
	if (rowLogicInds.size() != dst.rows || colLogicInds.size() != dst.cols){
		CV_Error(CV_StsVecLengthErr, std::string("Size of each dimension in the 'src' matrix must match the size of the respective logical indexing arrays."));
	}

	for (int rDst = 0, rSrc = 0; rDst < dst.rows; ++rDst){
		if (rowLogicInds[rDst]){
			for (int cDst = 0, cSrc = 0; cDst < dst.cols; ++cDst){
				if (colLogicInds[cDst]){
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
template <class T> void assignI(const cv::Mat_<T> &src, cv::Mat_<T> &dst,
	cv::InputArray _rowInds, cv::InputArray _colInds)
{

	CV_Assert(src.dims <= 2);

	if (src.empty() || _rowInds.empty() || _colInds.empty())
		return;

	std::vector<int> rowInds, colInds;
	_rowInds.getMat().convertTo(rowInds, cv::DataType<int>::type);
	_colInds.getMat().convertTo(colInds, cv::DataType<int>::type);

	int nRows = rowInds.size();
	int nCols = colInds.size();

	auto minMax = std::minmax_element(rowInds.begin(), rowInds.end());
	if (*minMax.first < 0|| *minMax.second >= dst.rows)
		CV_Error(CV_StsVecLengthErr, std::string("One or more index in '_rowInds' exceed the horizontal dimension of 'src'."));

	minMax = std::minmax_element(colInds.begin(), colInds.end());
	if (*minMax.first < 0 || *minMax.second >= dst.cols)
		CV_Error(CV_StsVecLengthErr, std::string("One or more index in '_colInds' exceed the vertical dimension of 'src'."));

	if (nRows != src.rows || nCols != src.cols){
		CV_Error(CV_StsVecLengthErr, std::string("Size of each dimension in the 'src' matrix must match the size of the respective logical indexing arrays."));
	}

	for (int r = 0; r < nRows; ++r){
		for (int c = 0; c < nCols; ++c){
			dst(rowInds[r], colInds[c]) = src(r, c);
		}
	}
}

std::vector<int> getNonZeroInds(cv::InputArray in){
	
	if (in.empty())
		return std::vector<int>();

	if (in.size().width > 1 && in.size().height > 1){
		CV_Error(CV_StsVecLengthErr, std::string("Input array must be one-dimensional."));
	}

	std::vector<int> logicalInds;
	in.getMat().copyTo(logicalInds);

	std::vector<int> inds;
	for (int i = 0; i < logicalInds.size(); ++i){
		if (logicalInds[i])
			inds.push_back(i);
	}
	return inds;
}

void getNonZeroInds(cv::InputArray in, std::vector<int> &rowInds, std::vector<int> &colInds){

	if (in.empty())
		return;


	if (in.size().width < 1 || in.size().height < 1){
		CV_Error(CV_StsVecLengthErr, std::string("Input array must be two-dimensional."));
	}

	cv::Mat_<int> mat;
	in.getMat().copyTo(mat);

	rowInds.clear();
	colInds.clear();
	for (int r = 0; r < mat.rows; ++r){
		for (int c = 0; c < mat.cols; ++c){
			if (mat(r, c)){
				rowInds.push_back(r);
				colInds.push_back(c);
			}
		}
	}
}

/*Finds the first element in src that compares equal to 'val' and store its
position on 'pos'. The return indicates whether or not the element was found.*/
template <class T> bool findFirst(cv::Mat_<T> &src, T val, cv::Point2i &pos){
	bool found = false;
	for (int r = 0; r < src.rows && !found; ++r){
		for (int c = 0; c < src.cols && !found; ++c){
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
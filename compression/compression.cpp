#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>
#include "cv.h"
using namespace cv;
using namespace std;

//练习1
//鼠标响应函数
std::vector<Point>  mousePoints;
Point points;

void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
	Mat hh;
	hh = *(Mat*)userdata;
	Point p(x, y);
	switch (EVENT)
	{
	case EVENT_LBUTTONDOWN:
	{
		points.x = x;
		points.y = y;
		mousePoints.push_back(points);
		circle(hh, points, 4, cvScalar(255, 255, 255), -1);
		imshow("mouseCallback", hh);
	}
	break;
	}
}

int selectPolygon(cv::Mat srcMat, cv::Mat &dstMat)
{

	vector<vector<Point>> contours;
	cv::Mat selectMat;

	cv::Mat m = cv::Mat::zeros(srcMat.size(), CV_32F);

	m = 1;

	if (!srcMat.empty()) {
		srcMat.copyTo(selectMat);
		srcMat.copyTo(dstMat);
	}
	else {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	namedWindow("mouseCallback");
	imshow("mouseCallback", selectMat);
	setMouseCallback("mouseCallback", on_mouse, &selectMat);
	waitKey(0);
	destroyAllWindows();
	//计算roi
	contours.push_back(mousePoints);
	if (contours[0].size() < 3) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	drawContours(m, contours, 0, Scalar(0), -1);

	m.copyTo(dstMat);

	return 0;
}

int mouseROI()
{
	cv::Mat srcMat = imread("E:\\9\\hogTemplate.jpg");
	cv::Mat dstMat;

	selectPolygon(srcMat, dstMat);

	imshow("srcMat", srcMat);
	imshow("select Area", dstMat);
	waitKey(0);

	return 0;
}

int dftDemo() {

	cv::Mat srcMat = imread("E:\\9\\hogTemplate.jpg", 0);

	if (srcMat.empty()) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	Mat padMat;
	//当图像的尺寸是2，3，5的整数倍时，离散傅里叶变换的计算速度最快。	
	//获得输入图像的最佳变换尺寸
	int m = getOptimalDFTSize(srcMat.rows);
	int n = getOptimalDFTSize(srcMat.cols);
	//对新尺寸的图片进行边缘边缘填充
	copyMakeBorder(srcMat, padMat, 0, m - srcMat.rows, 0, n - srcMat.cols, BORDER_CONSTANT, Scalar::all(0));

	//定义一个数组,存储频域转换成float类型的对象，再存储一个和它一样大小空间的对象来存储复数部分
	Mat planes[] = { Mat_<float>(padMat), Mat::zeros(padMat.size(), CV_32F) };
	Mat complexMat;

	//将2个单通道的图像合成一幅多通道图像
	merge(planes, 2, complexMat);
	//进行傅里叶变换,结果保存在原Mat里,傅里叶变换结果为复数.通道1存的是实部,通道二存的是虚部
	dft(complexMat, complexMat);
	//将双通道的图分离成量个单通道的图 
	//实部：planes[0] = Re(DFT(I),
	//虚部：planes[1]=  Im(DFT(I))) 
	split(complexMat, planes);
	//求相位，保存在planes[0]
	magnitude(planes[0], planes[1], planes[0]);

	//以下步骤均为了显示方便
	Mat magMat = planes[0];
	// log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	magMat += Scalar::all(1);
	log(magMat, magMat);

	//确保对称
	magMat = magMat(Rect(0, 0, magMat.cols & -2, magMat.rows & -2));
	int cx = magMat.cols / 2;
	int cy = magMat.rows / 2;
	//将图像移相
	/*
	0 | 1         3 | 2
	-------  ===> -------
	2 | 3         1 | 0
	*/
	Mat q0(magMat, Rect(0, 0, cx, cy));
	Mat q1(magMat, Rect(cx, 0, cx, cy));
	Mat q2(magMat, Rect(0, cy, cx, cy));
	Mat q3(magMat, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//为了imshow可以显示，归一化到0和1之间
	normalize(magMat, magMat, 0, 1, NORM_MINMAX);
	magMat = magMat * 255;

	imshow("Input Image", srcMat);    // Show the result
	imshow("spectrum magnitude", magMat);
	waitKey(0);

	return 0;

}

int main()
{	
	dftDemo();
	waitKey();
	return 0;
}

#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>
#include "cv.h"
using namespace cv;
using namespace std;

//课前准备

int main()
{
	int bins = 1000000;
	Mat src = imread("E:\\9\\hogTemplate.jpg");
	Mat src1 = imread("E:\\9\\img1.jpg");
	Mat src2 = imread("E:\\9\\img2.jpg");


	if (src.data != NULL)
	{
		vector<int>compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(90);     //图像压缩参数，该参数取值范围为0-100，数值越高，图像质量越高

		bool bRet = imwrite("E:\\OpencvTest\\test2.jpg", src, compression_params);
		if (bRet)
		{
			cout << "图像保存成功" << endl;
		}
		else
		{
			cout << "图像保存失败" << endl;
		}
	}
	else
	{
		cout << "图片加载失败，请检查文件是否存在!" << endl;
	}
	Mat bigImage;
	//放大图像
	resize(src, bigImage, Size(src.cols * 2, src.rows * 2));

	Mat smallImage;
	//缩小图像
	resize(src, smallImage, Size(src.cols / 2, src.rows / 2));

	imshow("原始图像", src);

	imshow("放大图像", bigImage);

	imshow("缩小图像", smallImage);

	waitKey();
	return 0;
}

int round(double c){
    return int(c+0.5*(c<0?-1:1)); 
}
int main()
{
	IplImage *lena=cvLoadImage("lena.bmp");
	IplImage *grayImage=cvCreateImage(cvSize(lena->width,lena->height),lena->depth,1);
	IplImage *resultImage=cvCreateImage(cvSize(lena->width,lena->height),lena->depth,1);
	cvCvtColor(lena,grayImage,CV_BGR2GRAY);
	CvMat *grayMat=cvCreateMat(grayImage->height,grayImage->width,CV_64FC1);
	CvMat *temp=cvCreateMat(grayImage->height,grayImage->width,CV_64FC1);
	CvMat *grayDctMat=cvCreateMat(grayImage->height,grayImage->width,CV_64FC1);
	CvMat *quantizationMat=cvCreateMat(grayImage->height,grayImage->width,CV_64FC1);
	cvScale(grayImage,grayMat);
	//每个像素减去128
	for(int i=0;i<grayMat->rows;i++)
	{
		for (int j=0;j<grayMat->cols;j++)
		{
			double a=cvmGet(grayMat,i,j);
			double b=a-128;
			cvmSet(temp,i,j,b);
		}
	}
	//DCT变换
	cvDCT(temp,grayDctMat,CV_DXT_FORWARD);
	/*
	for(int i=0;i<8;i++)
	{
		for (int j=0;j<8;j++)
		{
			cout<<grayMat->data.db[i*grayMat->rows+j]<<" ";
		}
		cout<<endl;
	}
	cout<<"---------------------------------------------"<<endl;
	*/
	//量化矩阵
	double quantizationData[]={16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,
		103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99};
	CvMat quantizationMatrix=cvMat(8,8,CV_64FC1,quantizationData);
	//量化过程
	for (int i=0;i<32;i++){
		for (int j=0;j<32;j++){
			for(int m=0;m<8;m++){
				for (int n=0;n<8;n++)
				{
					//量化
					double a=cvGetReal2D(grayDctMat,i*8+m,j*8+n);
					double b=cvGetReal2D(&quantizationMatrix,m,n);
					int c=round(a/b);
					//反量化
					int d=int(b*c);
					cvmSet(quantizationMat,i*8+m,j*8+n,d);
				}
			}
		}
	}
	//逆DCT变换
	cvDCT(quantizationMat,temp,CV_DXT_INVERSE);
	for(int i=0;i<temp->rows;i++)
	{
		for (int j=0;j<temp->cols;j++)
		{
			double a=cvmGet(temp,i,j);
			double b=round(a)+128;
			cvmSet(temp,i,j,b);
		}
	}
	cvScale(grayMat,resultImage);
	//计算均方误差
	double MSE=0;
	double square=0;
	double PSNR=0;
	for(int i=0;i<grayMat->rows;i++)
	{
		for (int j=0;j<grayMat->cols;j++)
		{
			double a=cvmGet(grayMat,i,j);
			double b=cvmGet(temp,i,j);;
			double c=a-b;
		//	cout<<c<<endl;
			square+=c*c;
		}
	}
	square/=256;
	square/=256;
	MSE=sqrt(square);
	PSNR=10*log10(255*255/MSE);
//	cout<<"square:"<<square<<endl;
	cout<<"MSE:"<<MSE<<endl;
	cout<<"PSNR:"<<PSNR<<endl;
	/*
	for(int i=0;i<8;i++)
	{
		for (int j=0;j<8;j++)
		{
			cout<<grayMat->data.db[i*grayMat->rows+j]<<" ";
		}
		cout<<endl;
	}*/
	/*
	for(int i=0;i<quantizationMatrix.rows;i++)
	{
		for (int j=0;j<quantizationMatrix.cols;j++)
		{
			cout<<quantizationMatrix.data.db[i*quantizationMatrix.rows+j]<<" ";
		}
		cout<<endl;
	}*/
	/*
	for (int i=0;i<20;i++)
	{
		for (int j=0;j<8;j++)
		{
			cout<<grayMat->data.db[i*grayMat->rows+j]<<" ";
		}
		cout<<endl;
		for (int j=0;j<8;j++)
		{
			cout<<grayDctMat->data.db[i*grayDctMat->rows+j]<<" ";
		}
		cout<<endl;
	}*/
 
	cvNamedWindow("gray");
	cvShowImage("gray",grayImage);
	cvNamedWindow("jpeg");
	cvShowImage("jpeg",resultImage);
	cvWaitKey(0);
	cvDestroyWindow("gray");
	cvDestroyWindow("jpeg");
	cvReleaseImage(&grayImage);
	cvReleaseImage(&lena);
	getchar();
	return 0;

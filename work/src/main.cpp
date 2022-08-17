
// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// project
#include "invert.hpp"


using namespace cv;
using namespace std;



Mat cgraHsvImage(const Mat& m) {
	// NOTE: this implemnetation is only suitable
	// for a 3-channel ubyte-format image

	Mat r = m.clone();
	Mat g = m.clone();
	Mat b = m.clone();
	Mat HSVImage = m.clone();
	Mat h = HSVImage.clone();
	Mat s = HSVImage.clone();
	Mat v = HSVImage.clone();
	Mat bgrconcate;
	Mat hsvconcate;
	cvtColor(m.clone(),HSVImage,40);

	//BGR
	// manually iterate over all pixels
	for (int i = 0; i < b.rows; ++i) { // rows
		for (int j = 0; j < b.cols; ++j) { // cols
			Vec3b bgr = b.at<Vec3b>(i, j);
			b.at<Vec3b>(i, j) = Vec3b(bgr[0], bgr[0], bgr[0]);
			g.at<Vec3b>(i, j) = Vec3b(bgr[1], bgr[1], bgr[1]);
			r.at<Vec3b>(i, j) = Vec3b(bgr[2], bgr[2], bgr[2]);
		}
	}


	//HSV
	// manually iterate over all pixels
	for (int i = 0; i < h.rows; ++i) { // rows
		for (int j = 0; j < h.cols; ++j) { // cols
			Vec3b hsv = HSVImage.at<Vec3b>(i, j);
			h.at<Vec3b>(i, j) = Vec3b(hsv[0], hsv[0], hsv[0]);
			s.at<Vec3b>(i, j) = Vec3b(hsv[1], hsv[1], hsv[1]);
			v.at<Vec3b>(i, j) = Vec3b(hsv[2], hsv[2], hsv[2]);
		}
	}


	//Combines all images together
	Mat finalImage;
	hconcat(b, g, bgrconcate);
	hconcat(bgrconcate, r, bgrconcate);
	hconcat(h, s, hsvconcate);
	hconcat(hsvconcate, v, hsvconcate);
	vconcat(bgrconcate,hsvconcate,finalImage);

	return finalImage;
}


Mat cgraHsvMultiplyImage(const Mat& m) {
	// NOTE: this implemnetation is only suitable
	// for a 3-channel ubyte-format image

	
	Mat HSVImage = m.clone();
	Mat h = HSVImage.clone();
	Mat s = HSVImage.clone();
	Mat v = HSVImage.clone();
	
	cvtColor(m.clone(), HSVImage, 40);

	Mat finalH;
	
	for (float iteration = 0.0; iteration < 1; iteration += 0.2) {
		// manually iterate over all pixels
		Mat hsvCombine;
		for (int i = 0; i < HSVImage.rows; ++i) { // rows
			for (int j = 0; j < HSVImage.cols; ++j) { // cols
				Vec3b hsv = HSVImage.at<Vec3b>(i, j);
				h.at<Vec3b>(i, j) = Vec3b(hsv[0]*iteration,hsv[1],hsv[2]);
				s.at<Vec3b>(i, j) = Vec3b(hsv[0], hsv[1] * iteration, hsv[2]);
				v.at<Vec3b>(i, j) = Vec3b(hsv[0], hsv[1], hsv[2] * iteration);

				Mat temp;

				vconcat(h, s, temp);
				vconcat(temp, v, hsvCombine);
				
			}
		
		}
		if (iteration == 0) {
			finalH = hsvCombine;
		}
		else {
			hconcat(finalH, hsvCombine, finalH);
		}
		
	}
	
	cvtColor(finalH, finalH, COLOR_HSV2BGR);

	return finalH;
}




Mat cgraMaskImage(const Mat& m) {
	Mat mask = m.clone();
	Vec3b maskPoint = mask.at<Vec3b>(80, 80);

// manually iterate over all pixels
	for (int i = 0; i < mask.rows; ++i) { // rows
		for (int j = 0; j < mask.cols; ++j) { // cols
			Vec3b bgr = mask.at<Vec3b>(i, j);
			float dis = sqrt((bgr[0] - maskPoint[0])* (bgr[0] - maskPoint[0]) + (bgr[1] - maskPoint[1]) * (bgr[1] - maskPoint[1]) + (bgr[2] - maskPoint[2]) * (bgr[2] - maskPoint[2]));
			if (dis < 100) {
				mask.at<Vec3b>(i, j) = Vec3b(255,255,255);
			}
			else {
				mask.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}
	}

	return mask;
}


Mat cgraConvolutionImage(const Mat& m, const Mat& k) {
	Mat matte (m.cols, m.rows, CV_32FC1);
	Mat mfloat;
	m.convertTo(mfloat, CV_32FC1);


	filter2D(mfloat,matte,-1,k);
	double max, min;
	minMaxIdx(matte,&min,&max);
	//Mat final(matte.cols, matte.rows, CV_32FC1);
	float mScale = 0.5/-min;
	float pScale = 0.5 / max;
	//Manually Scales points 
	for (int i = 0; i < matte.rows; ++i) { // rows
		for (int j = 0; j < matte.cols; ++j) { // cols
			float f = matte.at<float>(i, j);
			float p;
			if (f < 0) {
				p = f * mScale+0.5;
			}
			else {
				p = f * pScale+0.5;
			}
			matte.at<float>(i, j) = p;
		}
	}
	return matte;
}



Mat cgraEqualizedImage(const Mat& m) {
	Mat hist;
	float range[] = { 0,256 };
	const float* pRange[] = {range};
	const int size = 256;
	Mat finalImage = m.clone();

	calcHist(&m,1,0,Mat(), hist, 1, &size, pRange, true, false);

	for (int i = 1; i < 256; i++) {
	//	cout <<"First" << hist.at<float>(i)<<" ";
		hist.at<float>(i) += hist.at<float>(i - 1);
	//	cout <<"second" << hist.at<float>(i)<<"\n";
	}
	float maxV = hist.at < float >(255);
	for (int i = 0; i < finalImage.rows; ++i) { // rows
		for (int j = 0; j < finalImage.cols; ++j) { // cols
		finalImage.at<unsigned char>(i, j) = (unsigned char)(255 * hist.at<float>(m.at<unsigned char>(i, j)) / maxV);
	}
	}


	return finalImage;
}






// main program
// 
int main( int argc, char** argv ) {

	// check we have exactly one additional argument
	// eg. res/vgc-logo.png
	if( argc != 2) {
		cerr << "Usage: cgra352 <Image>" << endl;
		abort();
	}


	// read the file
	Mat image;
	image = imread(argv[1], 1); 
	Mat greyImage = imread(argv[1], 0);

	// check for invalid input
	if(!image.data ) {
		cerr << "Could not open or find the image" << std::endl;
		abort();
	}

	Mat laplacian = (Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
	Mat sobelX = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat sobelY = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	//Reference used to find the Matrix for sobel X and sobel Y 
	//https://en.wikipedia.org/wiki/Sobel_operator


	Mat l = cgraConvolutionImage(greyImage, laplacian);
	Mat sx = cgraConvolutionImage(greyImage, sobelX);
	Mat sy = cgraConvolutionImage(greyImage, sobelY);

	// use our function to invert it
	//
	
	Mat inverted = cgraInvertImage(image);
	Mat hsv = cgraHsvImage(image);
	Mat hsvMultiply = cgraHsvMultiplyImage(image);
	Mat mask = cgraMaskImage(image);
	Mat equalized = cgraEqualizedImage(greyImage);

	// save image
	imwrite("output/image.png", image);
	imwrite("output/inverted.png", image);


	// create a window for display and show our image inside it
	string img_display = "Image Display";
	namedWindow(img_display, WINDOW_AUTOSIZE);
	imshow(img_display, image);

	string inv_img_display = "Inverted Image Display";
	namedWindow(inv_img_display, WINDOW_AUTOSIZE);
	imshow(inv_img_display, inverted);

	string hsv_img_display = "Hsv Image Display";
	namedWindow(hsv_img_display, WINDOW_AUTOSIZE);
	imshow(hsv_img_display, hsv);

	string hsvMultiply_img_display = "Hsv Multiply Image Display";
	namedWindow(hsvMultiply_img_display, WINDOW_AUTOSIZE);
	imshow(hsvMultiply_img_display, hsvMultiply);

	string mask_img_display = "Mask";
	namedWindow(mask_img_display, WINDOW_AUTOSIZE);
	imshow(mask_img_display, mask);

	string laplacian_img_display = "Laplacian Display";
	namedWindow(laplacian_img_display, WINDOW_AUTOSIZE);
	imshow(laplacian_img_display, l);

	string sobelX_img_display = "Sobel X Display";
	namedWindow(sobelX_img_display, WINDOW_AUTOSIZE);
	imshow(sobelX_img_display, sx);

	string sobelY_img_display = "Sobel Y Display";
	namedWindow(sobelY_img_display, WINDOW_AUTOSIZE);
	imshow(sobelY_img_display, sy);

	string Equalized_img_display = "Equalized Display";
	namedWindow(Equalized_img_display, WINDOW_AUTOSIZE);
	imshow(Equalized_img_display, equalized);

	// wait for a keystroke in the window before exiting
	waitKey(0);
}

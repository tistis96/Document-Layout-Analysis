#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include <list>

using namespace cv;
using namespace std;

struct ConnectedComponent {
	int inc = 0;
	int minDst = INT_MAX;
	float area;
	float density;
	float gamma;
	int minX = INT_MAX;
	int maxX = INT_MIN;
	int minY = INT_MAX;
	int maxY = INT_MIN;
	int height;
	int width;
	vector<std::pair<int, int>> pixelCoordinates;
	vector<int> labelValues;
	vector<int> BNNs;
	vector<int> BNNFor;
	bool hasTextRegion;

	void ComputeParameters()
	{
		area = (maxX - minX) * (maxY - minY);
		height = maxY - minY;
		width = maxX - minX;
		//gamma = min(height, width) / max(height, width);
	}

	void Encapsulate(int x, int y)
	{
		if (x < minX)
		{
			minX = x;
		}
		if (x > maxX)
		{
			maxX = x;
		}
		if (y < minY)
		{
			minY = y;
		}
		if (y > maxY)
		{
			maxY = y;
		}
		pixelCoordinates.push_back(std::pair<int, int>(x, y));
	}

	void EncapsulateDst(float minX, float maxX)
	{
		float dst;
		/*
		if (this->maxX <= minX)
		{
			dst = minX - this->maxX;
		}
		else
		{
			dst = this->minX - maxX;
		}
		*/
		dst = abs(this->minX - maxX);
		if (dst < minDst)
		{
			minDst = dst;
		}
	}

	void DisplayParameters()
	{
		cout << "labelValue = " << labelValues[0] << ", inc = " << inc << ", area = " << area << ", density = " << density << ", gamma = " << gamma << ", minX = " << minX << ", maxX = " << maxX << ", minY = " << minY << ", maxY = " << maxY << ", height = " << height << ", width = " << width << endl;
	}
};

struct TextRegion:ConnectedComponent {
	vector<int> textLinesIndices;

	TextRegion() {}

	TextRegion(const ConnectedComponent& cc)
	{
		this->inc = cc.inc;
		this->minDst = cc.minDst;
		this->area = cc.area;
		this->density = cc.density;
		this->gamma = cc.gamma;
		this->minX = cc.minX;
		this->maxX = cc.maxX;
		this->minY = cc.minY;
		this->maxY = cc.maxY;
		this->height = cc.height;
		this->width = cc.width;
		this->pixelCoordinates = cc.pixelCoordinates;
		this->labelValues = cc.labelValues;

		//not used
		//this->BNNs = cc.BNNs;
		//this->BNNFor = cc.BNNFor;
		//this->hasTextRegion = cc.hasTextRegion;
	}
};

struct less_than_key
{
	inline bool operator() (const ConnectedComponent& struct1, const ConnectedComponent& struct2)
	{
		return (struct1.minY < struct2.minY);
	}
};

bool TextLineCheck(ConnectedComponent& c1, ConnectedComponent& c2, float omega)
{
	if (max(c1.minY, c2.minY) - min(c1.maxY, c2.maxY) < 0)
	{
		//if ((c1.maxX <= c2.minX && abs(c2.minX - c1.maxX) <= 1.3f * max(c1.height, c2.height)) || (c1.minX > c2.maxX && abs(c1.minX - c2.maxX) <= 1.3f * max(c1.height, c2.height)))
		if ((abs(c1.minX - c2.maxX) <= omega * max(c1.height, c2.height)) /*|| (c1.maxX > c2.maxX && c1.minX < c2.minX)*/)
		{
			if (max(c1.height, c2.height) <= 2 * min(c1.height, c2.height))
			{
				return true;
			}
		}
	}

	return false;
}

bool OverlapCheck(ConnectedComponent& c1, ConnectedComponent& c2)
{
	//check if one component is fully enclosed in the other one
	if ((c1.maxX >= c2.maxX && c1.minX <= c2.minX) && (c1.maxY >= c2.maxY && c1.minY <= c2.minY)) 
	{
		return true;
	}

	return false;
}

bool HorizontalOverlap(ConnectedComponent& c1, ConnectedComponent& c2)
{
	// If one rectangle is on left side of other 
	if (c1.minX > c2.maxX || c2.minX > c1.maxX)
	{
		return false;
	}

	return true;
}

bool VerticalOverlap(ConnectedComponent& c1, ConnectedComponent& c2)
{
	// If one rectangle is above other
	if (c1.minY > c2.maxY || c2.minY > c1.maxY)
	{
		return false;
	}

	return true;
}

bool ProximityCheck(ConnectedComponent& c1, ConnectedComponent& c2, float omega)
{
	if ((abs(c1.minX - c2.maxX) <= omega * max(c1.height, c2.height)))
	{
		return true;
	}

	return false;
}

void GetAndSetBNNs(int cIndex, vector<ConnectedComponent>& connected_components)
{
	bool stop = false;
	for (int ci = (cIndex + 1); ci < connected_components.size(); ++ci)
	{
		if ((abs(connected_components[ci].minY - connected_components[cIndex].maxY) <= 5.0f || abs(connected_components[cIndex].minY - connected_components[ci].maxY) <= 5.0f) && HorizontalOverlap(connected_components[cIndex], connected_components[ci])) // T first encounters a text line A and A overlaps horizontally with T
		{
			connected_components[cIndex].BNNs.push_back(ci);
			connected_components[ci].BNNFor.push_back(cIndex);
			// A is a BNN with T (first text line in the BNN chain)
			// find the rest of the BNN (the rest of the text lines in order to form the entire BNN chain)
			for (int cj = (ci + 1); cj < connected_components.size(); ++cj)
			{
				// Subsequent text lines need first to overlap horizontally with T
				if ((abs(connected_components[cj].minY - connected_components[cIndex].maxY) <= 5.0f || abs(connected_components[cIndex].minY - connected_components[cj].maxY) <= 5.0f) && HorizontalOverlap(connected_components[cIndex], connected_components[cj]))
				{
					// Subsequent text lines also need also to overlap vertically with A to be a BNN of T
					if (VerticalOverlap(connected_components[ci], connected_components[cj]))
					{
						connected_components[cIndex].BNNs.push_back(cj);
						connected_components[cj].BNNFor.push_back(cIndex);
					}
					else
					{
						break;
					}
				}
			}

			break;
		}
	}
}

int main()
{
	string filename = "images/page.png";
	Mat image = imread(filename, IMREAD_GRAYSCALE);
	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//Display original image
	namedWindow("Display original image", WINDOW_AUTOSIZE);
	imshow("Display original image", image);
	cout << "Press any key to continue... \n";
	waitKey(0);

	/*
	//Blur the image with 3x3 Gaussian kernel
	Mat image_blurred_with_3x3_kernel;
	GaussianBlur(image, image_blurred_with_3x3_kernel, Size(5, 5), 0);

	//Display blured image
	namedWindow("Display blured image", WINDOW_NORMAL);
	imshow("Display blured image", image_blurred_with_3x3_kernel);
	cout << "Press any key to continue... \n";
	waitKey(0);
	*/

	Mat otsu_binarized_image;
	threshold(image, otsu_binarized_image, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	//Display binarized image
	namedWindow("Display binarized image", WINDOW_AUTOSIZE);
	imshow("Display binarized image", otsu_binarized_image);
	cout << "Press any key to continue... \n";
	waitKey(0);

	Mat labelImage(otsu_binarized_image.size(), CV_32S);
	int nLabels = connectedComponents(otsu_binarized_image, labelImage, 8);
	std::vector<ConnectedComponent> connected_components(nLabels);
	std::vector<Vec3b> colors(nLabels);
	colors[0] = Vec3b(0, 0, 0); //background
	connected_components[0].labelValues.push_back(0);
	for (int label = 1; label < nLabels; ++label)
	{
		colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
		connected_components[label].labelValues.push_back(label);
	}
	Mat dst(otsu_binarized_image.size(), CV_8UC3);
	for (int r = 0; r < dst.rows; ++r) {
		for (int c = 0; c < dst.cols; ++c) {
			int label = labelImage.at<int>(r, c);
			Vec3b& pixel = dst.at<Vec3b>(r, c);
			pixel = colors[label];

			connected_components[label].inc++;
			connected_components[label].Encapsulate(c, r);
		}
	}
	for (int label = 0; label < nLabels; ++label)
	{
		connected_components[label].ComputeParameters();
		//connected_components[label].DisplayParameters();
	}

	//Display connected components
	namedWindow("Connected Components", WINDOW_AUTOSIZE);
	imshow("Connected Components", dst);
	cout << "Press any key to continue... \n";
	waitKey(0);

	//Merge connected components to extract text lines
	bool repeat;
	int iteration = 0;
	do {
		//cout << "Iteration " << iteration << endl;
		repeat = false;
		int ci = 0, cj = 0;
		for (ci = 1; ci < connected_components.size(); ++ci)
		{
			for (cj = 1; (cj < connected_components.size()) && cj != ci; ++cj)
			{
				if (TextLineCheck(connected_components[ci], connected_components[cj], 1.35f) || TextLineCheck(connected_components[cj], connected_components[ci], 1.35f))
				{
					//merge connected components into one
					for (int p = 0; p < connected_components[cj].pixelCoordinates.size(); ++p)
					{
						connected_components[ci].Encapsulate(connected_components[cj].pixelCoordinates[p].first, connected_components[cj].pixelCoordinates[p].second);
					}
					connected_components[ci].EncapsulateDst(connected_components[cj].minX, connected_components[cj].maxX);
					for (int l = 0; l < connected_components[cj].labelValues.size(); ++l)
					{
						connected_components[ci].labelValues.push_back(connected_components[cj].labelValues[l]);
					}
					connected_components.erase(connected_components.begin() + cj);

					repeat = true;
					break;
				}
			}
			if (repeat)
			{
				break;
			}
		}
		if (repeat)
		{
			iteration++;
		}
	} while (repeat);

	cout << "Total text lines = " << connected_components.size() << endl;


	Mat text_lines(otsu_binarized_image.size(), CV_8UC3);
	for (int c = 0; c < connected_components.size(); ++c)
	{
		int label = connected_components[c].labelValues[0];
		for (int p = 0; p < connected_components[c].pixelCoordinates.size(); ++p)
		{
			Vec3b& pixel = text_lines.at<Vec3b>(connected_components[c].pixelCoordinates[p].second, connected_components[c].pixelCoordinates[p].first);
			pixel = colors[label];
		}

		// and its top left corner...
		cv::Point pt1(connected_components[c].minX, connected_components[c].minY);
		// and its bottom right corner.
		cv::Point pt2(connected_components[c].maxX, connected_components[c].maxY);
		cv::rectangle(text_lines, pt1, pt2, cv::Scalar(0, 0, 255));
	}

	//Display connected components
	namedWindow("Connected Components Lines Extraction", WINDOW_AUTOSIZE);
	imshow("Connected Components Lines Extraction", text_lines);
	cout << "Press any key to continue... \n";
	waitKey(0);

	for (int c = 0; c < connected_components.size(); ++c)
	{
		connected_components[c].ComputeParameters();
	}

	//Refine the text lines
	do {
		//cout << "Iteration " << iteration << endl;
		repeat = false;
		int ci = 0, cj = 0;
		for (ci = 1; ci < connected_components.size(); ++ci)
		{
			for (cj = 1; cj < connected_components.size(); ++cj)
			{
				if (cj == ci)
				{
					continue;
				}
				if ((HorizontalOverlap(connected_components[ci], connected_components[cj]) && VerticalOverlap(connected_components[cj], connected_components[ci]))
					|| (VerticalOverlap(connected_components[ci], connected_components[cj]) && (ProximityCheck(connected_components[ci], connected_components[cj], 0.75f) || ProximityCheck(connected_components[cj], connected_components[ci], 0.75f))))
				{
					//merge connected components into one
					for (int p = 0; p < connected_components[cj].pixelCoordinates.size(); ++p)
					{
						connected_components[ci].Encapsulate(connected_components[cj].pixelCoordinates[p].first, connected_components[cj].pixelCoordinates[p].second);
					}
					for (int l = 0; l < connected_components[cj].labelValues.size(); ++l)
					{
						connected_components[ci].labelValues.push_back(connected_components[cj].labelValues[l]);
					}
					connected_components.erase(connected_components.begin() + cj);

					repeat = true;
					break;
				}
			}
			if (repeat)
			{
				break;
			}
		}
		if (repeat)
		{
			iteration++;
		}
	} while (repeat);

	{
		Mat text_lines(otsu_binarized_image.size(), CV_8UC3);
		for (int c = 0; c < connected_components.size(); ++c)
		{
			int label = connected_components[c].labelValues[0];
			for (int p = 0; p < connected_components[c].pixelCoordinates.size(); ++p)
			{
				Vec3b& pixel = text_lines.at<Vec3b>(connected_components[c].pixelCoordinates[p].second, connected_components[c].pixelCoordinates[p].first);
				pixel = colors[label];
			}

			// and its top left corner...
			cv::Point pt1(connected_components[c].minX, connected_components[c].minY);
			// and its bottom right corner.
			cv::Point pt2(connected_components[c].maxX, connected_components[c].maxY);
			cv::rectangle(text_lines, pt1, pt2, cv::Scalar(0, 0, 255));
		}

		//Display connected components
		namedWindow("Connected Components Lines Extraction", WINDOW_AUTOSIZE);
		imshow("Connected Components Lines Extraction", text_lines);
		cout << "Press any key to continue... \n";
		waitKey(0);
	}

	//Group text lines into text blocks
	//First sort text lines in ascending order of ymin
	std::sort(connected_components.begin(), connected_components.end(), less_than_key());

	//Compute the BNN relations between text lines
	for (int c = 0; c < connected_components.size(); ++c)
	{
		//connected_components[c].DisplayParameters();
		GetAndSetBNNs(c, connected_components);
	}
	
	/*
	for (int c = 0; c < connected_components.size(); ++c)
	{
		connected_components[c].DisplayParameters();
		cout << connected_components[c].BNNs.size() << " - " << connected_components[c].BNNFor.size() << endl;
	}
	waitKey(0);
	*/
	
	/*
	//Merge components of text lines in a single line
	do {
		repeat = false;
		for (int c = 0; c < connected_components.size(); ++c)
		{
			if (connected_components[c].BNNs.size() > 1)
			{
				bool hasSthBelow = false;
				for (int b = 0; b < connected_components[c].BNNs.size(); ++b)
				{
					if (connected_components[connected_components[c].BNNs[b]].BNNs.size() > 0)
					{
						hasSthBelow = true;
						break;
					}
				}
				if (!hasSthBelow)
				{
					//merge the BNNs into a single line (merge all the others to the first BNN)
					for (int b = (connected_components[c].BNNs.size() - 1); b >= 1; --b)
					{
						ConnectedComponent nextBNN = connected_components[connected_components[c].BNNs[b]];
						for (int p = 0; p < nextBNN.pixelCoordinates.size(); ++p)
						{
							connected_components[connected_components[c].BNNs[0]].Encapsulate(nextBNN.pixelCoordinates[p].first, nextBNN.pixelCoordinates[p].second);
						}
						for (int l = 0; l < nextBNN.labelValues.size(); ++l)
						{
							connected_components[connected_components[c].BNNs[0]].labelValues.push_back(nextBNN.labelValues[l]);
						}
						connected_components.erase(connected_components.begin() + connected_components[c].BNNs[b]);
					}
					repeat = true;
					break;
				}
			}
		}

		for (int c = 0; c < connected_components.size(); ++c)
		{
			connected_components[c].BNNs.clear();
			connected_components[c].BNNFor.clear();
		}
		for (int c = 0; c < connected_components.size(); ++c)
		{
			GetAndSetBNNs(c, connected_components);
		}
	} while (repeat);
	*/

	{
		Mat text_lines(otsu_binarized_image.size(), CV_8UC3);
		for (int c = 0; c < connected_components.size(); ++c)
		{
			int label = connected_components[c].labelValues[0];
			for (int p = 0; p < connected_components[c].pixelCoordinates.size(); ++p)
			{
				Vec3b& pixel = text_lines.at<Vec3b>(connected_components[c].pixelCoordinates[p].second, connected_components[c].pixelCoordinates[p].first);
				pixel = colors[label];
			}

			// and its top left corner...
			cv::Point pt1(connected_components[c].minX, connected_components[c].minY);
			// and its bottom right corner.
			cv::Point pt2(connected_components[c].maxX, connected_components[c].maxY);
			cv::rectangle(text_lines, pt1, pt2, cv::Scalar(0, 0, 255));
		}

		//Display connected components
		namedWindow("Connected Components Lines Extraction", WINDOW_AUTOSIZE);
		imshow("Connected Components Lines Extraction", text_lines);
		cout << "Press any key to continue... \n";
		waitKey(0);
	}

	cout << "Begin building text regions..." << endl;
	vector<TextRegion> text_regions;
	text_regions.push_back(connected_components[0]);
	text_regions[0].textLinesIndices.push_back(0);
	int currentIndex = 1; // current T
	//merge current T with first text region
	text_regions.push_back(connected_components[currentIndex]);
	text_regions[text_regions.size() - 1].textLinesIndices.push_back(currentIndex);
	connected_components[currentIndex].hasTextRegion = true;
	repeat = true;
	do {
		if (connected_components[currentIndex].BNNs.size() == 1)
		{
			int BNNindex = connected_components[currentIndex].BNNs[0];
			if (connected_components[BNNindex].BNNFor.size() <= 1)
			{
				//merge BNN with text region
				for (int p = 0; p < connected_components[BNNindex].pixelCoordinates.size(); ++p)
				{
					text_regions[text_regions.size() - 1].Encapsulate(connected_components[BNNindex].pixelCoordinates[p].first, connected_components[BNNindex].pixelCoordinates[p].second);
				}
				for (int l = 0; l < connected_components[BNNindex].labelValues.size(); ++l)
				{
					text_regions[text_regions.size() - 1].labelValues.push_back(connected_components[BNNindex].labelValues[l]);
				}
				text_regions[text_regions.size() - 1].textLinesIndices.push_back(BNNindex);
				connected_components[BNNindex].hasTextRegion = true;
				//connected_components.erase(connected_components.begin() + BNNindex);
				currentIndex = BNNindex;
			}
			else
			{
				//stop (search for next T)
				bool found = false;
				for (int c = 1; c < connected_components.size(); ++c)
				{
					if (!connected_components[c].hasTextRegion)
					{
						currentIndex = c;
						//merge current T with next text region
						TextRegion newTextRegion;
						for (int p = 0; p < connected_components[currentIndex].pixelCoordinates.size(); ++p)
						{
							newTextRegion.Encapsulate(connected_components[currentIndex].pixelCoordinates[p].first, connected_components[currentIndex].pixelCoordinates[p].second);
						}
						for (int l = 0; l < connected_components[currentIndex].labelValues.size(); ++l)
						{
							newTextRegion.labelValues.push_back(connected_components[currentIndex].labelValues[l]);
						}
						text_regions.push_back(newTextRegion);
						text_regions[text_regions.size() - 1].textLinesIndices.push_back(currentIndex);
						connected_components[currentIndex].hasTextRegion = true;
						found = true;
						break;
					}
				}
				if (!found)
				{
					repeat = false;
				}
			}
		}
		else
		{
			//stop (search for next T)
			bool found = false;
			for (int c = 1; c < connected_components.size(); ++c)
			{
				if (!connected_components[c].hasTextRegion)
				{
					currentIndex = c;
					//merge current T with next text region
					TextRegion newTextRegion;
					for (int p = 0; p < connected_components[currentIndex].pixelCoordinates.size(); ++p)
					{
						newTextRegion.Encapsulate(connected_components[currentIndex].pixelCoordinates[p].first, connected_components[currentIndex].pixelCoordinates[p].second);
					}
					for (int l = 0; l < connected_components[currentIndex].labelValues.size(); ++l)
					{
						newTextRegion.labelValues.push_back(connected_components[currentIndex].labelValues[l]);
					}
					text_regions.push_back(newTextRegion);
					text_regions[text_regions.size() - 1].textLinesIndices.push_back(currentIndex);
					connected_components[currentIndex].hasTextRegion = true;
					found = true;
					break;
				}
			}
			if (!found)
			{
				repeat = false;
			}
		}
	} while (repeat);

	cout << "Total text regions = " << text_regions.size() << endl;
	
	Mat mat_text_regions(otsu_binarized_image.size(), CV_8UC3);
	for (int t = 0; t < text_regions.size(); ++t)
	{
		cout << text_regions[t].labelValues.size() << endl;
		int label = text_regions[t].labelValues[0];
		for (int p = 0; p < text_regions[t].pixelCoordinates.size(); ++p)
		{
			Vec3b& pixel = mat_text_regions.at<Vec3b>(text_regions[t].pixelCoordinates[p].second, text_regions[t].pixelCoordinates[p].first);
			pixel = colors[label];
		}

		// and its top left corner...
		cv::Point pt1(text_regions[t].minX, text_regions[t].minY);
		// and its bottom right corner.
		cv::Point pt2(text_regions[t].maxX, text_regions[t].maxY);
		cv::rectangle(mat_text_regions, pt1, pt2, cv::Scalar(0, 0, 255));
	}

	//Display connected components
	namedWindow("Connected Components Text Regions", WINDOW_AUTOSIZE);
	imshow("Connected Components Text Regions", mat_text_regions);
	cout << "Press any key to continue... \n";
	waitKey(0);

	//Refine the text regions
	do {
		//cout << "Iteration " << iteration << endl;
		repeat = false;
		int ci = 0, cj = 0;
		for (ci = 1; ci < text_regions.size(); ++ci)
		{
			for (cj = 1; cj < text_regions.size(); ++cj)
			{
				if (cj == ci)
				{
					continue;
				}
				if (OverlapCheck(text_regions[ci], text_regions[cj]) || OverlapCheck(text_regions[cj], text_regions[ci]))
				{
					//merge connected components into one
					for (int p = 0; p < text_regions[cj].pixelCoordinates.size(); ++p)
					{
						text_regions[ci].Encapsulate(text_regions[cj].pixelCoordinates[p].first, text_regions[cj].pixelCoordinates[p].second);
					}
					for (int l = 0; l < text_regions[cj].labelValues.size(); ++l)
					{
						text_regions[ci].labelValues.push_back(text_regions[cj].labelValues[l]);
					}
					text_regions.erase(text_regions.begin() + cj);

					repeat = true;
					break;
				}
			}
			if (repeat)
			{
				break;
			}
		}
		if (repeat)
		{
			iteration++;
		}
	} while (repeat);
	
	{
		Mat mat_text_regions(otsu_binarized_image.size(), CV_8UC3);
		for (int t = 0; t < text_regions.size(); ++t)
		{
			cout << text_regions[t].labelValues.size() << endl;
			int label = text_regions[t].labelValues[0];
			for (int p = 0; p < text_regions[t].pixelCoordinates.size(); ++p)
			{
				Vec3b& pixel = mat_text_regions.at<Vec3b>(text_regions[t].pixelCoordinates[p].second, text_regions[t].pixelCoordinates[p].first);
				pixel = colors[label];
			}

			// and its top left corner...
			cv::Point pt1(text_regions[t].minX, text_regions[t].minY);
			// and its bottom right corner.
			cv::Point pt2(text_regions[t].maxX, text_regions[t].maxY);
			cv::rectangle(mat_text_regions, pt1, pt2, cv::Scalar(0, 0, 255));
		}

		//Display connected components
		namedWindow("Connected Components Text Regions", WINDOW_AUTOSIZE);
		imshow("Connected Components Text Regions", mat_text_regions);
		cout << "Press any key to continue... \n";
		waitKey(0);
	}

	//cout << "incepem";
	for (int tr = 0; tr < text_regions.size(); ++tr)
	{
		text_regions[tr].ComputeParameters();
		//std::sort(connected_components.begin(), connected_components.end(), less_than_key());
		/*
		for (int tl = 0; tl < text_regions[tr].textLinesIndices.size(); ++tl)
		{
			cout << connected_components[text_regions[tr].textLinesIndices[tl]].labelValues.size() << "   ";
		}
		cout << endl;*/
	}

	
	//Paragraph segmentation (scans every three adjacent text lines (tli−1,tli,tli+1) for each text region to find the segmentation line)
	vector<TextRegion> paragraphs;
	paragraphs.push_back(text_regions[0]);
	for (int tr = 1; tr < text_regions.size(); ++tr)
	{
		if (text_regions[tr].textLinesIndices.size() > 3 && text_regions[tr].width >= 150.0f)
		{
			int startL = 0;
			int totalLines = 0;
			float leftIndent = connected_components[text_regions[tr].textLinesIndices[0]].minX - text_regions[tr].minX;
			float rightIndent = text_regions[tr].maxX - connected_components[text_regions[tr].textLinesIndices[0]].maxX;
			for (int tl = 1; tl < (text_regions[tr].textLinesIndices.size() - 1); ++tl)
			{
				ConnectedComponent firstLine = connected_components[text_regions[tr].textLinesIndices[tl - 1]];
				ConnectedComponent secondLine = connected_components[text_regions[tr].textLinesIndices[tl]];
				ConnectedComponent thirdLine = connected_components[text_regions[tr].textLinesIndices[tl + 1]];
				float secondRightIndent = text_regions[tr].maxX - secondLine.maxX;
				float thirdLeftIndent = thirdLine.minX - text_regions[tr].minX;
				if ((secondRightIndent - rightIndent) > 5 && (thirdLeftIndent - leftIndent) > 5)
				{
					//split here into paragraph
					TextRegion newParagraph;
					for (int tlj = startL; tlj <= tl; ++tlj)
					{
						ConnectedComponent cc = connected_components[text_regions[tr].textLinesIndices[tlj]];
						for (int p = 0; p < cc.pixelCoordinates.size(); ++p)
						{
							newParagraph.Encapsulate(cc.pixelCoordinates[p].first, cc.pixelCoordinates[p].second);
						}
						for (int l = 0; l < cc.labelValues.size(); ++l)
						{
							newParagraph.labelValues.push_back(cc.labelValues[l]);
						}
						newParagraph.textLinesIndices.push_back(text_regions[tr].textLinesIndices[tlj]);
						totalLines++;
					}
					paragraphs.push_back(newParagraph);
					leftIndent = secondLine.minX - text_regions[tr].minX;
					rightIndent = text_regions[tr].maxX - thirdLine.maxX;
					tl += 1;
					startL = tl;
				}
			}

			if (totalLines < text_regions[tr].textLinesIndices.size())
			{
				//split here into paragraph
				TextRegion newParagraph;
				for (int tlj = startL; tlj < text_regions[tr].textLinesIndices.size(); ++tlj)
				{
					ConnectedComponent cc = connected_components[text_regions[tr].textLinesIndices[tlj]];
					for (int p = 0; p < cc.pixelCoordinates.size(); ++p)
					{
						newParagraph.Encapsulate(cc.pixelCoordinates[p].first, cc.pixelCoordinates[p].second);
					}
					for (int l = 0; l < cc.labelValues.size(); ++l)
					{
						newParagraph.labelValues.push_back(cc.labelValues[l]);
					}
					newParagraph.textLinesIndices.push_back(text_regions[tr].textLinesIndices[tlj]);
					totalLines++;
				}
				paragraphs.push_back(newParagraph);
			}
		}
		else
		{
			//cout << text_regions[tr].textLinesIndices.size() << "   " << text_regions[tr].width << endl;
			TextRegion newParagraph;
			for (int tlj = 0; tlj < text_regions[tr].textLinesIndices.size(); ++tlj)
			{
				ConnectedComponent cc = connected_components[text_regions[tr].textLinesIndices[tlj]];
				for (int p = 0; p < cc.pixelCoordinates.size(); ++p)
				{
					newParagraph.Encapsulate(cc.pixelCoordinates[p].first, cc.pixelCoordinates[p].second);
				}
				for (int l = 0; l < cc.labelValues.size(); ++l)
				{
					newParagraph.labelValues.push_back(cc.labelValues[l]);
				}
				newParagraph.textLinesIndices.push_back(text_regions[tr].textLinesIndices[tlj]);
			}
			paragraphs.push_back(newParagraph);
		}
	}

	cout << "Total text paragraphs = " << paragraphs.size() << endl;

	Mat mat_text_paragraphs(otsu_binarized_image.size(), CV_8UC3);
	for (int tp = 0; tp < paragraphs.size(); ++tp)
	{
		cout << paragraphs[tp].labelValues.size() << endl;
		if (paragraphs[tp].labelValues.size() > 0)
		{
			int label = paragraphs[tp].labelValues[0];
			for (int p = 0; p < paragraphs[tp].pixelCoordinates.size(); ++p)
			{
				Vec3b& pixel = mat_text_paragraphs.at<Vec3b>(paragraphs[tp].pixelCoordinates[p].second, paragraphs[tp].pixelCoordinates[p].first);
				pixel = colors[label];
				if (tp > 0)
				{
					pixel = Vec3b(0, 0, 0);
				}
				else
				{
					pixel = Vec3b(255, 255, 255);
				}
			}

			// and its top left corner...
			cv::Point pt1(paragraphs[tp].minX, paragraphs[tp].minY);
			// and its bottom right corner.
			cv::Point pt2(paragraphs[tp].maxX, paragraphs[tp].maxY);
			cv::rectangle(mat_text_paragraphs, pt1, pt2, cv::Scalar(0, 0, 255));
		}
		else
		{
			cout << tp << "   " << paragraphs[tp].labelValues.size() << endl;
		}
		
	}

	//Display Text Paragraphs
	namedWindow("Connected Components Text Paragraphs", WINDOW_AUTOSIZE);
	imshow("Connected Components Text Paragraphs", mat_text_paragraphs);
	cout << "Press any key to continue... \n";
	waitKey(0);

	//Text Region Segmentation (adaptive morphological closing)
	for (int tp = 1; tp < paragraphs.size(); ++tp)
	{
		cout << "Paragraph " << tp << " --- " << paragraphs[tp].textLinesIndices.size() << " (" << paragraphs[tp].minX << "," << paragraphs[tp].minY << ") - (" << paragraphs[tp].maxX << "," << paragraphs[tp].maxY << ")" << endl;
		int kernelHeight;
		if (paragraphs[tp].textLinesIndices.size() > 1)
		{
			vector<int> wSet; // vertical distance between the two adjacent text lines
			cout << "Vertical distances between the " << paragraphs[tp].textLinesIndices.size() << " text lines: " << endl;
			for (int tl = 0; tl < (paragraphs[tp].textLinesIndices.size() - 1); ++tl)
			{
				wSet.push_back(connected_components[paragraphs[tp].textLinesIndices[tl + 1]].maxY - connected_components[paragraphs[tp].textLinesIndices[tl]].maxY);
				cout << (connected_components[paragraphs[tp].textLinesIndices[tl + 1]].maxY - connected_components[paragraphs[tp].textLinesIndices[tl]].maxY) << "  " << endl;
			}
			cout << endl;
			kernelHeight = wSet[floor((3.0f / 4.0f) * wSet.size())];
		}
		else
		{
			kernelHeight = paragraphs[tp].maxY - paragraphs[tp].minY;
		}
		cout << "kernerlHeight = " << kernelHeight << endl;

		int kernelWidth;
		vector<int> WSSet; // distances between the two nearest text elements
		cout << "Min dst for the " << paragraphs[tp].textLinesIndices.size() << " text lines: " << endl;
		for (int tl = 0; tl < paragraphs[tp].textLinesIndices.size(); ++tl)
		{
			int d = connected_components[paragraphs[tp].textLinesIndices[tl]].minDst;
			if (d == INT_MAX)
			{
				d = connected_components[paragraphs[tp].textLinesIndices[tl]].maxX - connected_components[paragraphs[tp].textLinesIndices[tl]].minX;
			}
			WSSet.push_back(d);
			cout << connected_components[paragraphs[tp].textLinesIndices[tl]].minDst << "  " << endl;
		}
		cout << endl;
		kernelWidth = WSSet[floor((3.0f / 4.0f) * WSSet.size())];
		if (kernelWidth == INT_MAX)
		{
			kernelWidth = paragraphs[tp].maxX - paragraphs[tp].minX;
		}
		cout << "kernerlWidth = " << kernelWidth << endl;
	}
	waitKey(0);

	return 0;
}


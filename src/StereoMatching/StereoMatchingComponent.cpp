/*
 * Ubitrack - Library for Ubiquitous Tracking
 * Copyright 2006, Technische Universitaet Muenchen, and individual
 * contributors as indicated by the @authors tag. See the
 * copyright.txt in the distribution for a full listing of individual
 * contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */

/**
 * @ingroup vision_components
 * @file
 * Reads stereo camera images and intrinsics to generate a depth map
 *
 * @author Philipp Müller
 * @TODO:	
 *		- Read camera calibration from ubitrack calib files
 *		- Fix heatmap display of disparity / depth map
 */

#include <string>
#include <list>
#include <iostream>
#include <iomanip>
#include <strstream>
#include <log4cpp/Category.hh>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>

#include <utDataflow/TriggerComponent.h>
#include <utDataflow/TriggerInPort.h>
#include <utDataflow/TriggerOutPort.h>
#include <utDataflow/ComponentFactory.h>
#include <utMeasurement/Measurement.h>


#include <utUtil/TracingProvider.h>
#include <opencv/cv.h>
#include <utVision/Image.h>


// get a logger
static log4cpp::Category& logger(log4cpp::Category::getInstance("Ubitrack.Vision.StereoMatching"));

using namespace Ubitrack;
using namespace Ubitrack::Vision;

namespace Ubitrack { namespace Drivers {

/**
* @ingroup vision_components
*
* @par Input Ports
* None.
*
* @par Output Ports
* \c Output push port of type Ubitrack::Measurement::ImageMeasurement.
*
* @par Configuration
* The configuration tag contains a \c <dsvl_input> configuration.
* For details, see the DirectShow documentation...
*
*/
class StereoMatchingComponent
	: public Dataflow::TriggerComponent
{
public:
	StereoMatchingComponent(const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph >);
	~StereoMatchingComponent();

	void start();
	void stop();
	void compute(Measurement::Timestamp t);
	void stereoMatching(Measurement::Timestamp timeStamp);

protected:
	Dataflow::TriggerInPort< Measurement::ImageMeasurement > m_grayImage1;
	Dataflow::TriggerInPort< Measurement::ImageMeasurement > m_grayImage2;
	Dataflow::PullConsumer< Measurement::CameraIntrinsics > m_camera1Intrinsics;
	Dataflow::PullConsumer< Measurement::CameraIntrinsics > m_camera2Intrinsics;
	Dataflow::TriggerInPort< Measurement::Pose > m_camera1toCamera2;

	Dataflow::TriggerOutPort< Measurement::ImageMeasurement > m_outDepthImage;
};


StereoMatchingComponent::StereoMatchingComponent(const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > subgraph)
	: Dataflow::TriggerComponent(sName, subgraph)
	, m_grayImage1("GrayImage1", *this)
	, m_grayImage2("GrayImage2", *this)
	, m_camera1Intrinsics("CameraIntrinsics1", *this)
	, m_camera2Intrinsics("CameraIntrinsics2", *this)
	, m_camera1toCamera2("Camera1toCamera2", *this)
	, m_outDepthImage("DepthImage", *this)
{

}

StereoMatchingComponent::~StereoMatchingComponent()
{

}


void StereoMatchingComponent::start()
{
	Component::start();
}

void StereoMatchingComponent::stop()
{
	Component::stop();
}

void StereoMatchingComponent::compute(Measurement::Timestamp t)
{
	LOG4CPP_INFO(logger, "Stereo matching started...");
	stereoMatching(t);
	//m_outDepthImage.send(image1);
}

template<typename T, std::size_t X, std::size_t Y>
Math::Matrix<T, Y, X> transpose(const Math::Matrix<T, X, Y> &m)
{
	Math::Matrix<T, Y, X> mTransposed;
	T *mData = (T*)m.content();
	T *tData = (T*)mTransposed.content();
	for (uint32_t y = 0; y < X; y++)
	{
		for (uint32_t x = 0; x < Y; x++)
		{
			tData[y + x * Y] = mData[x + y * X];
		}
	}

	return mTransposed;
}

void StereoMatchingComponent::stereoMatching(Measurement::Timestamp timeStamp)
{
	std::string intrinsic_filename = "C://Users//FAR-Student//Desktop//Phil//CalibDataset//intrinsics.yml";
	std::string extrinsic_filename = "C://Users//FAR-Student//Desktop//Phil//CalibDataset//extrinsics.yml";

	enum 
	{ 
		STEREO_BM = 0,		// Block matching
		STEREO_SGBM = 1,	// Semi global block matching (1 pass consider only 5 directions instead of 8)
		STEREO_HH = 2,		// Semi global block matching (full-scale two-pass dynamic programming algorithm)
		STEREO_VAR = 3,		// Semi global block matching (???)
		STEREO_3WAY = 4		// Semi global block matching (???)
	};
	int alg = STEREO_HH;
	int SADWindowSize = 11;			// Sum of absolute differences window size
	int numberOfDisparities = 128;	
	float scale = 1.0f;

	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);

	if (numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
	{
		LOG4CPP_ERROR(logger, "The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
		return;
	}
	if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
	{
		LOG4CPP_ERROR(logger, "The block size (--blocksize=<...>) must be a positive odd number\n");
		return;
	}

	int colorMode = alg == STEREO_BM ? 0 : -1;
	cv::Mat img1 = m_grayImage1.get()->Mat();
	cv::Mat img2 = m_grayImage2.get()->Mat();

	if (img1.empty())
	{
		LOG4CPP_ERROR(logger, "Could not load the first input image file\n");
		return;
	}
	if (img2.empty())
	{
		LOG4CPP_ERROR(logger, "Could not load the second input image file\n");
		return;
	}
	if (scale != 1.f)
	{
		cv::Mat temp1, temp2;
		int method = scale < 1 ? cv::INTER_AREA : cv::INTER_CUBIC;
		resize(img1, temp1, cv::Size(), scale, scale, method);
		img1 = temp1;
		resize(img2, temp2, cv::Size(), scale, scale, method);
		img2 = temp2;
	}

	cv::Size img_size = img1.size();

	cv::Rect roi1, roi2;
	cv::Mat Q;

	if (m_camera1Intrinsics.isConnected() && m_camera2Intrinsics.isConnected())
	{
		// Reading intrinsic parameters
		cv::FileStorage fs(intrinsic_filename, cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			LOG4CPP_ERROR(logger, "Failed to open file " << intrinsic_filename.c_str());
			return;
		}

		cv::Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		//Math::Matrix<double, 3, 3> camMat1 = transpose(m_camera1Intrinsics.get(timeStamp)->matrix);
		//Math::Vector4d camDist1 = m_camera1Intrinsics.get(timeStamp)->radial_params;
		//Math::Matrix<double, 3, 3> camMat2 = transpose(m_camera2Intrinsics.get(timeStamp)->matrix);
		//Math::Vector4d camDist2 = m_camera2Intrinsics.get(timeStamp)->radial_params;

		//M1 = cv::Mat(3, 3, CV_64FC1, (void*)camMat1.content());
		//D1 = cv::Mat(1, 4, CV_64FC1, (void*)camDist1.content());
		//M2 = cv::Mat(3, 3, CV_64FC1, (void*)camMat2.content());
		//D2 = cv::Mat(1, 4, CV_64FC1, (void*)camDist2.content());

		M1 *= scale;
		M2 *= scale;

		// Reading extrinsic parameters
		fs.open(extrinsic_filename, cv::FileStorage::READ);
		if (!fs.isOpened())
		{
			LOG4CPP_ERROR(logger, "Failed to open file " << extrinsic_filename.c_str());
			return;
		}

		cv::Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;

		//Math::Matrix3x3d camPoseR; m_camera1toCamera2.get()->rotation().toMatrix(camPoseR);
		//Math::Vector3d camPoseT = m_camera1toCamera2.get()->translation();

		// Rectification
		//int64 t = cv::getTickCount();

		cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

		cv::Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		cv::Mat img1r, img2r;
		remap(img1, img1r, map11, map12, cv::INTER_LINEAR);
		remap(img2, img2r, map21, map22, cv::INTER_LINEAR);

		img1 = img1r;
		img2 = img2r;

		//t = cv::getTickCount() - t;
		//LOG4CPP_INFO(logger, "Rectification time elapsed: " << (t * 1000 / cv::getTickFrequency()) << "ms");
	}

	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterCap(31);
	bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
	bm->setMinDisparity(0);
	bm->setNumDisparities(numberOfDisparities);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(1);

	sgbm->setPreFilterCap(63);
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm->setBlockSize(sgbmWinSize);

	int cn = img1.channels();

	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	if (alg == STEREO_HH)
		sgbm->setMode(cv::StereoSGBM::MODE_HH);
	else if (alg == STEREO_SGBM)
		sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
	else if (alg == STEREO_3WAY)
		sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

	cv::Mat disp, disp8;
	int64 t = cv::getTickCount();
	if (alg == STEREO_BM)
		bm->compute(img1, img2, disp);
	else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
		sgbm->compute(img1, img2, disp);
	t = cv::getTickCount() - t;
	LOG4CPP_INFO(logger, "Stereo matching time elapsed: " << (t * 1000 / cv::getTickFrequency()) << "ms");

	//disp = dispp.colRange(numberOfDisparities, img1p.cols);
	if (alg != STEREO_VAR)
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.0));
	else
		disp.convertTo(disp8, CV_8U);

	// Output resulting depth image
	if (m_outDepthImage.isConnected())
	{
		cv::Mat xyz;
		cv::reprojectImageTo3D(disp, xyz, Q, false, CV_32F);

		//cv::Mat heatmap;
		//cv::applyColorMap(xyz, heatmap, cv::ColormapTypes::COLORMAP_HOT);
		//cv::imwrite("C://Users//FAR-Student//Desktop//Phil//heatmap.png", heatmap);
		//m_outDepthImage.send(Measurement::ImageMeasurement(timeStamp, ImagePtr(new Image(heatMap))));

		m_outDepthImage.send(Measurement::ImageMeasurement(timeStamp, ImagePtr(new Image(disp8))));
	}
}

} } // namespace Ubitrack::Driver

UBITRACK_REGISTER_COMPONENT(Dataflow::ComponentFactory* const cf) {
	cf->registerComponent< Ubitrack::Drivers::StereoMatchingComponent >("StereoMatchingComponent");
}


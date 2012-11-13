/*
Sample code for using SimKinect
*/
#include "SimpleKinectReader.h" 
#include <iostream>
#include <cv.h>
#include <ml.h>
#include <highgui.h>

using namespace std;
using namespace cv;

//edit this path if you want to read data from oni file
char* file_path = "D:/2_side/SideBin01.oni";
//edit this path if you want to save oni file
const char* save_path = "sample.oni";
//default width and height
int video_size_width = 640;
int video_size_height = 480;

int detect_window_size = 64;
int ROI_size = 160;
int detectObject ( const Mat& image, const CvSVM& svm );
int main ( ) {
	CvSVM svm = CvSVM();

	svm.load("E:/source/gestureRecognition/objectDetector/SVM_DATA.xml");
	unsigned char* depth_data=NULL; 
	unsigned char* color_data=NULL;
	int* depth_map=NULL;
	int ret;
	bool quit = false;
	char buffer[100];
	//Create a SimKinect instance
	SimKinect sensor;
	if ( ret = sensor.Init() ) {
		printf("Failed to initialize: %d\n",ret);
		exit(1);
	}
	color_data = new unsigned char[video_size_width*video_size_height*3];
	depth_data = new unsigned char[video_size_width*video_size_height*3];
	depth_map = new int[video_size_width*video_size_height];
	int frame_cnt =0 ;
	vector<SKUser> users;
	while ( !quit ) {
		double t = (double)getTickCount(); //for calc FPS
		sensor.GetNextFrame(color_data,depth_data,depth_map);
		sensor.GetUsers(users);
		//Draw the frames with OpenCV
		Mat color_img(video_size_height,video_size_width,CV_MAKETYPE(8,3),color_data);
		Mat depth_img(video_size_height,video_size_width,CV_MAKETYPE(8,3),depth_data);
		Mat test_img(cvSize(64,64),CV_MAKETYPE(8,3));
 
		if ( users.size()>0 ) {
			SKUser user=users[0]; 
			for ( int i=1; i <users.size() ; i++ ) {
				if ( users[i].real_joints[SK_SKEL_COM].z < user.real_joints[SK_SKEL_COM].z ) {
					user = users[i];
				}
			}
			int row_begin = max(0,user.proj_joints[SK_SKEL_LEFT_HAND].y-ROI_size/2);
			int row_end = min(video_size_height,row_begin+ROI_size);
			int col_begin = max(0,user.proj_joints[SK_SKEL_LEFT_HAND].x-ROI_size/2);
			int col_end = min(video_size_width,col_begin+ROI_size);
			try {
				Mat imageROI = color_img(Range(row_begin,row_end),Range(col_begin,col_end));
				Mat smallImageROI;
				resize(imageROI,smallImageROI,cvSize(ROI_size/2,ROI_size/2));
				imshow("small",smallImageROI);
				waitKey(10);
				//rectangle(color_img,cvPoint(col_begin,row_begin),cvPoint(col_end,row_end),cvScalar(255,255,0),2);
				vector<Rect> BBs; 
				
				for ( int i=0 ; i<imageROI.rows-detect_window_size ; i+=50 ) {
					for ( int j=0 ; j<imageROI.cols-detect_window_size ; j+=50 ) {
						test_img = imageROI(Range(i,i+detect_window_size),Range(j,j+detect_window_size));
						if ( detectObject(test_img,svm) ) {
							BBs.push_back(cvRect(col_begin+j,row_begin+i,detect_window_size,detect_window_size));
						}
					}
				}
				/*
				for ( int i=0 ; i<smallImageROI.rows-detect_window_size ; i+=10 ) {
					for ( int j=0 ; j<smallImageROI.cols-detect_window_size ; j+=10 ) {
						test_img = smallImageROI(Range(i,i+detect_window_size),Range(j,j+detect_window_size));
						if ( detectObject(test_img,svm) ) {
							BBs.push_back(cvRect(col_begin+j*2,row_begin+i*2,detect_window_size*2,detect_window_size*2));
						}
					}
				}
				*/
				for ( int i=0 ; i<BBs.size () ; i++ ) {
					putText(color_img,"bottle",cvPoint(BBs[i].x,BBs[i].y),CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(0,255,0),1);
					rectangle(color_img,BBs[i],cvScalar(0,255,0),2);
				}

				for ( int i=0 ; i<users.size() ; i++ ) {
					DrawUser(users[i],depth_img);  //Add user information ( user label, and joint dot )
					//DrawUser(users[i],color_img);
				}

				t = getTickFrequency()/((double)getTickCount()-t);
				sprintf(buffer,"%d",(int)t);
				putText(color_img,string(buffer),cvPoint(10,50),CV_FONT_HERSHEY_SIMPLEX,1,CV_RGB(0,0,0),2);
				putText(depth_img,string(buffer),cvPoint(10,50),CV_FONT_HERSHEY_SIMPLEX,1,CV_RGB(255,255,255),2);
			}
			catch ( Exception e ) {
			}
		}
		imshow("COLOR",color_img);
		imshow("DEPTH",depth_img);
		char time_buf[10];
		char key = waitKey(30);
		switch (key) {
		case 27: 
			quit = true;
			break;
		case 'c':
			sprintf(time_buf,"%d",(int)time(NULL));
			imwrite("screenshot_color_"+string(time_buf)+".jpg",color_img);
			imwrite("screenshot_depth_"+string(time_buf)+".jpg",depth_img);
			printf("screenshot saved\n");
			break;
		}

	}
	//Stop record before exit
	//sensor.StopRecord();
}

int detectObject ( const Mat& image, const CvSVM& svm ) {
	HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);   
  vector<float>descriptors;   
  hog->compute(image, descriptors,Size(1,1), Size(0,0)); 
  //cout<<"HOG dims: "<<descriptors.size()<<endl;  
  CvMat* SVMtrainMat=cvCreateMat(1,descriptors.size(),CV_32FC1);  
  int n=0;  
  for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++) {  
		cvmSet(SVMtrainMat,0,n,*iter);  
    n++;  
	}
  int ret = svm.predict(SVMtrainMat);
	return ret;
}
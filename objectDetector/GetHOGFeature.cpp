#include <cv.h>  
#include <highgui.h> 
#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
using namespace cv;  
using namespace std;  

void saveFeature ( int cat, vector<float>& feature, string filename );

int main(int argc, char** argv)    
{    
	vector<string> img_path;  
	vector<int> img_catg;  
	int nLine = 0;  
	string buf;  
	ifstream svm_data( "E:/source/gestureRecognition/objectDetector/bottle_svm_data" );  
	Mat src;  
	Mat trainImg(cvSize(64,64),8,3);

	while( svm_data )  
	{  
		if( getline( svm_data, buf ) )  
		{   
			if( nLine % 2 == 0 )  
			{  
				img_catg.push_back( atoi( buf.c_str() ) );  
			}  
			else  
			{  
				img_path.push_back( buf );
			}
			nLine ++; 
		}  
	}  
	svm_data.close();  

	

	for( string::size_type i = 0; i != img_path.size(); i++ )  
	{  
		try {
			src=imread(img_path[i],1);
		}
		catch ( Exception e ) {
			cout<<e.what()<<endl;
			continue;
		}

		cout<<" processing "<<img_path[i].c_str()<<endl;  

		resize(src,trainImg,cvSize(64,64));    
		HOGDescriptor hog(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);    
		vector<float>hog_features;   
		hog.compute(trainImg, hog_features,Size(1,1), Size(0,0));   
		cout<<"HOG dims: "<<hog_features.size()<<endl; 
		saveFeature(img_catg[i],hog_features,"train_data.txt");
		cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;  
	}  

	return 0;  
}  

void saveFeature ( int cat, vector<float>& feature, string filename ) {
	fstream saveStream(filename,fstream::out | fstream::app);
	saveStream<<cat;
	for ( int i=0 ; i<feature.size() ; i++ ) {
		saveStream<<" "<<i<<":"<<feature[i];
	}
	saveStream<<endl;
}
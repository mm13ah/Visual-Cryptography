#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QPixmap>
#include <QImage>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <algorithm>
#include <random>
#include <math.h>
#include <chrono>
#include <limits>

/*global variables used throughout the buttons*/
cv::Mat img; //input image
cv::Mat img_bw; //inut image in black & white
cv::Mat img_g; //input image in greyscale
cv::Mat combShare; //combined share
std::vector<cv::Mat> shares; //contains shares in opencv format
std::vector<QImage> sharesArray; //contains shares as images to be displayed by qt
std::vector<cv::Mat> combinedShares; //contains instances of combined shares
unsigned int count; //count of shares combined
int scheme; //used to communicate which share is selected between functions
int k,n; //threshold and no. of shares

/*function that finds all subsets of a set */
std::vector<std::vector<int> > findAllSubsets(std::vector<int>& set)
{
  std::vector<std::vector<int> > subset;
  std::vector<int> emptySet;
  subset.push_back(emptySet);

  for(unsigned int i=0; i<set.size();i++)
  {
    std::vector<std::vector<int> > subsetTemp = subset;

    for(unsigned int j=0; j<subsetTemp.size(); j++)
    {
      subsetTemp[j].push_back(set[i]);
    }
    for (unsigned int j=0; j<subsetTemp.size(); j++)
    {
      subset.push_back(subsetTemp[j]);
    }
  }
  return subset;
}


/*function that returns even subsets given a set of subsets */
std::vector<std::vector<int> > findEvenSubsets(std::vector<std::vector<int> >& subSets)
{
  std::vector<std::vector<int> > even;
  for(std::vector<std::vector<int> >::size_type i = 0; i < subSets.size(); i++)
  {
    if (subSets[i].size() % 2 == 0)
    {
      even.push_back(subSets[i]);
    }
  }
  return even;
}


/*function that returns odd subsets given a set of subsets */
std::vector<std::vector<int> > findOddSubsets(std::vector<std::vector<int> >& subSets)
{
  std::vector<std::vector<int> > odd;
  for(std::vector<std::vector<int> >::size_type i=0; i<subSets.size(); i++)
  {
    if (subSets[i].size() % 2 == 1)
    {
      odd.push_back(subSets[i]);
    }
  }
  return odd;
}


/*function that, given a set of even subsets and a list {1,..,k}, calculates
  white base matrix (matrix used to share white pixel) for (k,k) scheme */
std::vector<std::vector<int> > whiteBaseMatrix(std::vector<std::vector<int> >& even, std::vector<int>& set)
{
  int k = set.size();
  int row = k;
  int column = pow(2, k-1);
  std::vector<std::vector<int> > B0(row, std::vector<int>(column));

  int tempcount;
  for(int i=0; i<k; i++)
  {
    for(std::vector<std::vector<int> >::size_type j=0; j<even.size(); j++)
    {
      tempcount = 0;
      for(std::vector<std::vector<int> >::size_type a=0; a<even[j].size(); a++)
      {
        if(even[j][a]==set[i])
        {
          tempcount++;
        }
      }
      if(tempcount>0)
      {
        B0[i][j]=1;
      }
      else
      {
        B0[i][j]=0;
      }
    }
  }
  return B0;
}


/*function that, given a set of odd subsets and a list {1,...,k}, calculates
  black base matrix (matrix used to share black pixel) for (k,k) scheme */
std::vector<std::vector<int> > blackBaseMatrix(std::vector<std::vector<int> >& odd, std::vector<int>& set)
{
  int k = set.size();
  int row = k;
  int column = pow(2, k-1);
  std::vector<std::vector<int> > B1(row, std::vector<int>(column));

  int tempcount;
  for(int i=0; i<k; i++)
  {
    for(std::vector<std::vector<int> >::size_type j=0; j<odd.size(); j++)
    {
      tempcount = 0;
      for(std::vector<std::vector<int> >::size_type a=0; a<odd[j].size(); a++)
      {
        if(odd[j][a]==set[i])
        {
          tempcount++;
        }
      }
      if(tempcount>0)
      {
        B1[i][j]=1;
      }
      else
      {
        B1[i][j]=0;
      }
    }
  }
  return B1;
}


/*function that, given a matrix, returns a random permutation
  of the columns of that matrix */
std::vector<std::vector<int> > permuteMatrix(std::vector<std::vector<int> >& matrix)
{
  /*get number of rows and columns of matrix */
  int rows = matrix.size();
  int columns = matrix[0].size();

  /*initialise a set the size of the no. of columns that
  can be permuted to give a random permutation */
  std::vector<int> set;
  for(int i=0; i<columns; i++)
  {
    set.insert(set.begin()+i, i);
  }

  /* shuffle set - use seed from std::srand(time(0)) */
  std::random_shuffle(std::begin(set), std::end(set));

  /*initalise empty matrix to store permutation */
  std::vector<std::vector<int> > permutedMatrix(rows, std::vector<int>(columns));

  for(std::vector<std::vector<int> >::size_type i=0; i<matrix.size(); i++)
  {
    for(std::vector<std::vector<int> >::size_type j=0; j<matrix[i].size(); j++)
    {
      permutedMatrix[i][j]=matrix[i][set[j]];
    }
  }
  return permutedMatrix;
}


/*function that takes a binary matrix and converts it to opencv
  format, i.e. white:0 --> 255, black: 1 --> 0 */
std::vector<std::vector<int> > convertToOpenCv(std::vector<std::vector<int> >& matrix)
{
  std::vector<std::vector<int> > opencvMatrix(matrix.size(), std::vector<int>(matrix[0].size()));
  for(std::vector<std::vector<int> >::size_type i=0; i<matrix.size(); i++)
  {
    for(std::vector<std::vector<int> >::size_type j=0; j<matrix[i].size(); j++)
    {
      if (matrix[i][j]==0) //white
      {
        opencvMatrix[i][j]=255;
      }
      else if (matrix[i][j]==1) //black
      {
        opencvMatrix[i][j]=0;
      }
    }
  }
  return opencvMatrix;
}


/*(k,k) scheme, takes secret image and k as input and returns
  a vector of the shares generated */
std::vector<cv::Mat> KbyKscheme(cv::Mat secretImage, int k)
{
  int m = pow(2,k-1); //pixel expansion

  //int row = k; //initalise row and column lengths of the base matrices
  //int column = pow(2,k-1);

  std::vector<int> set; // set = {1,...,k} used to calculate base matrices
  for (int i=0; i<k; i++)
  {
    set.insert(set.begin()+i, i+1);
  }

  /* Get all subsets and even and odd subsets */
  std::vector<std::vector<int> > allSubsets = findAllSubsets(set);
  std::vector<std::vector<int> > evenSubsets = findEvenSubsets(allSubsets);
  std::vector<std::vector<int> > oddSubsets = findOddSubsets(allSubsets);

  /* Initialise base matrics and convert to Opencv format */
  std::vector<std::vector<int> > Base0 = whiteBaseMatrix(evenSubsets, set);
  std::vector<std::vector<int> > Base1 = blackBaseMatrix(oddSubsets, set);
  std::vector<std::vector<int> > B0 = convertToOpenCv(Base0);
  std::vector<std::vector<int> > B1 = convertToOpenCv(Base1);

  /* create k blank images, used for shares - clone is a deep copy (own pixel-pointers) */
  std::vector<cv::Mat> shares;
  cv::Mat blankshare(m*secretImage.rows, m*secretImage.cols, CV_8UC1, cv::Scalar(255,255,255));
  for(int i=0;i<k;i++)
  {
    shares.insert(shares.begin() + i, blankshare.clone());
  }

  /* Initialise empty matrix used for permutations */
  std::vector<std::vector<int> > permutedMatrix;

  /* loop through each pixel of the secret image, checking colour of pixel */
  for(int i=0;i<secretImage.rows;i++)
  {
    for(int j=0;j<secretImage.cols;j++)
    {
      if ((int)secretImage.at<uchar>(i,j) == 255) //whitepixel
      {
          for (int x=0; x<m; x++)
          {
              permutedMatrix = permuteMatrix(B0);
              for(int numberShares=0; numberShares<k; numberShares++)
              {
                  for(int subPixels=0; subPixels<m; subPixels++)
                  {
                      shares.at(numberShares).at<uchar>(m*i+x,m*j+subPixels)=permutedMatrix[numberShares][subPixels];
                  }
              }
          }
      }
      else if ((int)secretImage.at<uchar>(i,j) == 0) //blackpixel
      {
          for (int x=0; x<m; x++)
          {
              permutedMatrix = permuteMatrix(B1);
              for(int numberShares=0; numberShares<k; numberShares++)
              {
                  for(int subPixels=0; subPixels<m; subPixels++)
                  {
                      shares.at(numberShares).at<uchar>(m*i+x,m*j+subPixels)=permutedMatrix[numberShares][subPixels];
                  }
              }
          }
      }
    }
  }
  return shares;
}


/*(2,n) scheme, takes secret image and n as input and returns
  a vector of shares */
std::vector<cv::Mat> TwoByNscheme(cv::Mat secretImage, int n)
{
  int m=n; //pixel expansion

  /*initialise base matrices */
  std::vector<std::vector<int> > Base0(m, std::vector<int>(m));
  std::vector<std::vector<int> > Base1(m, std::vector<int>(m));
  for(int i=0; i<n; i++)
  {
    for(int j=0; j<n; j++)
    {
      if (j==0)
      {
        Base0[i][j]=1;
      }
      else
      {
        Base0[i][j]=0;
      }
    }
  }

  for(int i=0; i<n; i++)
  {
    for(int j=0; j<n; j++)
    {
      if (i==j)
      {
        Base1[i][j]=1;
      }
      else
      {
        Base1[i][j]=0;
      }
    }
  }

  std::vector<std::vector<int> > B0 = convertToOpenCv(Base0);
  std::vector<std::vector<int> > B1 = convertToOpenCv(Base1);

  //initialise blank matrix used for permutations
  std::vector<std::vector<int> > permutedMatrix;

  //create blank shares
  std::vector<cv::Mat> shares;
  cv::Mat blankshare(n*secretImage.rows, n*secretImage.cols, CV_8UC1, cv::Scalar(255,255,255));
  for(int i=0;i<n;i++)
  {
    shares.insert(shares.begin() + i, blankshare.clone()); //clone is a deep copy (own pixel-pointers)
  }

  /* loop through each pixel of the secret image, checking colour of pixel */
  for(int i=0;i<secretImage.rows;i++)
  {
    for(int j=0;j<secretImage.cols;j++)
    {
      if ((int)secretImage.at<uchar>(i,j) == 255) //whitepixel
      {
        for (int x=0; x<m; x++)
        {
          permutedMatrix = permuteMatrix(B0);
          for(int numberShares=0; numberShares<n; numberShares++)
          {
            for(int subPixels=0; subPixels<m; subPixels++)
            {
              shares.at(numberShares).at<uchar>(m*i+x,m*j+subPixels)=permutedMatrix[numberShares][subPixels];
            }
          }
        }
      }

      else if ((int)secretImage.at<uchar>(i,j) == 0) //blackpixel
      {
        for (int x=0; x<m; x++)
        {
          permutedMatrix = permuteMatrix(B1);
          for(int numberShares=0; numberShares<n; numberShares++)
          {
            for(int subPixels=0; subPixels<m; subPixels++)
            {
              shares.at(numberShares).at<uchar>(m*i+x,m*j+subPixels)=permutedMatrix[numberShares][subPixels];
            }
          }
        }
      }
    }
  }
  return shares;
}


/*(3,4) scheme, takes secret image and returns vector of shares */
std::vector<cv::Mat> ThreeByFourScheme(cv::Mat secretImage)
{
  int k=4;
  int m=6;

  /* Define base matrices */
  std::vector<std::vector<int> > B0 = { {255,255,0,0,0,255},
                                        {255,0,255,0,0,255},
                                        {255,0,0,255,0,255},
                                        {255,0,0,0,255,255} };

  std::vector<std::vector<int> > B1 = { {0,255,255,255,0,0},
                                        {0,255,255,0,255,0},
                                        {0,255,0,255,255,0},
                                        {0,0,255,255,255,0} };

  /* Initialise empty matrix used for permutations */
  std::vector<std::vector<int> > permutedMatrix;

  /* create 4 blank shares - clone is a deep copy (own pixel-pointers) */
  std::vector<cv::Mat> shares;
  cv::Mat blankshare(m*secretImage.rows, m*secretImage.cols, CV_8UC1, cv::Scalar(255,255,255));
  for(int i=0;i<k;i++)
  {
    shares.insert(shares.begin() + i, blankshare.clone());
  }

  /* loop through each pixel of the secret image, checking colour of pixel */
  for(int i=0;i<secretImage.rows;i++)
  {
    for(int j=0;j<secretImage.cols;j++)
    {
      if ((int)secretImage.at<uchar>(i,j) == 255) //whitepixel
      {
          for (int x=0; x<m; x++)
          {
              permutedMatrix = permuteMatrix(B0);
              for(int numberShares=0; numberShares<k; numberShares++)
              {
                  for(int subPixels=0; subPixels<m; subPixels++)
                  {
                      shares.at(numberShares).at<uchar>(m*i+x,m*j+subPixels)=permutedMatrix[numberShares][subPixels];
                  }
              }
          }
      }
      else if ((int)secretImage.at<uchar>(i,j) == 0) //blackpixel
      {
          for (int x=0; x<m; x++)
          {
              permutedMatrix = permuteMatrix(B1);
              for(int numberShares=0; numberShares<k; numberShares++)
              {
                  for(int subPixels=0; subPixels<m; subPixels++)
                  {
                      shares.at(numberShares).at<uchar>(m*i+x,m*j+subPixels)=permutedMatrix[numberShares][subPixels];
                  }
              }
          }
      }
    }
  }
  return shares;
}


/*function that takes vector of shares and returns the
  superposition of them */
cv::Mat combineShares(std::vector<cv::Mat>& shares)
{
  cv::Mat combShare(shares[0].rows, shares[0].cols, CV_8UC1, cv::Scalar(0,0,0));
  int temp;
  for(int i=0;i<combShare.rows;i++)
  {
    for(int j=0;j<combShare.cols;j++)
    {
      temp = 0;
      for(unsigned int k=0;k<shares.size();k++)
      {
        if (shares.at(k).at<uchar>(i,j)==0)
        {
          temp++;
        }
      }
      if (temp > 0) //if any of the pixels were black
      {
        combShare.at<uchar>(i,j)=0;
      }
      else if (temp==0) //else pixel is white
      {
        combShare.at<uchar>(i,j)=255;
      }
    }
  }
  return combShare;
}


/*function that takes a grey image and converts it to 1D halftone */
cv::Mat convertToHalftone(cv::Mat image)
{

  cv::Mat halftone(image.rows, image.cols, CV_8UC1, cv::Scalar(255,255,255));
  cv::Scalar intensity;
  int error;

  /* convert image to halftone */
  for(int i=0;i<image.rows;i++)
  {
    error = 0;
    for(int j=0;j<image.cols;j++)
    {
      intensity = image.at<uchar>(i,j);
      intensity[0] += error;

      if (intensity[0] < 128)
      {
        halftone.at<uchar>(i,j) = 0;
        error = intensity[0];
      }
      else
      {
        halftone.at<uchar>(i,j) = 255;
        error = intensity[0] - 255;
      }
    }
  }
  return halftone;
}


/*function that takes image and position of pixel to be halftoned
  (i.e R, G or B) as input and returns the halftone image*/
cv::Mat convertToColourHalftone(cv::Mat image, int x)
{
  cv::Mat copy = image.clone();
  cv::Mat halftone = image.clone();
  cv::Scalar intensity;
  int error;

  for(int i=0; i<image.rows; i++)
  {
    error = 0;
    for(int j=0; j<image.cols; j++)
    {
      intensity = image.at<cv::Vec3b>(i,j)[x];
      intensity[0] += error;

      if (intensity[0] < 128)
      {
        halftone.at<cv::Vec3b>(i,j)[x] = 0;
        error = intensity[0];
      }
      else
      {
        halftone.at<cv::Vec3b>(i,j)[x] = 255;
        error = intensity[0] - 255;
      }
    }
  }
  return halftone;
}


/*function that takes cmyk shares as input and returns
  combined share - order of input vector is CMYK (K = mask)*/
cv::Mat combineCMYKshares(std::vector<cv::Mat> &shares)
{
  cv::Mat combShare(shares[0].rows, shares[0].cols, CV_8UC3, cv::Scalar(255,255,255));

  /* Combine shares */
  for(int i=0; i<shares[0].rows; i++)
  {
    for(int j=0; j<shares[0].cols; j++)
    {
      if (shares[3].at<uchar>(i,j)==0)
      {
        combShare.at<cv::Vec3b>(i,j)[0] = 128; //128 used here as using full
        combShare.at<cv::Vec3b>(i,j)[1] = 128; //black makes image very dark
        combShare.at<cv::Vec3b>(i,j)[2] = 128;
      }
      else
      {
        combShare.at<cv::Vec3b>(i,j)[0] = 255 - shares[2].at<cv::Vec3b>(i,j)[0];
        combShare.at<cv::Vec3b>(i,j)[1] = 255 - shares[1].at<cv::Vec3b>(i,j)[1];
        combShare.at<cv::Vec3b>(i,j)[2] = 255 - shares[0].at<cv::Vec3b>(i,j)[2];
      }
    }
  }
  return combShare;
}


/*function that takes an image as input and returns the
  CMYK shares as a vector */
std::vector<cv::Mat> CMYKScheme(cv::Mat img)
{
  /*Split image into BGR components*/
  std::vector<cv::Mat> bgr;
  cv::split(img, bgr);

  cv::Mat blue = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
  cv::Mat green = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
  cv::Mat red = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

  for (int i=0; i<img.rows; i++)
  {
    for (int j=0; j<img.cols; j++)
    {
      blue.at<cv::Vec3b>(i,j)[0] = img.at<cv::Vec3b>(i,j)[0];
      green.at<cv::Vec3b>(i,j)[1] = img.at<cv::Vec3b>(i,j)[1];
      red.at<cv::Vec3b>(i,j)[2] = img.at<cv::Vec3b>(i,j)[2];
    }
  }

  /*Convert to CMYK*/
  cv::Mat cyan = cv::Mat::zeros(img.rows, img.cols, CV_8UC3); //(255,255,0)
  cv::Mat magenta = cv::Mat::zeros(img.rows, img.cols, CV_8UC3); //(255,0,255)
  cv::Mat yellow = cv::Mat::zeros(img.rows, img.cols, CV_8UC3); //(0,255,255)

  for(int i=0; i<img.rows; i++)
  {
    for(int j=0; j<img.cols; j++)
    {
      for(int k=0; k<3; k++)
      {
        cyan.at<cv::Vec3b>(i,j)[k] = 255 - red.at<cv::Vec3b>(i,j)[k];
        magenta.at<cv::Vec3b>(i,j)[k] = 255 - green.at<cv::Vec3b>(i,j)[k];
        yellow.at<cv::Vec3b>(i,j)[k] = 255 - blue.at<cv::Vec3b>(i,j)[k];
      }
    }
  }

  /*Convert to Halftone*/
  cv::Mat cyan_halftone = convertToColourHalftone(cyan, 2);
  cv::Mat magenta_halftone = convertToColourHalftone(magenta, 1);
  cv::Mat yellow_halftone = convertToColourHalftone(yellow, 0);

  /*Initialise cyan, magenta, yellow and mask shares*/
  cv::Mat cyanShare(2*img.rows, 2*img.cols, CV_8UC3, cv::Scalar(255,255,255));
  cv::Mat yellowShare(2*img.rows, 2*img.cols, CV_8UC3, cv::Scalar(255,255,255));
  cv::Mat magentaShare(2*img.rows, 2*img.cols, CV_8UC3, cv::Scalar(255,255,255));
  cv::Mat mask(2*img.rows, 2*img.cols, CV_8UC1, cv::Scalar(255,255,255));

  srand(time(NULL));
  int mask_chooser;

  /* Generate mask */
  for(int i=0; i<2*img.rows; i++)
  {
    for(int j=0; j<2*img.cols; j=j+2)
    {
      mask_chooser = rand() % 2;
      mask.at<uchar>(i, j+mask_chooser) = 0;
    }
  }

  /* Generate cyan share */
  for(int i=0; i<img.rows; i++)
  {
    for(int j=0; j<img.cols; j++)
    {
      if (cyan_halftone.at<cv::Vec3b>(i,j)[2] == 0) //cyan pixel
      {
        if (mask.at<uchar>(2*i,2*j)==255)
        {
          cyanShare.at<cv::Vec3b>(2*i,2*j)[2]=0;
        }
        else
        {
          cyanShare.at<cv::Vec3b>(2*i,2*j+1)[2]=0;
        }
      }
      else //white pixel
      {
        if (mask.at<uchar>(2*i,2*j)==0)
        {
          cyanShare.at<cv::Vec3b>(2*i,2*j)[2]=0;
        }
        else
        {
          cyanShare.at<cv::Vec3b>(2*i,2*j+1)[2]=0;
        }
      }
    }
  }

  /* Generate magenta share */
  for(int i=0; i<img.rows; i++)
  {
    for(int j=0; j<img.cols; j++)
    {
      if (magenta_halftone.at<cv::Vec3b>(i,j)[1] == 0) //magenta pixel
      {
        if (mask.at<uchar>(2*i,2*j)==255)
        {
          magentaShare.at<cv::Vec3b>(2*i,2*j)[1]=0;
        }
        else
        {
          magentaShare.at<cv::Vec3b>(2*i,2*j+1)[1]=0;
        }
      }
      else //white pixel
      {
        if (mask.at<uchar>(2*i,2*j)==0)
        {
          magentaShare.at<cv::Vec3b>(2*i,2*j)[1]=0;
        }
        else
        {
          magentaShare.at<cv::Vec3b>(2*i,2*j+1)[1]=0;
        }
      }
    }
  }

  /* Generate yellow share */
  for(int i=0; i<img.rows; i++)
  {
    for(int j=0; j<img.cols; j++)
    {
      if (yellow_halftone.at<cv::Vec3b>(i,j)[0] == 0) //yellow pixel
      {
        if (mask.at<uchar>(2*i,2*j)==255)
        {
          yellowShare.at<cv::Vec3b>(2*i,2*j)[0]=0;
        }
        else
        {
          yellowShare.at<cv::Vec3b>(2*i,2*j+1)[0]=0;
        }
      }
      else //white pixel
      {
        if (mask.at<uchar>(2*i,2*j)==0)
        {
          yellowShare.at<cv::Vec3b>(2*i,2*j)[0]=0;
        }
        else
        {
          yellowShare.at<cv::Vec3b>(2*i,2*j+1)[0]=0;
        }
      }
    }
  }

  std::vector<cv::Mat> colouredShares;
  colouredShares.push_back(cyanShare);
  colouredShares.push_back(magentaShare);
  colouredShares.push_back(yellowShare);
  colouredShares.push_back(mask);

  return colouredShares;
}

/*functions that takes a black and white reconstructed image as
  input, checks to see whether each group of m pixels is over the
  threshold l and fixes them - i.e. if under threshold make them
  white, if over the threshold make them black */
cv::Mat fixCombinedShare(cv::Mat image, int m, int l)
{
  int temp;
  cv::Mat reconstruction(image.rows, image.cols, CV_8UC1, cv::Scalar(255,255,255));
  for (int i=0; i<image.rows; i++)
  {
    for (int j=0; j<image.cols-m; j+=m)
    {
      temp = 0;
      for (int x=0; x<m; x++)
      {
        if (image.at<uchar>(i, j+x) == 0)
        {
          temp++;
        }
      }
      if (temp>l)
      {
        for (int x=0; x<m; x++)
        {
          reconstruction.at<uchar>(i,j+x)=0;
        }
      }
      else
      {
        for (int x=0; x<m; x++)
        {
          reconstruction.at<uchar>(i,j+x)=255;
        }
      }
    }
  }
  return reconstruction;
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    for(int i=2; i<5; i++)
    {
        ui->threshold->addItem(QString::number(i)); //k - threshold
        ui->numShares->addItem(QString::number(i)); //n - no. of shares
    }

    /*seed random generator*/
    std::srand(time(0));

    /*set all widgets to invisible*/
    ui->InputImage->setVisible(false);
    ui->OutputImage->setVisible(false);
    ui->secretImageText->setVisible(false);
    ui->reconImageText->setVisible(false);
    ui->thresholdText->setVisible(false);
    ui->threshold->setVisible(false);
    ui->numSharesText->setVisible(false);
    ui->numShares->setVisible(false);
    ui->BlackWhiteButton->setVisible(false);
    ui->GreyButton->setVisible(false);
    ui->ColourButton->setVisible(false);
    ui->BlackWhiteEg->setVisible(false);
    ui->GreyEg->setVisible(false);
    ui->ColourEg->setVisible(false);
    ui->Share1Pic->setVisible(false);
    ui->Share1Check->setVisible(false);
    ui->Share2Pic->setVisible(false);
    ui->Share2Check->setVisible(false);
    ui->Share3Pic->setVisible(false);
    ui->Share3Check->setVisible(false);
    ui->Share4Pic->setVisible(false);
    ui->Share4Check->setVisible(false);
    ui->SchemeInfo->setVisible(false);
    ui->combineSharesCheck->setVisible(false);
    ui->selectSharesInfo->setVisible(false);
    ui->nextButton->setVisible(false);
    ui->previousButton->setVisible(false);
    ui->fixImageCheck->setVisible(false);


    /*set all buttons excluding load image to disabled*/
    ui->ChooseScheme->setDisabled(true);
    ui->SetParameters->setDisabled(true);
    ui->GenerateShares->setDisabled(true);
    ui->CombineShares->setDisabled(true);
    ui->SaveToFile->setDisabled(true);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_LoadImage_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "",
        tr("JPEG (*.jpg);;PNG (*.png);;Bitmap (*.bmp)"));

    img = cv::imread(fileName.toStdString());
    if(!img.data)
    {
        //if no image - do nothing
    }
    else
    {
        /*set input image to visible (i.e. display picture) and all other
        * widgets to invisible*/
        ui->InputImage->setVisible(true);
        ui->OutputImage->setVisible(false);
        ui->secretImageText->setVisible(true);
        ui->reconImageText->setVisible(false);
        ui->thresholdText->setVisible(false);
        ui->threshold->setVisible(false);
        ui->numSharesText->setVisible(false);
        ui->numShares->setVisible(false);
        ui->BlackWhiteButton->setVisible(false);
        ui->GreyButton->setVisible(false);
        ui->ColourButton->setVisible(false);
        ui->BlackWhiteEg->setVisible(false);
        ui->GreyEg->setVisible(false);
        ui->ColourEg->setVisible(false);
        ui->Share1Pic->setVisible(false);
        ui->Share1Check->setVisible(false);
        ui->Share2Pic->setVisible(false);
        ui->Share2Check->setVisible(false);
        ui->Share3Pic->setVisible(false);
        ui->Share3Check->setVisible(false);
        ui->Share4Pic->setVisible(false);
        ui->Share4Check->setVisible(false);
        ui->SchemeInfo->setVisible(false);
        ui->combineSharesCheck->setVisible(false);
        ui->selectSharesInfo->setVisible(false);
        ui->nextButton->setVisible(false);
        ui->previousButton->setVisible(false);
        ui->fixImageCheck->setVisible(false);


        /*create black & white and grey copies of image*/
        cvtColor(img, img_g, CV_BGR2GRAY);
        img_bw = img_g > 128; //black and white

        /*convert images to Qt format*/
        cvtColor(img, img, CV_BGR2RGB);
        QPixmap original = QPixmap::fromImage(QImage(img.data,img.cols,img.rows,img.step,
                                                     QImage::Format_RGB888));

        int w = ui->InputImage->width();
        int h = ui->InputImage->height();
        ui->InputImage->setPixmap(original.scaled(w,h,Qt::KeepAspectRatio));

        /*disable all buttons excluding choose scheme & load image*/
        ui->ChooseScheme->setDisabled(false);
        ui->SetParameters->setDisabled(true);
        ui->GenerateShares->setDisabled(true);
        ui->CombineShares->setDisabled(true);
        ui->SaveToFile->setDisabled(true);
    }
}


void MainWindow::on_ChooseScheme_clicked()
{
    /*disable future buttons except set parameters*/
    ui->ChooseScheme->setDisabled(true);
    ui->GenerateShares->setDisabled(true);
    ui->CombineShares->setDisabled(true);
    ui->SaveToFile->setDisabled(true);
    ui->SetParameters->setDisabled(false);

    /*set radio buttons visibility for choosing scheme to true*/
    ui->BlackWhiteButton->setVisible(true);
    ui->GreyButton->setVisible(true);
    ui->ColourButton->setVisible(true);

    /*set other widgets visibility to false*/
    ui->InputImage->setVisible(false);
    ui->OutputImage->setVisible(false);
    ui->secretImageText->setVisible(false);
    ui->reconImageText->setVisible(false);
    ui->thresholdText->setVisible(false);
    ui->threshold->setVisible(false);
    ui->numSharesText->setVisible(false);
    ui->numShares->setVisible(false);
    ui->Share1Pic->setVisible(false);
    ui->Share1Check->setVisible(false);
    ui->Share2Pic->setVisible(false);
    ui->Share2Check->setVisible(false);
    ui->Share3Pic->setVisible(false);
    ui->Share3Check->setVisible(false);
    ui->Share4Pic->setVisible(false);
    ui->Share4Check->setVisible(false);
    ui->SchemeInfo->setVisible(false);
    ui->combineSharesCheck->setVisible(false);
    ui->selectSharesInfo->setVisible(false);
    ui->nextButton->setVisible(false);
    ui->previousButton->setVisible(false);
    ui->fixImageCheck->setVisible(false);
}


void MainWindow::on_SetParameters_clicked()
{
    shares.clear();
    sharesArray.clear();
    if(ui->ColourButton->isChecked())
    {
        ui->ChooseScheme->setDisabled(false);
        ui->SetParameters->setDisabled(true);
        ui->GenerateShares->setDisabled(false);

        scheme = 2; //colour scheme

        ui->BlackWhiteButton->setVisible(false);
        ui->GreyButton->setVisible(false);
        ui->ColourButton->setVisible(false);
        ui->InputImage->setVisible(false);
        ui->secretImageText->setVisible(false);
        ui->reconImageText->setVisible(false);

        ui->thresholdText->setVisible(true);
        ui->threshold->setVisible(true);
        ui->numSharesText->setVisible(true);
        ui->numShares->setVisible(true);

        ui->Share1Pic->setVisible(false);
        ui->Share1Check->setVisible(false);
        ui->Share2Pic->setVisible(false);
        ui->Share2Check->setVisible(false);
        ui->Share3Pic->setVisible(false);
        ui->Share3Check->setVisible(false);
        ui->Share4Pic->setVisible(false);
        ui->Share4Check->setVisible(false);
        ui->OutputImage->setVisible(false);
        ui->combineSharesCheck->setVisible(false);
        ui->selectSharesInfo->setVisible(false);
        ui->SchemeInfo->setVisible(true);        
        ui->SchemeInfo->setText("Colour scheme is (4,4)");

        ui->thresholdText->setDisabled(true);
        ui->threshold->setDisabled(true);
        ui->numSharesText->setDisabled(true);
        ui->numShares->setDisabled(true);

        ui->nextButton->setVisible(false);
        ui->previousButton->setVisible(false);
        ui->fixImageCheck->setVisible(false);
    }

    else if(ui->BlackWhiteButton->isChecked() || ui->GreyButton->isChecked())
    {
        ui->ChooseScheme->setDisabled(false);
        ui->SetParameters->setDisabled(true);
        ui->GenerateShares->setDisabled(false);

        if(ui->BlackWhiteButton->isChecked())
        {
            scheme = 0; //black & white scheme
        }
        else if(ui->GreyButton->isChecked())
        {
            scheme = 1; //black & white scheme
        }

        ui->BlackWhiteButton->setVisible(false);
        ui->GreyButton->setVisible(false);
        ui->ColourButton->setVisible(false);

        ui->thresholdText->setVisible(true);
        ui->threshold->setVisible(true);
        ui->numSharesText->setVisible(true);
        ui->numShares->setVisible(true);
        ui->InputImage->setVisible(false);
        ui->secretImageText->setVisible(false);
        ui->reconImageText->setVisible(false);

        ui->thresholdText->setDisabled(false);
        ui->threshold->setDisabled(false);
        ui->numSharesText->setDisabled(false);
        ui->numShares->setDisabled(false);

        ui->Share1Pic->setVisible(false);
        ui->Share1Check->setVisible(false);
        ui->Share2Pic->setVisible(false);
        ui->Share2Check->setVisible(false);
        ui->Share3Pic->setVisible(false);
        ui->Share3Check->setVisible(false);
        ui->Share4Pic->setVisible(false);
        ui->Share4Check->setVisible(false);
        ui->combineSharesCheck->setVisible(false);
        ui->selectSharesInfo->setVisible(false);
        ui->OutputImage->setVisible(false);

        ui->nextButton->setVisible(false);
        ui->previousButton->setVisible(false);
    }

    else //no scheme selected
    {
        QMessageBox NoSchemeChosen;
        NoSchemeChosen.setText("Please choose a scheme");
        NoSchemeChosen.exec();
    }
}


void MainWindow::on_GenerateShares_clicked()
{
    shares.clear();
    sharesArray.clear();
    int k = (ui->threshold->currentText().toInt());
    int n = (ui->numShares->currentText().toInt());
    if (k>n)
    {
        QMessageBox invalidParameters;
        invalidParameters.setText("k must be less than or equal to n");
        invalidParameters.exec();
    }
    else
    {
        ui->GenerateShares->setDisabled(true);
        ui->SetParameters->setDisabled(false);
        ui->CombineShares->setDisabled(false);
        ui->SchemeInfo->setVisible(false);

        ui->thresholdText->setVisible(false);
        ui->threshold->setVisible(false);
        ui->numSharesText->setVisible(false);
        ui->numShares->setVisible(false);
        ui->InputImage->setVisible(false);
        ui->OutputImage->setVisible(false);
        ui->secretImageText->setVisible(false);
        ui->reconImageText->setVisible(false);

        ui->Share1Pic->setVisible(true);
        ui->Share1Check->setVisible(true);
        ui->Share2Pic->setVisible(true);
        ui->Share2Check->setVisible(true);
        ui->combineSharesCheck->setVisible(true);
        ui->selectSharesInfo->setVisible(true);

        ui->Share1Check->setChecked(false);
        ui->Share2Check->setChecked(false);
        ui->Share3Check->setChecked(false);
        ui->Share4Check->setChecked(false);

        ui->nextButton->setVisible(false);
        ui->previousButton->setVisible(false);
        ui->fixImageCheck->setVisible(false);

        /*width and height of input image, for displaying shares*/
        int w = ui->InputImage->width();
        int h = ui->InputImage->height();

        if(scheme==0) //Black & white scheme
        {
            if (k == 2)
            {
                shares = TwoByNscheme(img_bw, n);
                for(int i=0; i<n; i++)
                {
                    sharesArray.push_back(QImage((uchar*) shares[i].data,shares[i].cols,shares[i].rows,
                    shares[i].step,QImage::Format_Grayscale8));
                }

                /*display generated shares*/
                ui->Share1Pic->setPixmap(QPixmap::fromImage(sharesArray[0]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share2Pic->setPixmap(QPixmap::fromImage(sharesArray[1]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                if (n == 3)
                {
                    ui->Share3Pic->setVisible(true);
                    ui->Share3Check->setVisible(true);
                    ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                }
                else if (n == 4)
                {
                    ui->Share3Pic->setVisible(true);
                    ui->Share3Check->setVisible(true);
                    ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                    ui->Share4Pic->setVisible(true);
                    ui->Share4Check->setVisible(true);
                    ui->Share4Pic->setPixmap(QPixmap::fromImage(sharesArray[3]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                }
            }
            else if (n == k)
            {
                shares = KbyKscheme(img_bw, k);
                for(int i=0; i<n; i++)
                {
                    sharesArray.push_back(QImage((uchar*) shares[i].data,shares[i].cols,shares[i].rows,
                    shares[i].step,QImage::Format_Grayscale8));
                }
                ui->Share1Pic->setPixmap(QPixmap::fromImage(sharesArray[0]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share2Pic->setPixmap(QPixmap::fromImage(sharesArray[1]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share3Pic->setVisible(true);
                ui->Share3Check->setVisible(true);
                ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                if (n == 4)
                {
                    ui->Share4Pic->setVisible(true);
                    ui->Share4Check->setVisible(true);
                    ui->Share4Pic->setPixmap(QPixmap::fromImage(sharesArray[3]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                }
            }
            else if (k==3 && n==4)
            {
                shares = ThreeByFourScheme(img_bw);
                for(int i=0; i<n; i++)
                {
                    sharesArray.push_back(QImage((uchar*) shares[i].data,shares[i].cols,shares[i].rows,
                    shares[i].step,QImage::Format_Grayscale8));
                }
                ui->Share1Pic->setPixmap(QPixmap::fromImage(sharesArray[0]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share2Pic->setPixmap(QPixmap::fromImage(sharesArray[1]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share3Pic->setVisible(true);
                ui->Share3Check->setVisible(true);
                ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share4Pic->setVisible(true);
                ui->Share4Check->setVisible(true);
                ui->Share4Pic->setPixmap(QPixmap::fromImage(sharesArray[3]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
            }
        }

        else if(scheme==1) //Grey halftone scheme
        {

            img_g = convertToHalftone(img_g);
            if (k == 2)
            {
                shares = TwoByNscheme(img_g, n);
                for(int i=0; i<n; i++)
                {
                    sharesArray.push_back(QImage((uchar*) shares[i].data,shares[i].cols,shares[i].rows,
                    shares[i].step,QImage::Format_Grayscale8));
                }

                /*display generated shares*/
                ui->Share1Pic->setPixmap(QPixmap::fromImage(sharesArray[0]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share2Pic->setPixmap(QPixmap::fromImage(sharesArray[1]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                if (n == 3)
                {
                    ui->Share3Pic->setVisible(true);
                    ui->Share3Check->setVisible(true);
                    ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                }
                else if (n == 4)
                {
                    ui->Share3Pic->setVisible(true);
                    ui->Share3Check->setVisible(true);
                    ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                    ui->Share4Pic->setVisible(true);
                    ui->Share4Check->setVisible(true);
                    ui->Share4Pic->setPixmap(QPixmap::fromImage(sharesArray[3]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                }
            }
            else if (n == k)
            {
                shares = KbyKscheme(img_g, k);
                for(int i=0; i<n; i++)
                {
                    sharesArray.push_back(QImage((uchar*) shares[i].data,shares[i].cols,shares[i].rows,
                    shares[i].step,QImage::Format_Grayscale8));
                }
                ui->Share1Pic->setPixmap(QPixmap::fromImage(sharesArray[0]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share2Pic->setPixmap(QPixmap::fromImage(sharesArray[1]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share3Pic->setVisible(true);
                ui->Share3Check->setVisible(true);
                ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                if (n == 4)
                {
                    ui->Share4Pic->setVisible(true);
                    ui->Share4Check->setVisible(true);
                    ui->Share4Pic->setPixmap(QPixmap::fromImage(sharesArray[3]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                }
            }
            else if (k==3 && n==4)
            {
                shares = ThreeByFourScheme(img_g);
                for(int i=0; i<n; i++)
                {
                    sharesArray.push_back(QImage((uchar*) shares[i].data,shares[i].cols,shares[i].rows,
                    shares[i].step,QImage::Format_Grayscale8));
                }
                ui->Share1Pic->setPixmap(QPixmap::fromImage(sharesArray[0]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share2Pic->setPixmap(QPixmap::fromImage(sharesArray[1]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share3Pic->setVisible(true);
                ui->Share3Check->setVisible(true);
                ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
                ui->Share4Pic->setVisible(true);
                ui->Share4Check->setVisible(true);
                ui->Share4Pic->setPixmap(QPixmap::fromImage(sharesArray[3]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
            }
        }

        else if(scheme==2) //Colour scheme - only (4,4)
        {
            k=4;
            n=4;
            shares = CMYKScheme(img);
            for(int i=0; i<3; i++)
            {
                sharesArray.push_back(QImage((uchar*) shares[i].data,shares[i].cols,shares[i].rows,
                shares[i].step,QImage::Format_RGB888));
            }
            sharesArray.push_back(QImage((uchar*) shares[3].data,shares[3].cols,shares[3].rows,
                shares[3].step,QImage::Format_Grayscale8));
            ui->Share3Pic->setVisible(true);
            ui->Share3Check->setVisible(true);
            ui->Share4Pic->setVisible(true);
            ui->Share4Check->setVisible(true);
            ui->Share1Pic->setPixmap(QPixmap::fromImage(sharesArray[0]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
            ui->Share2Pic->setPixmap(QPixmap::fromImage(sharesArray[1]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
            ui->Share3Pic->setPixmap(QPixmap::fromImage(sharesArray[2]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
            ui->Share4Pic->setPixmap(QPixmap::fromImage(sharesArray[3]).scaled(w,h,Qt::KeepAspectRatio,Qt::SmoothTransformation));
        }
    }
}


void MainWindow::on_CombineShares_clicked()
{
    if (ui->combineSharesCheck->isChecked()) //combine shares in order
    {
        /*buttons*/
        ui->CombineShares->setDisabled(true);
        ui->GenerateShares->setDisabled(false);
        ui->SaveToFile->setDisabled(false);
        ui->nextButton->setDisabled(false);
        ui->previousButton->setDisabled(true);

        /*widgets*/
        ui->Share1Pic->setVisible(false);
        ui->Share1Check->setVisible(false);
        ui->Share2Pic->setVisible(false);
        ui->Share2Check->setVisible(false);
        ui->Share3Pic->setVisible(false);
        ui->Share3Check->setVisible(false);
        ui->Share4Pic->setVisible(false);
        ui->Share4Check->setVisible(false);
        ui->combineSharesCheck->setVisible(false);
        ui->selectSharesInfo->setVisible(false);
        ui->nextButton->setVisible(true);
        ui->previousButton->setVisible(true);

        /*vector that stores different combine shares*/
        combinedShares.clear();
        combinedShares.push_back(shares[0].clone()); //first share

        if (scheme == 2) //colour scheme
        {
            cv::Mat Blankshare(shares[0].rows, shares[0].cols, CV_8UC1, cv::Scalar(255,255,255));
            std::vector<cv::Mat> tempShares;

            //Add first 2 shares and 2 blank shares to temp array
            tempShares.push_back(shares[0].clone());
            tempShares.push_back(shares[1].clone());
            tempShares.push_back(Blankshare.clone());
            tempShares.push_back(Blankshare.clone());

            //Subtract 255 off each BGR value for first 3 shares
            for (unsigned int x=0; x<shares.size()-1; x++)
            {
                for(int i=0; i<shares[0].rows; i++)
                {
                    for (int j=0; j<shares[0].cols; j++)
                    {
                        tempShares[x].at<cv::Vec3b>(i,j)[0] = 255 - tempShares[x].at<cv::Vec3b>(i,j)[0];
                        tempShares[x].at<cv::Vec3b>(i,j)[1] = 255 - tempShares[x].at<cv::Vec3b>(i,j)[1];
                        tempShares[x].at<cv::Vec3b>(i,j)[2] = 255 - tempShares[x].at<cv::Vec3b>(i,j)[2];
                    }
                }
            }
            combShare = combineCMYKshares(tempShares); //Add first two shares combined to array
            combinedShares.push_back(combShare);

            //Clear Array then add first 3 shares and a blankshare to temp array
            tempShares.clear();
            tempShares.push_back(shares[0].clone());
            tempShares.push_back(shares[1].clone());
            tempShares.push_back(shares[2].clone());
            tempShares.push_back(Blankshare.clone());

            //Subtract 255 off each BGR value for first 3 shares
            for (unsigned int x=0; x<shares.size()-1; x++)
            {
                for(int i=0; i<shares[0].rows; i++)
                {
                    for (int j=0; j<shares[0].cols; j++)
                    {
                        tempShares[x].at<cv::Vec3b>(i,j)[0] = 255 - tempShares[x].at<cv::Vec3b>(i,j)[0];
                        tempShares[x].at<cv::Vec3b>(i,j)[1] = 255 - tempShares[x].at<cv::Vec3b>(i,j)[1];
                        tempShares[x].at<cv::Vec3b>(i,j)[2] = 255 - tempShares[x].at<cv::Vec3b>(i,j)[2];
                    }
                }
            }
            combShare = combineCMYKshares(tempShares); //Add first three shares combined to array
            combinedShares.push_back(combShare);

            combShare = combineCMYKshares(shares); //Add all combined shares to array
            combinedShares.push_back(combShare);

            count = 0;

            //display first share
            QImage QcombShare = QImage((uchar*) combinedShares[0].data, combinedShares[0].cols,
                    combinedShares[0].rows, combinedShares[0].step, QImage::Format_RGB888);
            QPixmap QShare = QPixmap::fromImage(QcombShare);
            ui->OutputImage->setVisible(true);
            int w = ui->InputImage->width();
            int h = ui->InputImage->height();
            ui->OutputImage->setPixmap(QShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));
            ui->InputImage->setVisible(true);
            ui->secretImageText->setVisible(true);
            ui->reconImageText->setVisible(true);

        }

        else //black & white and grey schemes
        {
            for (unsigned int i=1; i<shares.size(); i++)
            {
                combinedShares.push_back(shares[i].clone());
                combinedShares[i] = combineShares(combinedShares);
            }

            count = 0;

            //display first share
            QImage QcombShare = QImage((uchar*) combinedShares[0].data, combinedShares[0].cols,
                    combinedShares[0].rows, combinedShares[0].step, QImage::Format_Grayscale8);
            QPixmap QShare = QPixmap::fromImage(QcombShare);
            ui->OutputImage->setVisible(true);
            int w = ui->InputImage->width();
            int h = ui->InputImage->height();
            ui->OutputImage->setPixmap(QShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));
            ui->InputImage->setVisible(true);
            ui->secretImageText->setVisible(true);
            ui->reconImageText->setVisible(true);

        }
    }

    else //combine selected shares
    {
        std::vector<cv::Mat> selectedShares; //stores only shares that are selected

        if (scheme == 2) //colour scheme
        {
            //Add blankshares instead of CMYK shares where shares aren't selected
            cv::Mat Blankshare(shares[0].rows, shares[0].cols, CV_8UC1, cv::Scalar(255,255,255));
            int blankshareCount = 0;
            if (ui->Share1Check->isChecked()) //Add selected shares to vector
            {
                selectedShares.push_back(shares[0].clone());
            }
            else
            {
                selectedShares.push_back(Blankshare.clone());
                blankshareCount++;
            }

            if (ui->Share2Check->isChecked())
            {
                selectedShares.push_back(shares[1].clone());
            }
            else
            {
                selectedShares.push_back(Blankshare.clone());
                blankshareCount++;
            }

            if (ui->Share3Check->isChecked())
            {
                selectedShares.push_back(shares[2].clone());
            }
            else
            {
                selectedShares.push_back(Blankshare.clone());
                blankshareCount++;
            }

            if (ui->Share4Check->isChecked())
            {
                selectedShares.push_back(shares[3].clone());
            }
            else
            {
                selectedShares.push_back(Blankshare.clone());
                blankshareCount++;
            }

            if (blankshareCount == 4) //if no shares selected output blankshare
            {
                QMessageBox NoSharesSelected;
                NoSharesSelected.setText("No Shares Selected");
                NoSharesSelected.exec();
            }
            else if (blankshareCount == 0) //if no blankshares combine CMYK shares as normal
            {
                /*buttons*/
                ui->CombineShares->setDisabled(true);
                ui->GenerateShares->setDisabled(false);
                ui->SaveToFile->setDisabled(false);

                /*widgets*/
                ui->Share1Pic->setVisible(false);
                ui->Share1Check->setVisible(false);
                ui->Share2Pic->setVisible(false);
                ui->Share2Check->setVisible(false);
                ui->Share3Pic->setVisible(false);
                ui->Share3Check->setVisible(false);
                ui->Share4Pic->setVisible(false);
                ui->Share4Check->setVisible(false);
                ui->combineSharesCheck->setVisible(false);
                ui->selectSharesInfo->setVisible(false);

                combShare = combineCMYKshares(selectedShares);
                cv::cvtColor(combShare, combShare, CV_BGR2RGB);
                QImage QcombShare = QImage((uchar*) combShare.data, combShare.cols, combShare.rows,
                                           combShare.step, QImage::Format_RGB888);
                QPixmap QQCombShare = QPixmap::fromImage(QcombShare);
                ui->OutputImage->setVisible(true);
                int w = ui->InputImage->width();
                int h = ui->InputImage->height();
                ui->OutputImage->setPixmap(QQCombShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));
                ui->InputImage->setVisible(true);
                ui->secretImageText->setVisible(true);
                ui->reconImageText->setVisible(true);

            }
            else //at least one blank shares, need to subtract 255 from each BGR value of each share so that CMYK functions correctly
            {
                /*buttons*/
                ui->CombineShares->setDisabled(true);
                ui->GenerateShares->setDisabled(false);
                ui->SaveToFile->setDisabled(false);

                /*widgets*/
                ui->Share1Pic->setVisible(false);
                ui->Share1Check->setVisible(false);
                ui->Share2Pic->setVisible(false);
                ui->Share2Check->setVisible(false);
                ui->Share3Pic->setVisible(false);
                ui->Share3Check->setVisible(false);
                ui->Share4Pic->setVisible(false);
                ui->Share4Check->setVisible(false);
                ui->secretImageText->setVisible(true);
                ui->reconImageText->setVisible(true);
                ui->combineSharesCheck->setVisible(false);
                ui->selectSharesInfo->setVisible(false);

                for (unsigned int x=0; x<selectedShares.size()-1; x++)
                {
                    for(int i=0; i<selectedShares[0].rows; i++)
                    {
                        for (int j=0; j<selectedShares[0].cols; j++)
                        {
                            selectedShares[x].at<cv::Vec3b>(i,j)[0] = 255 - selectedShares[x].at<cv::Vec3b>(i,j)[0];
                            selectedShares[x].at<cv::Vec3b>(i,j)[1] = 255 - selectedShares[x].at<cv::Vec3b>(i,j)[1];
                            selectedShares[x].at<cv::Vec3b>(i,j)[2] = 255 - selectedShares[x].at<cv::Vec3b>(i,j)[2];
                        }
                    }
                }

                combShare = combineCMYKshares(selectedShares);
                QImage QcombShare = QImage((uchar*) combShare.data, combShare.cols, combShare.rows,
                                           combShare.step, QImage::Format_RGB888);
                QPixmap QQCombShare = QPixmap::fromImage(QcombShare);
                ui->OutputImage->setVisible(true);
                int w = ui->InputImage->width();
                int h = ui->InputImage->height();
                ui->OutputImage->setPixmap(QQCombShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));
                ui->InputImage->setVisible(true);
            }

        }
        else //B&W or Greyscale scheme
        {
            //Blankshare displayed if no shares selected
            cv::Mat Blankshare(shares[0].rows, shares[0].cols, CV_8UC1, cv::Scalar(255,255,255));
            unsigned int blankshareCount = 0;
            if (ui->Share1Check->isChecked()) //Add selected shares to vector
            {
                selectedShares.push_back(shares[0].clone());
            }
            else
            {
                blankshareCount++;
            }

            if (ui->Share2Check->isChecked())
            {
                selectedShares.push_back(shares[1].clone());
            }
            else
            {
                blankshareCount++;
            }

            if (ui->Share3Check->isChecked())
            {
                selectedShares.push_back(shares[2].clone());
            }
            else if (shares.size()>2 && !(ui->Share3Check->isChecked()))
            {
                blankshareCount++;
            }

            if (ui->Share4Check->isChecked())
            {
                selectedShares.push_back(shares[3].clone());
            }
            else if (shares.size()>3 && !(ui->Share4Check->isChecked()))
            {
                blankshareCount++;
            }

            if (blankshareCount == shares.size())
            {
                QMessageBox NoSharesSelected;
                NoSharesSelected.setText("No Shares Selected");
                NoSharesSelected.exec();
            }

            else //else display selected shares
            {
                /*buttons*/
                ui->CombineShares->setDisabled(true);
                ui->GenerateShares->setDisabled(false);
                ui->SaveToFile->setDisabled(false);

                /*widgets*/
                ui->Share1Pic->setVisible(false);
                ui->Share1Check->setVisible(false);
                ui->Share2Pic->setVisible(false);
                ui->Share2Check->setVisible(false);
                ui->Share3Pic->setVisible(false);
                ui->Share3Check->setVisible(false);
                ui->Share4Pic->setVisible(false);
                ui->Share4Check->setVisible(false);
                ui->secretImageText->setVisible(true);
                ui->reconImageText->setVisible(true);
                ui->combineSharesCheck->setVisible(false);
                ui->selectSharesInfo->setVisible(false);
                //ui->fixImageCheck->setVisible(true);


                combShare = combineShares(selectedShares);
                QImage QcombShare = QImage((uchar*) combShare.data, combShare.cols, combShare.rows,
                                           combShare.step, QImage::Format_Grayscale8);
                QPixmap QQCombShare = QPixmap::fromImage(QcombShare);
                ui->OutputImage->setVisible(true);
                int w = ui->InputImage->width();
                int h = ui->InputImage->height();
                ui->OutputImage->setPixmap(QQCombShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));
                ui->InputImage->setVisible(true);

            }

        }
    }
}


void MainWindow::on_nextButton_clicked()
{
    count++;
    QImage QcombShare;
    if (scheme == 2)
    {
        QcombShare = QImage((uchar*) combinedShares[count].data, combinedShares[count].cols,
                            combinedShares[count].rows, combinedShares[count].step, QImage::Format_RGB888);
    }

    else
    {
        QcombShare = QImage((uchar*) combinedShares[count].data, combinedShares[count].cols,
                                   combinedShares[count].rows, combinedShares[count].step, QImage::Format_Grayscale8);
    }
    QPixmap QShare = QPixmap::fromImage(QcombShare);
    int w = ui->InputImage->width();
    int h = ui->InputImage->height();
    ui->OutputImage->setPixmap(QShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));

    ui->previousButton->setDisabled(false);

    if(count == shares.size()-1)
    {
        ui->nextButton->setDisabled(true);
        //ui->fixImageCheck->setVisible(true);
    }
}


void MainWindow::on_previousButton_clicked()
{
    count--;
    QImage QcombShare;
    if (scheme == 2)
    {
        QcombShare = QImage((uchar*) combinedShares[count].data, combinedShares[count].cols,
                            combinedShares[count].rows, combinedShares[count].step, QImage::Format_RGB888);
    }

    else
    {
        QcombShare = QImage((uchar*) combinedShares[count].data, combinedShares[count].cols,
                                   combinedShares[count].rows, combinedShares[count].step, QImage::Format_Grayscale8);
    }
    QPixmap QShare = QPixmap::fromImage(QcombShare);
    int w = ui->InputImage->width();
    int h = ui->InputImage->height();
    ui->OutputImage->setPixmap(QShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));

    ui->nextButton->setDisabled(false);

    if(count == 0)
    {
        ui->previousButton->setDisabled(true);
        //ui->fixImageCheck->setVisible(false);
    }
}


void MainWindow::on_combineSharesCheck_clicked()
{
    if (ui->combineSharesCheck->isChecked())
    {
        ui->Share1Check->setDisabled(true);
        ui->Share2Check->setDisabled(true);
        ui->Share3Check->setDisabled(true);
        ui->Share4Check->setDisabled(true);
    }

    else
    {
        ui->Share1Check->setDisabled(false);
        ui->Share2Check->setDisabled(false);
        ui->Share3Check->setDisabled(false);
        ui->Share4Check->setDisabled(false);
    }

}


void MainWindow::on_SaveToFile_clicked()
{
    ui->SaveToFile->setDisabled(true);
    system("mkdir Shares"); //make directory shares
    for(unsigned int i=0; i<shares.size(); i++) //save share_i.png in directory
    {
        std::ostringstream name;
        name << "share_" << i+1 << ".png";
        cv::imwrite("Shares/" + name.str(), shares[i]);
    }

    cv::cvtColor(combShare, combShare, CV_BGR2RGB);
    cv::imwrite("Shares/combined_share.png", combShare);

    QMessageBox Saved; //output successful to user
    Saved.setText("Saved in Directory 'Shares'");
    Saved.exec();
}


void MainWindow::on_fixImageCheck_clicked()
{
    combShare = combineShares(shares);
    if(ui->fixImageCheck->isChecked())
    {
        cv::Mat fixedCombShare;
        if (n == 2) //(2,n) scheme
        {
            fixedCombShare = fixCombinedShare(combShare.clone(), n, 1);
        }

        else if (n==k) //(k,k) scheme
        {
            fixedCombShare = fixCombinedShare(combShare.clone(), pow(2,k-1), pow(2,k-1)-1);
        }

        else //(3,4) scheme
        {
            fixedCombShare = fixCombinedShare(combShare.clone(), 6, 2);
        }

        QImage QcombShare = QImage((uchar*) fixedCombShare.data, fixedCombShare.cols,
                fixedCombShare.rows, fixedCombShare.step, QImage::Format_Grayscale8);
        QPixmap QShare = QPixmap::fromImage(QcombShare);
        ui->OutputImage->setVisible(true);
        int w = ui->InputImage->width();
        int h = ui->InputImage->height();
        ui->OutputImage->setPixmap(QShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    else
    {
        QImage QcombShare = QImage((uchar*) combShare.data, combShare.cols,
                combShare.rows, combShare.step, QImage::Format_Grayscale8);
        QPixmap QShare = QPixmap::fromImage(QcombShare);
        ui->OutputImage->setVisible(true);
        int w = ui->InputImage->width();
        int h = ui->InputImage->height();
        ui->OutputImage->setPixmap(QShare.scaled(w,h,Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

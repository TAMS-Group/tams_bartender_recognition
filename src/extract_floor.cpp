/*
*
* Project Name:   Visual perception for the visually impaired
* Author List:    Pankaj Baranwal, Ridhwan Luthra, Shreyas Sachan, Shashwat Yashaswi
* Filename:     remove_floor.cpp
* Functions:    callback, main 
* Global Variables: pub -> Ros publisher
*
*/
#include <ros/ros.h>
// PCL specific includes voxel
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// PCL specific includes planar segmentation
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <std_msgs/Float32MultiArray.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>



ros::Publisher voxel_pub, pub, coef_pub;
std::string surface_frame = "/surface";
bool has_surface_transform = false;
tf::Transform plane_tf;

void interpolateTransforms(const tf::Transform& t1, const tf::Transform& t2, double fraction, tf::Transform& t_out){
	t_out.setOrigin( t1.getOrigin()*(1-fraction) + t2.getOrigin()*fraction );
	t_out.setRotation( t1.getRotation().slerp(t2.getRotation(), fraction) );
}

void callback (const pcl::PCLPointCloud2ConstPtr& cloud_pcl2) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromPCLPointCloud2 (*cloud_pcl2, *cloud);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

  // filter range
  pcl::ModelCoefficients sphere_coeff;
  sphere_coeff.values.resize (4);
  sphere_coeff.values[0] = 0;
  sphere_coeff.values[1] = 0;
  sphere_coeff.values[2] = 0;
  sphere_coeff.values[3] = 0;

  pcl::ModelOutlierRemoval<pcl::PointXYZRGB> sphere_filter;
  sphere_filter.setModelCoefficients (sphere_coeff);
  sphere_filter.setThreshold (1.5);
  sphere_filter.setModelType (pcl::SACMODEL_SPHERE);
  sphere_filter.setInputCloud (cloud);
  sphere_filter.filter (*cloud);

  // starting the segmentation of planar components.
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (0.01);
  seg.setInputCloud (cloud);
    
  seg.segment(*inliers, *coefficients);

  // normalize coefficients
  coefficients->values[3] = -coefficients->values[3];
  if(coefficients->values[2] > 0 && coefficients->values[3] > 0) {
  	coefficients->values[0] = -coefficients->values[0];
  	coefficients->values[1] = -coefficients->values[1];
  	coefficients->values[2] = -coefficients->values[2];
  	coefficients->values[3] = -coefficients->values[3];
  }
  std::cerr << "Plane coefficients: " << *coefficients<< std::endl;

  // Exit if no plane found
  if (inliers->indices.size() == 0) return;

  // publish plane coefficients
  std_msgs::Float32MultiArray coef_msg;
  coef_msg.data = coefficients->values;
  coef_pub.publish(coef_msg);

  float a = coefficients->values[0];
  float b = coefficients->values[1];
  float c = coefficients->values[2];
  float d = coefficients->values[3];

  float sqrt_abc = std::sqrt(std::pow(a,2) + std::pow(b,2) + std::pow(c,2));
  float p = d / sqrt_abc;

  tf::Vector3 normal(a / sqrt_abc, b / sqrt_abc, c / sqrt_abc);

  geometry_msgs::Pose pose;
  pose.position.x = p * normal[0];
  pose.position.y = p * normal[1];
  pose.position.z = p * normal[2];

  tf::Vector3 up(0.0, 0.0, 1.0);
  tf::Vector3 norm=normal.cross(up).normalized();
  float up_angle = -1.0 * std::acos(normal.dot(up));
  tf::Quaternion q(norm, up_angle);
  q.normalize();
  norm = q.getAxis();
  up_angle = q.getAngle();
  tf::quaternionTFToMsg(q, pose.orientation);

  tf::Transform new_tf;
  tf::poseMsgToTF(pose, new_tf);
  if(has_surface_transform) {
	  interpolateTransforms(plane_tf, new_tf, 0.1, new_tf);
  }
  plane_tf = new_tf;
  static tf::TransformBroadcaster tf_broadcaster;
  tf_broadcaster.sendTransform(tf::StampedTransform(plane_tf, ros::Time::now(), cloud->header.frame_id, surface_frame));
  has_surface_transform = true;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr surfaceCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  surfaceCloud->header.frame_id = surface_frame;
  for(pcl::PointXYZRGB point : *cloud) {
	  float point_distance = (point.x * a + point.y * b + point.z * c - d / sqrt_abc);
	  if(0.03 < point_distance && point_distance < 0.25) {
		  surfaceCloud->push_back(point);
	  }
  }

  // Executing the transformation
  Eigen::Affine3d surface_affine = Eigen::Affine3d::Identity();
  tf::transformTFToEigen(plane_tf, surface_affine);
  pcl::transformPointCloud (*surfaceCloud, *surfaceCloud, surface_affine.inverse());

  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud (surfaceCloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.filter (*surfaceCloud);

  pcl::PCLPointCloud2 outcloud;
  pcl::toPCLPointCloud2 (*surfaceCloud, outcloud);
  pub.publish (outcloud);

  /*
  // filter range
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr surfaceCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ModelOutlierRemoval<pcl::PointXYZRGB> plane_filter;
  coefficients->values[3] = -coefficients->values[3];
  plane_filter.setModelCoefficients (*coefficients);
  plane_filter.setThreshold (0.01);
  plane_filter.setModelType (pcl::SACMODEL_PLANE);
  plane_filter.setInputCloud (cloud);
  plane_filter.filter (*surfaceCloud);

  pcl::ConvexHull<pcl::PointXYZRGB> hull;
  // hull.setDimension (2); // not necessarily needed, but we need to check the dimensionality of the output
  hull.setInputCloud (cloud);
  hull.reconstruct (*surfaceCloud);
  if (hull.getDimension () == 2)
  { 
	  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
	  pcl::ExtractPolygonalPrismData<pcl::PointXYZRGB> prism;
	  prism.setInputCloud(cloud);
	  prism.setInputPlanarHull(surfaceCloud);
	  prism.setHeightLimits(0.0, 0.5);
	  prism.segment(*inliers);

	  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	  extract.setInputCloud(cloud);
	  extract.setIndices(inliers);
	  extract.setNegative(false);
	  extract.filter(*cloud); 


  }
  */



  /*
  pcl::CropBox<pcl::PointXYZRGB> boxFilter;
  boxFilter.setMin(Eigen::Vector4f(-10, -10, 0.0, 1.0));
  boxFilter.setMax(Eigen::Vector4f(10, 10, 10, 1.0));
  Eigen::Affine3f boxTransform(Eigen::Affine3f::Identity());
  boxTransform.translate(Eigen::Vector3f(p * normal[0], p * normal[1], p * normal[2]));
  boxTransform.rotate(Eigen::AngleAxis<float>(up_angle, Eigen::Vector3f(norm[0], norm[1], norm[2])));

  boxFilter.setTransform(boxTransform);
  boxFilter.setInputCloud(cloud);
  boxFilter.filter(*cloud);
  */


  // pcl::PointCloud<pcl::PointXYZRGB> cloud_xyzrgb = *cloud_voxeled;

  // // colors the point cloud red
  // for (size_t i = 0; i < cloud_xyzrgb.points.size(); i++) {
  //   cloud_xyzrgb.points[i].r = 255;
  // }

  // Publish the plane removed cloud to a new topic.

}



/*
*
* Function Name: callback
* Input: input -> A ros message service that provides point cloud data from kinect
* Output:  Publishes the point cloud after removing floor.
* Logic:   first a voxelgrid filter is applied to make the cloud less dense.
*          then Sac segmentation is done using ransac model to extract the planes
*          Then using extractIndices the point cloud without the floor plane is extracted.
* Example Call: Callback function. Manual calling not required. 
*
*/
void callback_voxeled (const pcl::PCLPointCloud2ConstPtr& cloud_pcl2) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_voxeled (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromPCLPointCloud2 (*cloud_pcl2, *cloud);

  // this is the voxel grid filtering
  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel;
  voxel.setInputCloud (cloud_pcl2);
  voxel.setLeafSize (0.01f, 0.01f, 0.01f);
  pcl::PCLPointCloud2 cloud_voxeled_pcl2;
  voxel.filter (cloud_voxeled_pcl2);

  voxel_pub.publish(cloud_voxeled_pcl2);

  // Convert to the templated PointCloud
  pcl::fromPCLPointCloud2 (cloud_voxeled_pcl2, *cloud_voxeled);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

  // starting the segmentation of planar components.
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (0.3);
  seg.setInputCloud (cloud_voxeled);
    
  seg.segment(*inliers, *coefficients);
  std::cerr << "Plane coefficients: " << *coefficients<< std::endl;

  // normalize coefficients
  coefficients->values[3] = -coefficients->values[3];
  if(coefficients->values[2] > 0 && coefficients->values[3] > 0) {
  	coefficients->values[0] = -coefficients->values[0];
  	coefficients->values[1] = -coefficients->values[1];
  	coefficients->values[2] = -coefficients->values[2];
  	coefficients->values[3] = -coefficients->values[3];
  }
  std::cerr << "Plane coefficients: " << *coefficients<< std::endl;

  // Exit if no plane found
  if (inliers->indices.size() == 0) return;

  // publish plane coefficients
  std_msgs::Float32MultiArray coef_msg;
  coef_msg.data = coefficients->values;
  coef_pub.publish(coef_msg);

  float a = coefficients->values[0];
  float b = coefficients->values[1];
  float c = coefficients->values[2];
  float d = coefficients->values[3];

  float sqrt_abc = std::sqrt(std::pow(a,2) + std::pow(b,2) + std::pow(c,2));
  float p = d / sqrt_abc;

  tf::Vector3 normal(a / sqrt_abc, b / sqrt_abc, c / sqrt_abc);

  geometry_msgs::Pose pose;
  pose.position.x = p * normal[0];
  pose.position.y = p * normal[1];
  pose.position.z = p * normal[2];

  tf::Vector3 up(0.0, 0.0, 1.0);
  tf::Vector3 norm=normal.cross(up).normalized();
  float up_angle = -1.0 * std::acos(normal.dot(up));
  tf::Quaternion q(norm, up_angle);
  q.normalize();
  norm = q.getAxis();
  up_angle = q.getAngle();
  tf::quaternionTFToMsg(q, pose.orientation);

  /*
  pcl::CropBox<pcl::PointXYZRGB> boxFilter;
  boxFilter.setMin(Eigen::Vector4f(-10, -10, 0.0, 1.0));
  boxFilter.setMax(Eigen::Vector4f(10, 10, 10, 1.0));
  Eigen::Affine3f boxTransform(Eigen::Affine3f::Identity());
  boxTransform.translate(Eigen::Vector3f(p * normal[0], p * normal[1], p * normal[2]));
  boxTransform.rotate(Eigen::AngleAxis<float>(up_angle, Eigen::Vector3f(norm[0], norm[1], norm[2])));

  boxFilter.setTransform(boxTransform);
  boxFilter.setInputCloud(cloud);
  boxFilter.filter(*cloud);
  */


  // pcl::PointCloud<pcl::PointXYZRGB> cloud_xyzrgb = *cloud_voxeled;

  // // colors the point cloud red
  // for (size_t i = 0; i < cloud_xyzrgb.points.size(); i++) {
  //   cloud_xyzrgb.points[i].r = 255;
  // }

  // Publish the plane removed cloud to a new topic.
  pcl::PCLPointCloud2 outcloud;
  pcl::toPCLPointCloud2 (*cloud, outcloud);
  pub.publish (outcloud);
}

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "remove_floor");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points", 1, callback);
  //ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points", 1, callback_voxeled);


  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_surface", 1);
  voxel_pub = nh.advertise<sensor_msgs::PointCloud2> ("voxeled", 1);
  coef_pub = nh.advertise<std_msgs::Float32MultiArray> ("plane_coefficients", 1);

  // Spin
  ros::spin();
}

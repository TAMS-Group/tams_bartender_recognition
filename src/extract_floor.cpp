#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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



ros::Publisher pub, coef_pub;
std::string surface_frame = "/surface";
bool has_surface_transform = false;
tf::Transform plane_tf;

void interpolateTransforms(const tf::Transform& t1, const tf::Transform& t2, double fraction, tf::Transform& t_out){
	t_out.setOrigin( t1.getOrigin()*(1-fraction) + t2.getOrigin()*fraction );
	t_out.setRotation( t1.getRotation().slerp(t2.getRotation(), fraction) );
}

void filterRange(double range, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud, pcl::PointCloud<pcl::PointXYZRGB>& outcloud) {
	// filter range
	pcl::ModelCoefficients sphere_coeff;
	sphere_coeff.values.resize (4);

	pcl::ModelOutlierRemoval<pcl::PointXYZRGB> sphere_filter;
	sphere_filter.setModelCoefficients (sphere_coeff);
	sphere_filter.setThreshold (1.5);
	sphere_filter.setModelType (pcl::SACMODEL_SPHERE);
	sphere_filter.setInputCloud (incloud);
	sphere_filter.filter (outcloud);
}

bool segmentSurface(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr plane_coefs) {
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (1000);
	seg.setDistanceThreshold (0.01);
	seg.setInputCloud (cloud);

	seg.segment(*inliers, *plane_coefs);

	// success if there are any inliers
	return inliers->indices.size() > 0;
}


void callback (const pcl::PCLPointCloud2ConstPtr& cloud_pcl2) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromPCLPointCloud2 (*cloud_pcl2, *cloud);

  // filter range of view
  filterRange(1.5, cloud, *cloud);

  // segment the surface and get coefficients
  pcl::ModelCoefficients::Ptr plane_coefs (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  // segment the surface
  if (!segmentSurface(cloud, inliers, plane_coefs)) return;

  // normalize coefficients and flip orientation if normal points away from camera
  plane_coefs->values[3] = -plane_coefs->values[3];
  if(plane_coefs->values[2] > 0 && plane_coefs->values[3] > 0) {
    plane_coefs->values[0] = -plane_coefs->values[0];
    plane_coefs->values[1] = -plane_coefs->values[1];
    plane_coefs->values[2] = -plane_coefs->values[2];
    plane_coefs->values[3] = -plane_coefs->values[3];
  }

  std::cerr << "Plane coefficients: " << *plane_coefs<< std::endl;

  // Exit if no plane found


  // publish plane coefficients
  std_msgs::Float32MultiArray coef_msg;
  coef_msg.data = plane_coefs->values;
  coef_pub.publish(coef_msg);

  float a = plane_coefs->values[0];
  float b = plane_coefs->values[1];
  float c = plane_coefs->values[2];
  float d = plane_coefs->values[3];

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
}



int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "remove_floor");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points", 1, callback);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_surface", 1);
  coef_pub = nh.advertise<std_msgs::Float32MultiArray> ("plane_coefficients", 1);

  // Spin
  ros::spin();
}

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>



// #include <pcl/ModelCoefficients.h>
// #include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
// #include <pcl/sample_consensus/method_types.h>
// #include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <std_msgs/Float32MultiArray.h>

#include <visualization_msgs/Marker.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>



ros::Publisher pub;
ros::Publisher coef_pub;

/*
//http://pointclouds.org/documentation/tutorials/correspondence_grouping.php#correspondence-grouping
bool use_hough_ = true;
void cluster_cb(const sensor_msgs::PointCloud2ConstPtr& input)
{

	//  Compute Normals
	//
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	norm_est.setKSearch (10);
	norm_est.setInputCloud (model);
	norm_est.compute (*model_normals);

	norm_est.setInputCloud (scene);
	norm_est.compute (*scene_normals);

	//
	//  Downsample Clouds to Extract keypoints
	//

	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud (model);
	uniform_sampling.setRadiusSearch (model_ss_);
	uniform_sampling.filter (*model_keypoints);
	std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

	uniform_sampling.setInputCloud (scene);
	uniform_sampling.setRadiusSearch (scene_ss_);
	uniform_sampling.filter (*scene_keypoints);
	std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;


	//
	//  Compute Descriptor for keypoints
	//
	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	descr_est.setRadiusSearch (descr_rad_);

	descr_est.setInputCloud (model_keypoints);
	descr_est.setInputNormals (model_normals);
	descr_est.setSearchSurface (model);
	descr_est.compute (*model_descriptors);

	descr_est.setInputCloud (scene_keypoints);
	descr_est.setInputNormals (scene_normals);
	descr_est.setSearchSurface (scene);
	descr_est.compute (*scene_descriptors);

	//
	//  Find Model-Scene Correspondences with KdTree
	//
	pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

	pcl::KdTreeFLANN<DescriptorType> match_search;
	match_search.setInputCloud (model_descriptors);

	//  For each scene keypoint descriptor, find nearest neighbor into the model 
	//  keypoints descriptor cloud and add it to the correspondences vector.
	for (size_t i = 0; i < scene_descriptors->size (); ++i)
	{
		std::vector<int> neigh_indices (1);
		std::vector<float> neigh_sqr_dists (1);
		if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
		{
			continue;
		}
		int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);

		//  add match only if the squared descriptor distance is less than 0.25 
		//  (SHOT descriptor distances are between 0 and 1 by design)
		if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) 		{
			pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back (corr);
		}
	}
	std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

	//
	//  Actual Clustering
	//
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
	std::vector<pcl::Correspondences> clustered_corrs;

	//  Using Hough3D
	if (use_hough_)
	{
		//
		//  Compute (Keypoints) Reference Frames only for Hough
		//
		pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
		pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

		pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
		rf_est.setFindHoles (true);
		rf_est.setRadiusSearch (rf_rad_);

		rf_est.setInputCloud (model_keypoints);
		rf_est.setInputNormals (model_normals);
		rf_est.setSearchSurface (model);
		rf_est.compute (*model_rf);

		rf_est.setInputCloud (scene_keypoints);
		rf_est.setInputNormals (scene_normals);
		rf_est.setSearchSurface (scene);
		rf_est.compute (*scene_rf);

		//  Clustering
		pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
		clusterer.setHoughBinSize (cg_size_);
		clusterer.setHoughThreshold (cg_thresh_);
		clusterer.setUseInterpolation (true);
		clusterer.setUseDistanceWeight (false);

		clusterer.setInputCloud (model_keypoints);
		clusterer.setInputRf (model_rf);
		clusterer.setSceneCloud (scene_keypoints);
		clusterer.setSceneRf (scene_rf);
		clusterer.setModelSceneCorrespondences (model_scene_corrs);

		//clusterer.cluster (clustered_corrs);
		clusterer.recognize (rototranslations, clustered_corrs);
	}
	else // Using GeometricConsistency
	{
		pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
		gc_clusterer.setGCSize (cg_size_);
		gc_clusterer.setGCThreshold (cg_thresh_);

		gc_clusterer.setInputCloud (model_keypoints);
		gc_clusterer.setSceneCloud (scene_keypoints);
		gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

		//gc_clusterer.cluster (clustered_corrs);
		gc_clusterer.recognize (rototranslations, clustered_corrs);
	}
}
*/


void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg (*input, *cloud);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Build a passthrough filter to remove spurious NaNs
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 0.5);
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

  // pcl::PCLPointCloud2 outcloud;
  // pcl::toPCLPointCloud2 (*cloud_filtered, outcloud);
  // pub.publish (outcloud);

  // Estimate point normals
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud_filtered);
  ne.setKSearch (50);
  ne.compute (*cloud_normals);
  /*

  pcl::SACSegmentation<pcl::PointXYZRGB> sega;
  sega.setOptimizeCoefficients (true);
  sega.setModelType (pcl::SACMODEL_PLANE);
  sega.setMethodType (pcl::SAC_RANSAC);
  sega.setMaxIterations (1000);
  sega.setDistanceThreshold (0.01);
  sega.setInputCloud (cloud_filtered);

  // Create the segmentation object for the planar model and set all the parameters
  // pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> seg;
  pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);
  // seg.setOptimizeCoefficients (true);
  // seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
  // seg.setNormalDistanceWeight (0.1);
  // seg.setMethodType (pcl::SAC_RANSAC);
  // seg.setMaxIterations (100);
  // seg.setDistanceThreshold (0.03);
  // seg.setInputCloud (cloud_filtered);
  // seg.setInputNormals (cloud_normals);
  // Obtain the plane inliers and coefficients
  sega.segment (*inliers_plane, *coefficients_plane);
  std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers_plane);
  extract.setNegative (false);

  // Write the planar inliers to disk
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB> ());
  extract.filter (*cloud_plane);
  std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  extract.filter (*cloud_filtered);

  // pcl::PCLPointCloud2 outcloud;
  // pcl::toPCLPointCloud2 (*cloud_filtered, outcloud);
  // pub.publish (outcloud);

  pcl::ExtractIndices<pcl::Normal> extract_normals;
  extract_normals.setNegative (true);
  extract_normals.setInputCloud (cloud_normals);
  extract_normals.setIndices (inliers_plane);
  extract_normals.filter (*cloud_normals);

  */

  // Create the segmentation object for cylinder segmentation and set all the parameters
  pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> seg;
  pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations (10000);
  seg.setDistanceThreshold (0.05);
  seg.setRadiusLimits (0.035, 0.045);
  seg.setInputCloud (cloud_filtered);
  seg.setInputNormals (cloud_normals);
  /*

  pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);
  pcl::SampleConsensusModelCylinder<pcl::PointXYZRGB, pcl::Normal>::Ptr
	 cylinder_model(new pcl::SampleConsensusModelCylinder<pcl::PointXYZRGB, pcl::Normal> (cloud_filtered));
  cylinder_model->setAxis(Eigen::Vector3f(0, 0, 1));
  cylinder_model->setRadiusLimits (0.035, 0.045);
  cylinder_model->setInputCloud(cloud_filtered);
  cylinder_model->setInputNormals(cloud_normals);
  pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(cylinder_model);
  ransac.setDistanceThreshold(0.0);
  ransac.computeModel();
  ransac.getInliers(inliers_cylinder->indices);
  Eigen::VectorXf coefs;
  ransac.getModelCoefficients(coefs);
  if(coefs.size() == 4) {
	  coefficients_cylinder->values[0] = coefs[0];
	  coefficients_cylinder->values[1] = coefs[1];
	  coefficients_cylinder->values[2] = coefs[2];
	  coefficients_cylinder->values[3] = coefs[3];
  }
  */


  // Obtain the cylinder inliers and coefficients
  seg.segment (*inliers_cylinder, *coefficients_cylinder);
  std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

  // Write the cylinder inliers to disk
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers_cylinder);
  extract.setNegative (false);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cylinder (new pcl::PointCloud<pcl::PointXYZRGB> ());
  extract.filter (*cloud_cylinder);
  if (cloud_cylinder->points.empty ()) 
    std::cerr << "Can't find the cylindrical component." << std::endl;
  else
  {
    std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;

    pcl::PCLPointCloud2 outcloud;
    pcl::PCLPointCloud2 temp;
    //pcl::toPCLPointCloud2 (*cloud_plane, temp);
    pcl::toPCLPointCloud2 (*cloud_cylinder, outcloud);
    //pcl::concatenatePointCloud(outcloud, temp, outcloud);
    pub.publish (outcloud);
    std_msgs::Float32MultiArray coef_msg;
    coef_msg.data = coefficients_cylinder->values;
    coef_pub.publish(coef_msg);


   // static tf::TransformBroadcaster tf_broadcaster;

   // geometry_msgs::Pose pose;
   // pose.position.x = coefficients_cylinder->values[0];
   // pose.position.y = coefficients_cylinder->values[1];
   // pose.position.z = coefficients_cylinder->values[2];
   // pose.orientation.w = 1.0;

   // tf::Transform cyl_tf;

   // double x = coefficients_cylinder->values[3];
   // double y = coefficients_cylinder->values[4];
   // double z = coefficients_cylinder->values[5];

   // tf::Vector3 axis(x, y, z);
   // tf::Vector3 up(0.0, 0.0, 1.0);
   // tf::Vector3 norm=axis.cross(up).normalized();
   // tf::Quaternion q(norm, -1.0*std::acos(axis.dot(up)));
   // q.normalize();
   // tf::quaternionTFToMsg(q, pose.orientation);

   // tf::poseMsgToTF(pose, cyl_tf);
   // tf_broadcaster.sendTransform(tf::StampedTransform(cyl_tf, ros::Time::now(), "/camera_rgb_optical_frame", "/cylinder"));

   // visualization_msgs::Marker cyl;
   // cyl.header.frame_id = "/cylinder";
   // cyl.header.stamp = ros::Time();
   // cyl.ns = "";
   // cyl.id = 0;
   // cyl.type = visualization_msgs::Marker::CYLINDER;
   // cyl.action = visualization_msgs::Marker::ADD;
   // cyl.scale.x = 2 * coefficients_cylinder->values[6];
   // cyl.scale.y = 2 * coefficients_cylinder->values[6];
   // cyl.scale.z = 0.3;
   // cyl.color.a = 0.2;
   // cyl.color.g = 1.0;
   // cyl.pose.orientation.w = 1.0;
   // cyl_pub.publish (cyl);
  }
}

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "cluster_extraction");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  //std::string input_topic = "/camera/depth_registered/points";
  std::string input_topic = "/segmented_surface";
  ros::Subscriber sub = nh.subscribe (input_topic, 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("primitive_clusters", 1);
  coef_pub = nh.advertise<std_msgs::Float32MultiArray> ("/cylinder_coefficients", 1);

  //cyl_pub = nh.advertise<visualization_msgs::Marker> ("cylinders", 1);

  // Spin
  ros::spin ();
}

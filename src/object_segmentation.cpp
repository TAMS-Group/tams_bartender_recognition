#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d_omp.h>


#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <std_msgs/Float32MultiArray.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <visualization_msgs/Marker.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <tams_bartender_recognition/SegmentedObject.h>
#include <tams_bartender_recognition/SegmentedObjectArray.h>
#include <tams_bartender_recognition/SegmentationSwitch.h>


struct BoundingBox {
  float x;
  float y;
  float z;
  float width;
  float height;
  float depth;
};


ros::Subscriber pcl_sub_;
ros::Publisher surface_pub, cyl_marker_pub, objects_pub, clusters_pub, object_image_pub;
ros::ServiceServer switch_service;
std::string surface_frame_, point_cloud_topic_;
bool publish_surface_transform_ = false;
bool has_surface_transform = false;
bool has_cylinder_transform = false;
bool enabled = false;
ros::Time start_time_;
tf::Transform surface_tf;
tf::Transform cyl_tf;

std::map<int,tf::Transform> object_transforms;
std::map<int, ros::Publisher> image_pubs;


void interpolateTransforms(const tf::Transform& t1, const tf::Transform& t2, double fraction, tf::Transform& t_out){
  t_out.setOrigin( t1.getOrigin()*(1-fraction) + t2.getOrigin()*fraction );
  t_out.setRotation( t1.getRotation().slerp(t2.getRotation(), fraction) );
}

void estimateNormals(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::Normal>& normals) {
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud);
  ne.setKSearch (30);
  ne.compute (normals);
}

void filterRange(double range, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud, pcl::PointCloud<pcl::PointXYZRGB>& outcloud) {
  // filter range
  pcl::ModelCoefficients sphere_coeff;
  sphere_coeff.values.resize (4);

  pcl::ModelOutlierRemoval<pcl::PointXYZRGB> sphere_filter;
  sphere_filter.setModelCoefficients (sphere_coeff);
  sphere_filter.setThreshold (range);
  sphere_filter.setModelType (pcl::SACMODEL_SPHERE);
  sphere_filter.setInputCloud (incloud);
  sphere_filter.filter (outcloud);
}

void filterAboveSurface(const pcl::ModelCoefficients::Ptr plane_coefs, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud, pcl::PointCloud<pcl::PointXYZRGB>& outcloud, double min=0.015, double max=0.2) {
  float a = plane_coefs->values[0];
  float b = plane_coefs->values[1];
  float c = plane_coefs->values[2];
  float d = plane_coefs->values[3];
  float sqrt_abc = std::sqrt(std::pow(a,2) + std::pow(b,2) + std::pow(c,2));
  float p = d / sqrt_abc;

  for(pcl::PointXYZRGB point : *incloud) {
    float point_distance = (point.x * a + point.y * b + point.z * c - d / sqrt_abc);
    if(min < point_distance && point_distance < max) {
      outcloud.push_back(point);
    }
  }
}

void voxelFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud, pcl::PointCloud<pcl::PointXYZRGB>& outcloud)
{
  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud (incloud);
  sor.setLeafSize (0.005f, 0.005f, 0.005f);
  sor.filter (outcloud);
}

void removeStatisticalOutliers(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud, pcl::PointCloud<pcl::PointXYZRGB>& outcloud) {
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud (incloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.setKeepOrganized(true);
  sor.filter (outcloud);
}

bool segmentSurface(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr plane_coefs) {
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.03);
  seg.setInputCloud (cloud);

  seg.segment(*inliers, *plane_coefs);

  // success if there are any inliers
  return inliers->indices.size() > 0;
}

bool segmentCylinder(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients) {


  pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> seg;

  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.05);
  seg.setMaxIterations (5000);
  seg.setDistanceThreshold (0.15);
  seg.setRadiusLimits (0.028, 0.045);
  seg.setInputCloud (cloud);
  seg.setInputNormals (normals);

  // Obtain the cylinder inliers and coefficients
  seg.segment (*inliers, *coefficients);
  return inliers->indices.size() > 0;

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
}

void transformPointCloud(const tf::Transform transform, const std::string& frame_id, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud, pcl::PointCloud<pcl::PointXYZRGB>& outcloud) {
  Eigen::Affine3d surface_affine = Eigen::Affine3d::Identity();
  tf::transformTFToEigen(transform, surface_affine);
  pcl::transformPointCloud (*incloud, outcloud, surface_affine.inverse());
  outcloud.header.frame_id = frame_id;
}

void normalizeSurfaceCoefficients(pcl::ModelCoefficients::Ptr plane_coefs) {
  plane_coefs->values[3] = -plane_coefs->values[3];
  if(plane_coefs->values[2] > 0 && plane_coefs->values[3] > 0) {
    plane_coefs->values[0] = -plane_coefs->values[0];
    plane_coefs->values[1] = -plane_coefs->values[1];
    plane_coefs->values[2] = -plane_coefs->values[2];
    plane_coefs->values[3] = -plane_coefs->values[3];
  }
}

geometry_msgs::Pose getSurfacePoseFromCoefficients(pcl::ModelCoefficients::Ptr plane_coefs) {
  geometry_msgs::Pose pose;
  float a = plane_coefs->values[0];
  float b = plane_coefs->values[1];
  float c = plane_coefs->values[2];
  float d = plane_coefs->values[3];

  float sqrt_abc = std::sqrt(std::pow(a,2) + std::pow(b,2) + std::pow(c,2));
  float p = d / sqrt_abc;

  tf::Vector3 normal(a / sqrt_abc, b / sqrt_abc, c / sqrt_abc);

  pose.position.x = p * normal[0];
  pose.position.y = p * normal[1];
  pose.position.z = p * normal[2];

  tf::Vector3 up(0.0, 0.0, 1.0);
  tf::Vector3 norm=normal.cross(up).normalized();
  float up_angle = -1.0 * std::acos(normal.dot(up));
  tf::Quaternion q(norm, up_angle);
  q.normalize();
  tf::quaternionTFToMsg(q, pose.orientation);
  return pose;
}

void updateSurfaceTransform(const geometry_msgs::Pose& pose, const std::string& cloud_frame) {

  tf::Transform new_tf;
  tf::poseMsgToTF(pose, new_tf);
  if(has_surface_transform) {
    interpolateTransforms(surface_tf, new_tf, 0.1, new_tf);
  }
  surface_tf = new_tf;
  static tf::TransformBroadcaster tf_broadcaster;
  if(publish_surface_transform_)
    tf_broadcaster.sendTransform(tf::StampedTransform(surface_tf, ros::Time::now(), cloud_frame, surface_frame_));
  has_surface_transform = true;
}

void index_to_xy(int index, int width, int &x, int &y) {
  x = index % width;
  y = index / width;
}

sensor_msgs::Image cutoutImage(const sensor_msgs::Image *image,
    BoundingBox bb, const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud) {

  // Bounding Box
  Eigen::Vector4f min(bb.x - 0.5*bb.width, bb.y - 0.5*bb.height, bb.z - 0.5*bb.depth, 1.0);
  Eigen::Vector4f max(bb.x + 0.5*bb.width, bb.y + 0.5*bb.height, bb.z +0.5*bb.depth, 1.0);
  pcl::IndicesPtr indices(new std::vector<int>);
  pcl::CropBox<pcl::PointXYZRGB> box_cropper;
  box_cropper.setInputCloud(cloud);
  box_cropper.setMin(min);
  box_cropper.setMax(max);
  box_cropper.setKeepOrganized(true);
  box_cropper.filter(*indices);

  // Image
  double cloud_image_i_ratio = (double) (image->width * image->height) / (double) (cloud->width * cloud->height);
  int image_min_x = std::numeric_limits<int>::max();
  int image_min_y = std::numeric_limits<int>::max();
  int image_max_x = std::numeric_limits<int>::min();
  int image_max_y = std::numeric_limits<int>::min();
  for (auto cloud_index : *indices) {
    auto image_index = (int) ((double) cloud_index * cloud_image_i_ratio);
    int image_x, image_y;
    index_to_xy(image_index, image->width, image_x, image_y);
    if (image_x < image_min_x) {
      image_min_x = image_x;
    }
    if (image_y < image_min_y) {
      image_min_y = image_y;
    }
    if (image_x > image_max_x) {
      image_max_x = image_x;
    }
    if (image_y > image_max_y) {
      image_max_y = image_y;
    }
  }

  auto width = (unsigned) (abs(image_max_x - image_min_x));
  auto height = (unsigned) (abs(image_max_y - image_min_y));
  auto step = 3;

  sensor_msgs::Image output_image;
  output_image.header.seq = image->header.seq;
  output_image.header.stamp = image->header.stamp;
  output_image.header.frame_id = image->header.frame_id;
  output_image.width = width;
  output_image.height = height;
  output_image.encoding = image->encoding;
  output_image.is_bigendian = image->is_bigendian;
  output_image.step = step * width;

  for (int i = 0; i < (image->width * image->height); i++) {
    int x, y;
    index_to_xy(i, image->width, x, y);

    if (x >= image_min_x && x < image_max_x && y >= image_min_y && y < image_max_y) {
      output_image.data.push_back(image->data[i * step]);
      output_image.data.push_back(image->data[i * step + 1]);
      output_image.data.push_back(image->data[i * step + 2]);
    }
  }

  return output_image;
}

void setColor(pcl::PointXYZRGB &point, int color) {
  //std::map<std::string, std::vector<int>> const colors {
  //	{ "red", {255, 0, 0} },
  //	{ "green", {0, 255, 0} },
  //	{ "blue", {0, 0, 255} },
  //	{ "yellow", {255, 255, 0} },
  //	{ "magenta", {255, 0, 255} },
  //	{ "cyan", {0, 255, 255} },
  //	{ "black", {0, 0, 0} },
  //	{ "orange", {255, 91, 0} },
  //	{ "purple", {111, 49, 152}  }
  //};

  const std::vector<std::vector<int>> colors  = {
    { {255, 0, 0} },
    { {0, 255, 0} },
    { {0, 0, 255} },
    { {255, 255, 0} },
    { {255, 0, 255} },
    { {0, 255, 255} },
    { {0, 0, 0} },
    { {255, 91, 0} },
    { {111, 49, 152}  }
  };
  std::vector<int> c = colors[color % colors.size()];
  point.r = c[0];
  point.g = c[1];
  point.b = c[2];
}

void extractClusters(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered, std::vector<pcl::PointIndices> &cluster_indices) {

  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  //std::vector<int> mapping;
  //pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, mapping);

  //cloud_filtered->is_dense = false;
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (cloud_filtered);

  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (0.06);
  ec.setMinClusterSize (80);
  ec.setMaxClusterSize (5000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  //std::cerr << "Found " << cluster_indices.size() << " clusters" << std::endl;
  //std::cerr << "Cluster sizes: ";

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*cloud_filtered));

  std::vector<int> sizes;
  for(int i = 0; i < cluster_indices.size(); i++) {
    const pcl::PointIndices indices = cluster_indices[i];
    //std::cerr << indices.indices.size() << ' ';
    for (int j : indices.indices){
      setColor((*color_filtered_cloud)[j], i);
    }
  }
  //std::cerr << std::endl;

  pcl::PCLPointCloud2 outcloud;
  pcl::toPCLPointCloud2 (*color_filtered_cloud, outcloud);
  clusters_pub.publish (outcloud);
}



void callback (const pcl::PCLPointCloud2ConstPtr& cloud_pcl2) {
  if(!enabled)
    return;

  ros::Time cloud_time;
  pcl_conversions::fromPCL(cloud_pcl2->header.stamp, cloud_time);

  if(cloud_time < start_time_){
    double delay = ros::Duration(start_time_ - cloud_time).toSec();
    if(delay > 5.0)
      ROS_ERROR_THROTTLE(3,"Object segmentation failed - Received point cloud is from %f seconds ago!", delay);
    return;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromPCLPointCloud2 (*cloud_pcl2, *cloud);

  // leave if cloud is empty
  if(cloud->size() == 0)
    return;

  //
  //         Extract surface transform and filter points in the region above it
  //

  // filter range of view
  filterRange(0.95, cloud, *cloud_filtered);
  if(cloud_filtered->size() == 0)
    return;

  // remove NaNs
  std::vector<int> mapping;
  pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_filtered, mapping);
  if(cloud_filtered->size() == 0)
    return;

  // downsample cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxels (new pcl::PointCloud<pcl::PointXYZRGB>);
  voxelFilter(cloud_filtered, *voxels);
  if(voxels->size() == 0)
    return;

  // segment the surface and get coefficients
  pcl::ModelCoefficients::Ptr surface_coefs (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  if (!segmentSurface(voxels, inliers, surface_coefs)) return;

  // normalize coefficients and flip orientation if surface normal points away from camera
  normalizeSurfaceCoefficients(surface_coefs);
  //std::cerr << "Plane coefficients: " << *surface_coefs<< std::endl;

  // retrieve pose of surface
  geometry_msgs::Pose surface_pose = getSurfacePoseFromCoefficients(surface_coefs);

  // publish surface pose as surface_frame_ to /tf
  updateSurfaceTransform(surface_pose, cloud->header.frame_id);

  // filter point cloud to region above surface
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr surfaceCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr surfaceVoxels (new pcl::PointCloud<pcl::PointXYZRGB>);
  filterAboveSurface(surface_coefs, cloud_filtered, *surfaceCloud);
  filterAboveSurface(surface_coefs, voxels, *surfaceVoxels);
  surfaceVoxels->header.frame_id = cloud->header.frame_id;
  if(surfaceVoxels->size() == 0 || surfaceCloud->size() == 0)
    return;

  // publish segmented surface cloud
  pcl::PCLPointCloud2 outcloud;
  surfaceCloud->header.frame_id = cloud->header.frame_id;
  pcl::toPCLPointCloud2 (*surfaceCloud, outcloud);
  surface_pub.publish (outcloud);

  // remove statistical outliers and NaNs
  removeStatisticalOutliers(surfaceVoxels, *surfaceVoxels);
  if(surfaceVoxels->size() == 0)
    return;
  pcl::removeNaNFromPointCloud(*surfaceVoxels, *surfaceVoxels, mapping);
  if(surfaceVoxels->size() == 0)
    return;


  // extract clusters from voxels
  std::vector<pcl::PointIndices> cluster_indices;
  extractClusters(surfaceVoxels, cluster_indices);

  tams_bartender_recognition::SegmentedObjectArray objects;
  objects.header.frame_id = surface_frame_;
  objects.header.stamp = ros::Time::now();

  for(const pcl::PointIndices cluster : cluster_indices) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr cyl_coefs (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr cyl_inliers (new pcl::PointIndices);

    for (int i : cluster.indices)
      cluster_cloud->points.push_back (surfaceVoxels->points[i]);

    if(cluster_cloud->points.size() == 0)
      continue;

    cluster_cloud->width = cluster_cloud->points.size();
    cluster_cloud->height = 1;
    cluster_cloud->is_dense = true;

    // segment cluster as cylinder
    estimateNormals(cluster_cloud, *cloud_normals);
    if(!segmentCylinder(cluster_cloud, cloud_normals, cyl_inliers, cyl_coefs))
      continue;

    tams_bartender_recognition::SegmentedObject object_msg;

    // basic object parameters
    double x = cyl_coefs->values[0];
    double y = cyl_coefs->values[1];
    double z = cyl_coefs->values[2];
    double object_radius = cyl_coefs->values[6];

    // retrieve object pose in surface frame
    tf::Transform cam_to_object(tf::Quaternion::getIdentity(), tf::Vector3(x, y, z));
    tf::Transform surface_to_object = surface_tf.inverse() * cam_to_object;

    // fix object to upright rotation and align z with surface
    surface_to_object.setRotation(tf::Quaternion::getIdentity());
    tf::Vector3 objectXYZ = surface_to_object.getOrigin();
    objectXYZ.setZ(0.0);
    surface_to_object.setOrigin(objectXYZ);

    // create pose stamped in surface frame
    object_msg.pose.header.frame_id = surface_frame_;
    tf::poseTFToMsg(surface_to_object, object_msg.pose.pose);

    // Try to extract a 2d image of the object
    try{
      // get full 2d image of cloud
      sensor_msgs::Image object_image;
      object_image.width=100;
      object_image.height=200;
      pcl::toROSMsg(*cloud, object_image);

      objectXYZ.setZ(0.15);
      surface_to_object.setOrigin(objectXYZ);
      cam_to_object = surface_tf * surface_to_object;

      // define bounding box size and position
      BoundingBox bb;
      bb.x = cam_to_object.getOrigin().getX();
      bb.y = cam_to_object.getOrigin().getY();
      bb.z = cam_to_object.getOrigin().getZ();
      bb.width = 0.15;
      bb.height = 0.3;
      bb.depth = 0.3;

      // extract object image from full image
      object_msg.image = cutoutImage(&object_image, bb, cloud);
      tf::transformTFToMsg(surface_tf, object_msg.surface_transform);
      // add SegmentedObject message to SegmentedObjectArray
      objects.objects.push_back(object_msg);
      objects.count++;
    }
    catch (std::runtime_error)
    {
      ROS_ERROR("Unable to extract object image from cloud!");
    }
  }

  if(objects.objects.size() > 0)
  {
    objects_pub.publish(objects);
    object_image_pub.publish(objects.objects[0].image);
  }
  else
  {
    ROS_WARN_STREAM_THROTTLE(3, "Did not find any objects. Found " << cluster_indices.size() << " clusters (object candidates).");
  }
}

bool switch_cb(tams_bartender_recognition::SegmentationSwitch::Request  &req,
    tams_bartender_recognition::SegmentationSwitch::Response &res)
{
  // ignore if switch is already set to requested value
  if (enabled == req.enabled) {
    ROS_WARN("Service call ignored - Object Segmentation is already switched to requested state!");
    return false;
  }

  start_time_ = req.header.stamp;
  enabled = req.enabled;
  if (enabled){
    ros::NodeHandle nh;
    pcl_sub_ = nh.subscribe (point_cloud_topic_, 1, callback);
  }
  else
    pcl_sub_.shutdown();
  res.success = true;
  return true;
}



int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "tams_bartender_recognition");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  point_cloud_topic_ = pnh.param<std::string>("point_cloud_topic", "/camera/depth_registered/points");
  surface_frame_ = pnh.param<std::string>("surface_frame", "/surface");

  switch_service = nh.advertiseService("object_segmentation_switch", switch_cb);
  enabled = pnh.param("enabled", false);
  publish_surface_transform_ = pnh.param("publish_surface_transform", false);


  // Create a ROS publisher for the output point cloud
  surface_pub = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_surface", 1);
  clusters_pub = nh.advertise<sensor_msgs::PointCloud2> ("/extracted_clusters", 1);
  cyl_marker_pub = nh.advertise<visualization_msgs::Marker> ("cylinders", 1);
  objects_pub = nh.advertise<tams_bartender_recognition::SegmentedObjectArray>("/segmented_objects", 1);
  object_image_pub = nh.advertise<sensor_msgs::Image>("/object_image", 1);

  // Create a ROS subscriber for the input point cloud
  if(enabled) {
    start_time_ = ros::Time::now();
    pcl_sub_ = nh.subscribe (point_cloud_topic_, 1, callback);
  }

  // Spin
  ros::spin();
}

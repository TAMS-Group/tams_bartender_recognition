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
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>


#include <iostream>
#include <pcl/io/pcd_io.h>
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

#include <orbbec_astra_ip/SegmentedBottle.h>
#include <orbbec_astra_ip/SegmentedBottleArray.h>

struct BoundingBox {
    float x;
    float y;
    float z;
    float width;
    float height;
    float depth;
};


ros::Publisher surface_pub, cyl_marker_pub, bottles_pub;
std::string surface_frame = "/surface";
std::string bottle_frame = "/bottle";
bool has_surface_transform = false;
bool has_cylinder_transform = false;
tf::Transform surface_tf;
tf::Transform cyl_tf;

std::map<int,tf::Transform> bottle_transforms;
std::map<int, ros::Publisher> image_pubs;


void interpolateTransforms(const tf::Transform& t1, const tf::Transform& t2, double fraction, tf::Transform& t_out){
    t_out.setOrigin( t1.getOrigin()*(1-fraction) + t2.getOrigin()*fraction );
    t_out.setRotation( t1.getRotation().slerp(t2.getRotation(), fraction) );
}

void estimateNormals(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::Normal>& normals) {
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
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

void filterAboveSurface(const pcl::ModelCoefficients::Ptr plane_coefs, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr incloud, pcl::PointCloud<pcl::PointXYZRGB>& outcloud) {
    float a = plane_coefs->values[0];
    float b = plane_coefs->values[1];
    float c = plane_coefs->values[2];
    float d = plane_coefs->values[3];
    float sqrt_abc = std::sqrt(std::pow(a,2) + std::pow(b,2) + std::pow(c,2));
    float p = d / sqrt_abc;

    for(pcl::PointXYZRGB point : *incloud) {
        float point_distance = (point.x * a + point.y * b + point.z * c - d / sqrt_abc);
        if(0.015 < point_distance && point_distance < 0.3) {
            outcloud.push_back(point);
        }
    }
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
    seg.setMaxIterations (1000);
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
    seg.setNormalDistanceWeight (0.25);
    seg.setMaxIterations (2000);
    seg.setDistanceThreshold (0.15);
    seg.setRadiusLimits (0.035, 0.045);
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

void publishSurfaceTransform(const geometry_msgs::Pose& pose, const std::string& cloud_frame, const std::string& surface_frame) {

    tf::Transform new_tf;
    tf::poseMsgToTF(pose, new_tf);
    if(has_surface_transform) {
        interpolateTransforms(surface_tf, new_tf, 0.1, new_tf);
    }
    surface_tf = new_tf;
    static tf::TransformBroadcaster tf_broadcaster;
    tf_broadcaster.sendTransform(tf::StampedTransform(surface_tf, ros::Time::now(), cloud_frame, surface_frame));
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



void callback (const pcl::PCLPointCloud2ConstPtr& cloud_pcl2) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2 (*cloud_pcl2, *cloud);


    //
    //         Extract surface transform and filter points in the region above it
    //

    // filter range of view
    filterRange(1.3, cloud, *cloud_filtered);

    // segment the surface and get coefficients
    pcl::ModelCoefficients::Ptr surface_coefs (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    if (!segmentSurface(cloud_filtered, inliers, surface_coefs)) return;

    // normalize coefficients and flip orientation if surface normal points away from camera
    normalizeSurfaceCoefficients(surface_coefs);
    //std::cerr << "Plane coefficients: " << *surface_coefs<< std::endl;

    // retrieve pose of surface
    geometry_msgs::Pose surface_pose = getSurfacePoseFromCoefficients(surface_coefs);

    // publish surface pose as surface_frame to /tf
    publishSurfaceTransform(surface_pose, cloud->header.frame_id, surface_frame);

    // filter point cloud to region above surface
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr surfaceCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    filterAboveSurface(surface_coefs, cloud_filtered, *surfaceCloud);

    // remove statistical outliers
    removeStatisticalOutliers(surfaceCloud, *surfaceCloud);

    // publish segmented surface cloud
    pcl::PCLPointCloud2 outcloud;
    surfaceCloud->header.frame_id = cloud->header.frame_id;
    pcl::toPCLPointCloud2 (*surfaceCloud, outcloud);
    surface_pub.publish (outcloud);

    //
    // Segment Cylinders and extract bottle candidates
    //

    // Estimate point normals
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    estimateNormals(surfaceCloud, *cloud_normals);

    // Create the segmentation object for cylinder segmentation and set all the parameters
    pcl::ModelCoefficients::Ptr cyl_coefs (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr cyl_inliers (new pcl::PointIndices);
    std::vector<geometry_msgs::Pose> cylinder_poses;

    // initialize SegmentedBottleArray message
    orbbec_astra_ip::SegmentedBottleArray bottles;
    bottles.header.frame_id = surface_frame;
    bottles.header.stamp = ros::Time::now();

    int max_count = 5;
    int bottle_count = 0;

    // iteratively extract bottles from pointcloud
    while(bottle_count < max_count && segmentCylinder(surfaceCloud, cloud_normals, cyl_inliers, cyl_coefs)) {
        //std::cerr << "Cylinder coefficients: " << *cyl_coefs<< std::endl;
        orbbec_astra_ip::SegmentedBottle bottle_msg;

        // basic bottle parameters
        double bottle_radius = cyl_coefs->values[6];
        double bottle_height = 0.3;
        std::string bottle_frame_id = bottle_frame + std::to_string(bottle_count);

        // compute cylinder orientation (might be tested for correct upright rotation)
        tf::Vector3 cyl_axis(cyl_coefs->values[3], cyl_coefs->values[4], cyl_coefs->values[5]);
        const tf::Vector3 z_axis(0.0, 0.0, 1.0);
        tf::Vector3 norm=cyl_axis.cross(z_axis).normalized();
        float up_angle = -1.0 * std::acos(cyl_axis.dot(z_axis));
        tf::Quaternion q(norm, up_angle);
        q.normalize();

        // retrieve bottle pose in surface frame
        tf::Transform bottle_tf(tf::Quaternion::getIdentity(), tf::Vector3(cyl_coefs->values[0], cyl_coefs->values[1], cyl_coefs->values[2]));
        tf::Transform new_tf = surface_tf.inverse() * bottle_tf;

        // fix bottle to upright rotation
        new_tf.setRotation(tf::Quaternion::getIdentity());

        // create pose stamped in surface frame
        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = surface_frame;

        // fix bottle z position and add to SegmentedBottle Message
        tf::poseTFToMsg(new_tf, pose.pose);
        pose.pose.position.z = 0.5*bottle_height;  // center bounding box pose
        tf::poseMsgToTF(pose.pose, new_tf);

        pose.pose.position.z = 0.0;  // fix object position z to 0, we fix the height later
        bottle_msg.pose = pose;

        //  interpolate new pose with previous one - this now happens after classification
        //if(bottle_transforms.find(bottle_count) != bottle_transforms.end()) {
        //    interpolateTransforms(bottle_transforms[bottle_count], new_tf, 0.1, new_tf);
        //}
        //bottle_transforms[bottle_count] = new_tf;

        //static tf::TransformBroadcaster tf_broadcaster;
        //tf_broadcaster.sendTransform(tf::StampedTransform(new_tf, ros::Time::now(), surface_frame, bottle_frame_id));


        // Try to extract a 2d image of the bottle
        try{
            geometry_msgs::Pose cam_to_bottle;
            tf::poseTFToMsg(surface_tf * new_tf, cam_to_bottle);

            // get full 2d image of cloud
            sensor_msgs::Image bottle_image;
            bottle_image.width=100;
            bottle_image.height=200;
            pcl::toROSMsg(*cloud, bottle_image);

            // define bounding box size and position
            BoundingBox bb;
            bb.x = cam_to_bottle.position.x;
            bb.y = cam_to_bottle.position.y;
            bb.z = cam_to_bottle.position.z;
            bb.width = 0.15;
            bb.height = 0.3;
            bb.depth = 0.3;

            // extract bottle image from full image
            bottle_msg.image = cutoutImage(&bottle_image, bb, cloud);
        }
        catch (std::runtime_error)
        {
            ROS_ERROR("Unable to extract bottle image from cloud!");
            return;
        }

        // Create and publish bottle marker
        //visualization_msgs::Marker cyl;
        //cyl.header.frame_id = surface_frame;
        //cyl.header.stamp = ros::Time();
        //cyl.ns = "/bottles";
        //cyl.id = bottle_count;
        //cyl.type = visualization_msgs::Marker::CYLINDER;
        //cyl.action = visualization_msgs::Marker::ADD;
        //cyl.scale.x = 2 * bottle_radius;
        //cyl.scale.y = 2 * bottle_radius;
        //cyl.scale.z = 0.3;
        //cyl.color.a = 0.2;
        //cyl.color.g = 1.0;
        //cyl.pose = pose.pose;
        //cyl_marker_pub.publish (cyl);


        // add SegmentedBottle message to SegmentedBottleArray
        bottles.bottles.push_back(bottle_msg);

        // Remove bottle from pointcloud
        pcl::ModelOutlierRemoval<pcl::PointXYZRGB> cyl_filter;
        cyl_filter.setModelCoefficients (*cyl_coefs);
        cyl_filter.setModelType (pcl::SACMODEL_CYLINDER);
        cyl_filter.setThreshold(0.05);
        cyl_filter.setNegative(true);
        cyl_filter.setInputNormals(cloud_normals);
        cyl_filter.setInputCloud (surfaceCloud);
        cyl_filter.filter (*surfaceCloud);


        // increase bottle count
        bottle_count += 1;

        // leave if cloud is empty
        if(surfaceCloud->size()== 0) {
            break;
        }

        // else we compute normals again and look for other bottles
        estimateNormals(surfaceCloud, *cloud_normals);
    }


    // publish SegmentedBottlesArray
    bottles.count = bottle_count;
    bottles_pub.publish(bottles);
}



int main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "pcl_bottle_recognition");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points", 1, callback);

    // Create a ROS publisher for the output point cloud
    surface_pub = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_surface", 1);
    cyl_marker_pub = nh.advertise<visualization_msgs::Marker> ("cylinders", 1);
    bottles_pub = nh.advertise<orbbec_astra_ip::SegmentedBottleArray>("/segmented_bottles", 1);


    // Spin
    ros::spin();
}

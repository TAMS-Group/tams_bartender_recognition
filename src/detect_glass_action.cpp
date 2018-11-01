#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <tams_bartender_msgs/DetectGlassAction.h>
#include <tams_bartender_recognition/SegmentationSwitch.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>


#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shape_extents.h>

#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <apriltags2_ros/AprilTagDetectionArray.h>

std::string GLASS_MESH= "package://tams_bartender_recognition/meshes/glass-binary.stl";

class GlassDetectionServer
{

  private:

    // ros
    ros::NodeHandle nh_;
    ros::Subscriber tag_detections_sub_, image_sub_, camera_info_sub_;
    ros::Publisher image_pub_, camera_info_pub_;
    ros::ServiceClient segmentation_client_;
    actionlib::SimpleActionServer<tams_bartender_msgs::DetectGlassAction> as_;


    // surface and camera frames
    std::string upright_frame_;
    std::string camera_frame_; // TODO: check for correct frame


    // image forward parameters for april tag detection
    std::string image_topic_;
    std::string camera_info_topic_;

    // glass and tag ids
    std::string glass_id_;
    int glass_tag_id_;
    double offset_x_;
    double offset_y_;

    // tf and tag detection
    tf::Transform tag_transform;
    tf::TransformListener tf_listener;
    tf::StampedTransform upright_camera_transform_;
    bool detection_running_ = false;
    bool tag_found_= false;
    float filter_weight_;

    void interpolateTransforms(const tf::Transform& t1, const tf::Transform& t2, double fraction, tf::Transform& t_out){
      t_out.setOrigin( t1.getOrigin()*(1-fraction) + t2.getOrigin()*fraction );
      t_out.setRotation( t1.getRotation().slerp(t2.getRotation(), fraction) );
    }

    bool setPose(moveit_msgs::CollisionObject& object) {
      geometry_msgs::Pose glass_pose;

      // check if tag was found
      if (!tag_found_) {
        ROS_ERROR("Could not find any valid glass tag!");
        return false;
      }

      // compute transform from surface to tag
      tf::Transform upright_obj_transform = upright_camera_transform_ * tag_transform;

      // create glass pose with fixed orientation and defined offset
      tf::poseTFToMsg(upright_obj_transform, glass_pose);
      // offset between tag and glass
      glass_pose.position.x += offset_x_;
      glass_pose.position.x += offset_y_;
      double mesh_height = computeMeshHeight(object.meshes[0]);
      glass_pose.position.z = 0.5 * mesh_height + 0.002;
      //// upright orientation
      glass_pose.orientation.x = 0.0;
      glass_pose.orientation.y = 0.0;
      glass_pose.orientation.z = 0.0;
      glass_pose.orientation.w = 1.0;
      // create glass pose with fixed orientation and defined offset
      tf::poseMsgToTF(glass_pose, upright_obj_transform);
      tf::poseTFToMsg(upright_camera_transform_.inverse() * upright_obj_transform, glass_pose);
      object.mesh_poses[0] = glass_pose;
      object.header.frame_id = camera_frame_;
      return true;
    }

    void tagDetectionCallback(const apriltags2_ros::AprilTagDetectionArray& msg){

      // run detection only when requested
      if(detection_running_) {

        // iterate detections
        for (int i=0; i< msg.detections.size(); i++) {

          //check if detection is not empty
          if(msg.detections[i].id.size() > 0) {

            // check if detection belongs to the glass tag
            if(msg.detections[i].id[0] == glass_tag_id_) {
              tf::Transform new_tag_transform;
              tf::poseMsgToTF(msg.detections[i].pose.pose.pose, new_tag_transform);

              //// interpolate new tag with previous
              if(tag_found_)
                interpolateTransforms(tag_transform, new_tag_transform, filter_weight_, new_tag_transform);

              // save detection and mark tag as found
              tag_transform = new_tag_transform;
              tag_found_ = true;
            }
          }
        }
      }
    }

    bool createCollisionObject(std::string id, geometry_msgs::PoseStamped pose, moveit_msgs::CollisionObject& object) {
      collisionObjectFromResource(object, id, GLASS_MESH);
      object.header.frame_id = pose.header.frame_id;
      double mesh_height = computeMeshHeight(object.meshes[0]);
      object.mesh_poses.resize(1);

      // glass center
      object.mesh_poses[0] = pose.pose;
      object.mesh_poses[0].position.z = 0.5 * mesh_height + 0.002;

      return true;
    }

    void collisionObjectFromResource(moveit_msgs::CollisionObject& msg, const std::string& id, const std::string& resource) {
      msg.meshes.resize(1);

      // load mesh
      const Eigen::Vector3d scaling(1, 1, 1);
      shapes::Shape* shape = shapes::createMeshFromResource(resource, scaling);
      shapes::ShapeMsg shape_msg;
      shapes::constructMsgFromShape(shape, shape_msg);
      msg.meshes[0] = boost::get<shape_msgs::Mesh>(shape_msg);

      // set pose
      msg.mesh_poses.resize(1);
      msg.mesh_poses[0].orientation.w = 1.0;

      // fill in details for MoveIt
      msg.id = id;
      msg.operation = moveit_msgs::CollisionObject::ADD;
    }

    double computeMeshHeight(const shape_msgs::Mesh& mesh) {
      double x,y,z;
      geometric_shapes::getShapeExtents(mesh, x, y, z);
      return z;
    }

    bool setSegmentationEnabled(bool enabled) {

      tams_bartender_recognition::SegmentationSwitch srv;
      srv.request.enabled = enabled;
      srv.request.header.stamp = ros::Time::now();
      if (!segmentation_client_.call(srv))
      {
        ROS_ERROR_STREAM("Calling object_segmentation_switch service failed." << std::endl
            << "Aborting detect_glass_action.");
        return false;
      }
      return true;
    }

    bool findStableSurfaceFrame(double duration=3.0) {
      bool success = false;

      // switch segmentation on
      if(!setSegmentationEnabled(true))
        return false;

      // wait until transform is stable
      ros::Time start_time = ros::Time::now();
      ros::Duration timeout(duration);
      while(ros::Time::now() - start_time < timeout) {
        try {
          tf::StampedTransform new_transform;
          tf_listener.waitForTransform(upright_frame_, camera_frame_, ros::Time(0), ros::Duration(1.0));
          tf_listener.lookupTransform(upright_frame_, camera_frame_, ros::Time(0), new_transform);
          Eigen::Affine3d prev_mat;
          Eigen::Affine3d next_mat;
          tf::transformTFToEigen(new_transform, next_mat);
          tf::transformTFToEigen(upright_camera_transform_, prev_mat);
          upright_camera_transform_ = new_transform;

          // check if consecutive transforms are approx equal
          if(prev_mat.isApprox(next_mat)) {
            success = true;
            break;
          }
        }
        catch(...){}
        ROS_WARN_THROTTLE(10, "Waiting for surface->camera transform");
      }

      // switch segmentation off
      if(!setSegmentationEnabled(false))
        return false;

      return success;
    }

  protected:

    void forwardCameraImage(bool enabled)
    {
      if (enabled) {
        image_sub_ = nh_.subscribe(image_topic_, 1, &GlassDetectionServer::imageCallback, this);
        camera_info_sub_ = nh_.subscribe(camera_info_topic_, 1, &GlassDetectionServer::camerainfoCallback, this);
      } else {
        image_sub_.shutdown();
        camera_info_sub_.shutdown();
      }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& img)
    {
      if(detection_running_)
        image_pub_.publish(img);
    }

    void camerainfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info)
    {
      if(detection_running_)
        camera_info_pub_.publish(camera_info);
    }


    void execute_cb(const tams_bartender_msgs::DetectGlassGoalConstPtr &goal)
    {
      // retrieve stable surface frame
      if(!findStableSurfaceFrame()) {
        ROS_ERROR("Unable to find a stable surface frame!");
        as_.setAborted();
        return;
      }

      // start detection
      tag_found_ = false;
      detection_running_ = true;
      forwardCameraImage(true);

      // wait for timeout
      ros::Duration(goal->timeout).sleep();

      // stop detection
      detection_running_ = false;
      forwardCameraImage(false);

      // create collision object with mesh
      moveit_msgs::CollisionObject glass;
      collisionObjectFromResource(glass, glass_id_, GLASS_MESH);

      // retrieve glass pose
      geometry_msgs::PoseStamped glass_pose;
      if(!setPose(glass)) {
        as_.setAborted();
        return;
      }

      // add object to planning scene
      moveit::planning_interface::PlanningSceneInterface psi;
      psi.applyCollisionObject(glass);

      // return result
      tams_bartender_msgs::DetectGlassResult result;
      result.detected_glass = glass_id_;
      as_.setSucceeded(result);
    }

  public:
    GlassDetectionServer() : as_(nh_, "detect_glass_action", boost::bind(&GlassDetectionServer::execute_cb, this, _1), false)
  {
    ros::NodeHandle pnh("~");

    //  glass pose parameters
    glass_tag_id_ = pnh.param("glass_tag", 435);
    glass_id_ = pnh.param<std::string>("glass_id", "glass");
    offset_x_ = pnh.param("offset_x", 0.1);
    offset_y_ = pnh.param("offset_y", 0.0);
    upright_frame_ = pnh.param<std::string>("upright_frame", "/surface");
    camera_frame_ = pnh.param<std::string>("camera_frame", "/camera_rgb_optical_frame");
    filter_weight_ = pnh.param("glass_pose_filter_weight", 0.25);

    // image subscriber topics for apriltag detection
    image_topic_ = pnh.param<std::string>("image_topic", "/camera/rgb/image_rect_color");
    camera_info_topic_ = pnh.param<std::string>("camera_info_topic", "/camera/rgb/camera_info");

    // camera forward publishers
    std::string image_forward_topic = pnh.param<std::string>("image_forward_topic", "/apriltags2_image_forward");
    std::string camera_info_forward_topic = pnh.param<std::string>("camera_info_forward_topic", "/apriltags2_camera_info_forward");
    image_pub_ = nh_.advertise<sensor_msgs::Image>(image_forward_topic, 1);
    camera_info_pub_ = nh_.advertise<sensor_msgs::CameraInfo>(camera_info_forward_topic, 1);

    // setup detection pipeline
    segmentation_client_ = nh_.serviceClient<tams_bartender_recognition::SegmentationSwitch>("object_segmentation_switch");
    tag_detections_sub_ = nh_.subscribe("tag_detections", 1, &GlassDetectionServer::tagDetectionCallback, this);

    // start action server
    as_.start();
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "detect_glass_action");
  ros::AsyncSpinner spinner(2);
  spinner.start();
  GlassDetectionServer gds;
  ros::waitForShutdown();
}

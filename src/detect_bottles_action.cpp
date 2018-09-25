#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <tiago_bartender_msgs/DetectBottlesAction.h>
#include <tams_bartender_recognition/RecognizedObject.h>
#include <std_srvs/SetBool.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shape_extents.h>


//std::string BOTTLE_MESH = "package://tams_bartender_recognition/meshes/bottle-binary.stl";
std::string BOTTLE_MESH = "package://tams_bartender_recognition/meshes/bottle_small.stl";

std::string GLASS_MESH= "package://tams_bartender_recognition/meshes/glass-binary.stl";


class BottleActionServer
{

  private:
    ros::NodeHandle nh_;
    actionlib::SimpleActionServer<tiago_bartender_msgs::DetectBottlesAction> as_;
    ros::Subscriber object_pose_sub;
    tf::TransformListener tf_listener;

    std::string surface_frame_;
    std::string camera_frame_;
    tf::StampedTransform surface_camera_transform_;

    std::map<std::string, tams_bartender_recognition::RecognizedObject> objects_;
    std::map<std::string, int> object_count_;
    ros::ServiceClient segmentation_client_;


    bool recognize_objects_ = false;


    void object_pose_cb(const tams_bartender_recognition::RecognizedObject::ConstPtr& msg)
    {
      if(recognize_objects_) {
        objects_[msg->id] = *msg;
        object_count_[msg->id]++;
      }
    }


    bool createCollisionObject(std::string id, const tams_bartender_recognition::RecognizedObject& obj, moveit_msgs::CollisionObject& object) {

      // add mesh to object
      collisionObjectFromResource(object, id, BOTTLE_MESH);
      double mesh_height = computeMeshHeight(object.meshes[0]);

      // move object pose to half of mesh height
      geometry_msgs::Pose pose = obj.pose.pose;
      pose.position.z = 0.5 * mesh_height + 0.002;

      // transform object pose to camera frame
      tf::Transform camera_surface_transform;
      tf::Transform surface_obj_transform;
      tf::poseMsgToTF(pose, surface_obj_transform);
      tf::transformMsgToTF(obj.surface_transform, camera_surface_transform);
      object.mesh_poses.resize(1);
      tf::poseTFToMsg(camera_surface_transform * surface_obj_transform, object.mesh_poses[0]);
      object.header.frame_id = camera_frame_;

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

  protected:

    void execute_cb(const tiago_bartender_msgs::DetectBottlesGoalConstPtr &goal)
    {
      std_srvs::SetBool srv;
      srv.request.data = true;
      if (!segmentation_client_.call(srv))
      {
        ROS_ERROR_STREAM("Calling object_segmentation_switch service failed." << std::endl
            << "Aborting bottle_recogntion_action.");
        as_.setAborted();
        return;
      }

      // clear previous objects
      object_count_.clear();
      objects_.clear();

      recognize_objects_ = true;

      // wait for specified timeout for recognition
      ros::Duration(goal->timeout).sleep();

      recognize_objects_ = false;

      // switch segmentation off
      srv.request.data = false;
      if (!segmentation_client_.call(srv))
      {
        ROS_ERROR_STREAM("Calling object_segmentation_switch service failed." << std::endl
            << "Aborting bottle_recogntion_action.");
        as_.setAborted();
        return;
      }

      //try {
      //  tf_listener.waitForTransform(surface_frame_, camera_frame_, ros::Time(0), ros::Duration(1.0));
      //  tf_listener.lookupTransform(surface_frame_, camera_frame_, ros::Time(0), surface_camera_transform_);
      //}
      //catch (...) {
      //  ROS_ERROR_STREAM("Failed to detect bottles - No surface frame was found!");
      //  as_.setAborted();
      //  return;
      //}

      tiago_bartender_msgs::DetectBottlesResult result;
      std::vector<moveit_msgs::CollisionObject> objs;
      for (std::map<std::string,int>::iterator it=object_count_.begin(); it!=object_count_.end(); ++it) {
        std::string id = it->first;
        if(object_count_[id] >= goal->stability_threshold) {
          moveit_msgs::CollisionObject object;
          if(createCollisionObject(id, objects_[id], object)) {
            objs.push_back(object);
          }
          result.detected_bottles.push_back(id);
        }
      }

      moveit::planning_interface::PlanningSceneInterface psi;
      psi.applyCollisionObjects(objs);

      // return result message
      as_.setSucceeded(result);
    }

  public:
    BottleActionServer() : as_(nh_, "detect_bottles_action", boost::bind(&BottleActionServer::execute_cb, this, _1), false)
  {
    ros::NodeHandle pnh("~");
    surface_frame_ = pnh.param<std::string>("surface_frame", "/surface");
    camera_frame_ = pnh.param<std::string>("camera_frame", "xtion_rgb_optical_frame");

    segmentation_client_ = nh_.serviceClient<std_srvs::SetBool>("object_segmentation_switch");
    object_pose_sub = nh_.subscribe("object_poses", 1, &BottleActionServer::object_pose_cb, this);
    as_.start();
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "detect_bottles_action");
  ros::AsyncSpinner spinner(2);
  spinner.start();
  BottleActionServer bas;
  ros::waitForShutdown();
}

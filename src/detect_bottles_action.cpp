#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <tiago_bartender_msgs/DetectBottlesAction.h>
#include <tams_bartender_recognition/RecognizedObject.h>
#include <tams_bartender_recognition/SegmentationSwitch.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include <visualization_msgs/Marker.h>

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

    std::string camera_frame_;
    tf::StampedTransform surface_camera_transform_;

    std::map<std::string, tams_bartender_recognition::RecognizedObject> objects_;
    std::map<std::string, int> object_count_;
    ros::ServiceClient segmentation_client_;

    ros::Publisher marker_pub_;

    bool recognize_objects_ = false;


    void object_pose_cb(const tams_bartender_recognition::RecognizedObject::ConstPtr& msg)
    {
      if(recognize_objects_) {
        objects_[msg->id] = *msg;
        object_count_[msg->id]++;
      }
    }


    bool createCollisionObject(int i, std::string id, const tams_bartender_recognition::RecognizedObject& obj, moveit_msgs::CollisionObject& object) {

      // add mesh to object
      collisionObjectFromResource(object, id, BOTTLE_MESH);
      object.header.frame_id = camera_frame_;

      // move object pose to half of mesh height
      geometry_msgs::Pose pose = obj.pose.pose;
      double mesh_height = computeMeshHeight(object.meshes[0]);
      pose.position.z = 0.5 * mesh_height + 0.002;
      pose.orientation.w = 1.0;

      // transform object pose to camera frame
      tf::Transform camera_surface_transform;
      tf::Transform surface_obj_transform;
      tf::poseMsgToTF(pose, surface_obj_transform);
      tf::transformMsgToTF(obj.surface_transform, camera_surface_transform);
      tf::poseTFToMsg(camera_surface_transform * surface_obj_transform, object.mesh_poses[0]);

      // publish bottle marker
      publishBottleMarker(i, id, camera_frame_, object.mesh_poses[0]);

      // publish label above bottle
      pose.position.z += mesh_height;
      tf::Transform surface_label_transform;
      tf::poseMsgToTF(pose, surface_label_transform);
      tf::poseTFToMsg(camera_surface_transform * surface_label_transform, pose);
      publishTextMarker(i, id, camera_frame_, pose);

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

    void publishBottleMarker(int id, const std::string& label, const std::string& frame_id, const geometry_msgs::Pose& pose)
    {
      visualization_msgs::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp = ros::Time::now();
      m.type = visualization_msgs::Marker::CYLINDER;
      m.pose = pose;
      m.action = visualization_msgs::Marker::ADD;
      m.id = id;
      m.ns = "cylinders";
      m.scale.x = 0.075;
      m.scale.y = 0.075;
      m.scale.z = 0.25;
      m.color.g = 1.0;
      m.color.a = 0.5;
      m.lifetime = ros::Duration(10.0);
      marker_pub_.publish(m);
    }

    void publishTextMarker(int id, const std::string& label, const std::string& frame_id, const geometry_msgs::Pose& pose)
    {
      visualization_msgs::Marker t;
      t.header.frame_id = frame_id;
      t.header.stamp = ros::Time::now();
      t.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      t.action = visualization_msgs::Marker::ADD;
      t.text = label;
      t.pose = pose;
      t.id = id;
      t.ns = "labels";
      t.scale.z = 0.03;
      t.color.r = 1.0;
      t.color.g = 1.0;
      t.color.b = 1.0;
      t.color.a = 1.0;
      t.lifetime = ros::Duration(30.0);
      marker_pub_.publish(t);
    }


  protected:

    void execute_cb(const tiago_bartender_msgs::DetectBottlesGoalConstPtr &goal)
    {
      tams_bartender_recognition::SegmentationSwitch srv;
      srv.request.enabled = true;
      srv.request.header.stamp = ros::Time::now();
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
      srv.request.enabled = false;
      srv.request.header.stamp = ros::Time::now();
      if (!segmentation_client_.call(srv))
      {
        ROS_ERROR_STREAM("Calling object_segmentation_switch service failed." << std::endl
            << "Aborting bottle_recogntion_action.");
        as_.setAborted();
        return;
      }

      int i = 0;
      tiago_bartender_msgs::DetectBottlesResult result;
      std::vector<moveit_msgs::CollisionObject> objs;
      for (std::map<std::string,int>::iterator it=object_count_.begin(); it!=object_count_.end(); ++it) {
        std::string id = it->first;
        if(object_count_[id] >= goal->stability_threshold) {
          moveit_msgs::CollisionObject object;
          if(createCollisionObject(i++, id, objects_[id], object)) {
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
    camera_frame_ = pnh.param<std::string>("camera_frame", "xtion_rgb_optical_frame");

    segmentation_client_ = nh_.serviceClient<tams_bartender_recognition::SegmentationSwitch>("object_segmentation_switch");
    object_pose_sub = nh_.subscribe("object_poses", 1, &BottleActionServer::object_pose_cb, this);
    marker_pub_ = nh_.advertise<visualization_msgs::Marker> ("/detected_bottles", 1);

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

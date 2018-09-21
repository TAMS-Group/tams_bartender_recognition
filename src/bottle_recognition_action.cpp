#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <tiago_bartender_msgs/UpdateBottlesAction.h>
#include <pcl_object_recognition/RecognizedObject.h>
#include <std_srvs/SetBool.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shape_extents.h>


std::string BOTTLE_MESH = "package://pcl_object_recognition/meshes/bottle-binary.stl";
std::string GLASS_MESH= "package://pcl_object_recognition/meshes/glass-binary.stl";


class BottleActionServer
{
  protected:
    ros::NodeHandle nh_;
    actionlib::SimpleActionServer<tiago_bartender_msgs::UpdateBottlesAction> as_;
    ros::Subscriber object_pose_sub;


    bool recognize_objects_ = false;

  public:
    BottleActionServer() : as_(nh_, "bottle_recognition_action", boost::bind(&BottleActionServer::execute_cb, this, _1), false)
  {
    segmentation_client_ = nh_.serviceClient<std_srvs::SetBool>("object_segmentation_switch");
    object_pose_sub = nh_.subscribe("object_poses", 1, &BottleActionServer::object_pose_cb, this);
    as_.start();
  }

    void execute_cb(const tiago_bartender_msgs::UpdateBottlesGoalConstPtr &goal)
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
      object_poses_.clear();

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


      tiago_bartender_msgs::UpdateBottlesResult result;
      std::vector<moveit_msgs::CollisionObject> objs;
      for (std::map<std::string,int>::iterator it=object_count_.begin(); it!=object_count_.end(); ++it) {
        std::string id = it->first;
        if(object_count_[id] >= goal->stability_threshold) {
          moveit_msgs::CollisionObject object;
          if(createCollisionObject(id, object_poses_[id], object)) {
            objs.push_back(object);
          }
          result.updated_bottles.push_back(id);
        }
      }

      moveit::planning_interface::PlanningSceneInterface psi;
      psi.applyCollisionObjects(objs);

      // return result message
      as_.setSucceeded(result);
    }


  private:
    void object_pose_cb(const pcl_object_recognition::RecognizedObject::ConstPtr& msg)
    {
      ROS_INFO_STREAM("Bottle: " << msg->id);

      if(recognize_objects_) {
        object_poses_[msg->id] = msg->pose;
        object_count_[msg->id]++;
      }
    }

    std::map<std::string, geometry_msgs::PoseStamped> object_poses_;
    std::map<std::string, int> object_count_;
    ros::ServiceClient segmentation_client_;

    bool createCollisionObject(std::string id, geometry_msgs::PoseStamped pose, moveit_msgs::CollisionObject& object) {
      collisionObjectFromResource(object, id, BOTTLE_MESH);
      object.header.frame_id = "surface";
      double mesh_height = computeMeshHeight(object.meshes[0]);
      object.mesh_poses.resize(1);

      // bottle center
      object.mesh_poses[0] = pose.pose;
      object.mesh_poses[0].position.z = 0.5 * mesh_height + 0.002;

      // // bottle tip
      // object.mesh_poses[1] = pose.pose;
      // object.mesh_poses[1].position.z = mesh_height + 0.002;
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
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "bottle_recognition_action");
  ros::AsyncSpinner spinner(2);
  spinner.start();
  BottleActionServer bas;
  ros::waitForShutdown();
}

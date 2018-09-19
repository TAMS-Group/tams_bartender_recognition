#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <tiago_bartender_msgs/UpdateBottlesAction.h>
#include <pcl_object_recognition/RecognizedObject.h>

class BottleActionServer
{
protected:
  ros::NodeHandle nh_;
  actionlib::SimpleActionServer<tiago_bartender_msgs::UpdateBottlesAction> as_;
public:
  BottleActionServer() : as_(nh_, "bottle_recognition_action", boost::bind(&BottleActionServer::execute_cb, this, _1), false)
  {
    segmentation_client_ = nh_.serviceClient<std_srvs::SetBool>("object_segmentation_switch");
    object_pose_sub = nh_.subscribe("object_pose", 1000, &BottleActionServer::object_pose_cb, this);
    as_.start();
  }

  void execute_cb(const tiago_bartender_msgs::UpdateBottlesGoalConstPtr &goal)
  {
    std_srvs::SetBool srv;
    srv.request.data = true;
    if (!segmentation_client.call(srv))
    {
      ROS_ERROR_STREAM("Calling object_segmentation_switch service failed." << std::endl
                       "Aborting bottle_recogntion_action.");
      as_.setAborted();
      return;
    }
    
  }

  
private:
  void object_pose_cb(const pcl_object_recognition::RecognizedObject::ConstPtr& msg)
  {
    
  }

  std::vector<std::pair<std::string, geometry_msgs::PoseStamped>> recognized_objects;
  ros::ServiceClient segmentation_client_;
  int stability_threshold_;
  ros::Duration timeout_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "bottle_recognition_action");
  ros::AsyncSpinner spinner(2);
  spinner.start();
  BottleActionServer bas;
  ros::waitForShutdown();
}

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <tiago_bartender_msgs/UpdateBottlesAction.h>
#include <pcl_object_recognition/RecognizedObject.h>
#include <std_srvs/SetBool.h>

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
      for (std::map<std::string,int>::iterator it=object_count_.begin(); it!=object_count_.end(); ++it) {
        std::string id = it->first;
        if(object_count_[id] >= goal->stability_threshold)
          result.updated_bottles.push_back(id);

        // spawn collision object
      }

      // return result message
      as_.setSucceeded(result);
    }


  private:
    void object_pose_cb(const pcl_object_recognition::RecognizedObject::ConstPtr& msg)
    {
      if(recognize_objects_) {
        object_poses_[msg->id] = msg->pose;
        object_count_[msg->id]++;
      }
    }

    std::map<std::string, geometry_msgs::PoseStamped> object_poses_;
    std::map<std::string, int> object_count_;
    ros::ServiceClient segmentation_client_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "bottle_recognition_action");
  ros::AsyncSpinner spinner(2);
  spinner.start();
  BottleActionServer bas;
  ros::waitForShutdown();
}

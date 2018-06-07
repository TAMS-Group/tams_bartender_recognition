#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <std_msgs/Float32MultiArray.h>

ros::Publisher cyl_pub;

std::string camera_frame = "/camera_rgb_optical_frame";
std::string surface_frame = "/surface";
std::string cylinder_topic = "/cylinder";
bool has_surface_transform = false;
bool has_cylinder_transform = false;
tf::Transform plane_tf;
tf::Transform cyl_tf;

void interpolateTransforms(const tf::Transform& t1, const tf::Transform& t2, double fraction, tf::Transform& t_out){
	t_out.setOrigin( t1.getOrigin()*(1-fraction) + t2.getOrigin()*fraction );
	t_out.setRotation( t1.getRotation().slerp(t2.getRotation(), fraction) );
}


void handle_plane(const std_msgs::Float32MultiArray coefficient_msg) {
	const std::vector<float> coefs = coefficient_msg.data;
	if(coefs.size()==4) {

		float a = coefs[0];
		float b = coefs[1];
		float c = coefs[2];
		float d = coefs[3];

		float sqrt_abc = std::sqrt(std::pow(a,2) + std::pow(b,2) + std::pow(c,2));
		float p = d / sqrt_abc;

		tf::Vector3 normal(a / sqrt_abc, b / sqrt_abc, c / sqrt_abc);

		geometry_msgs::Pose pose;
		pose.position.x = p * normal[0];
		pose.position.y = p * normal[1];
		pose.position.z = p * normal[2];

		tf::Vector3 up(0.0, 0.0, 1.0);
		tf::Vector3 norm=normal.cross(up).normalized();
		tf::Quaternion q(norm, -1.0*std::acos(normal.dot(up)));
		q.normalize();
		tf::quaternionTFToMsg(q, pose.orientation);

		/*
		geometry_msgs::Pose pose;
		pose.position.x = -c[0]*c[3];
		pose.position.y = -c[1]*c[3];
		pose.position.z = -c[2]*c[3];

		tf::Vector3 axis(c[0], c[1], c[2]);
		tf::Vector3 up(0.0, 0.0, 1.0);
		tf::Vector3 norm=axis.cross(up).normalized();
		tf::Quaternion q(norm, -1.0*std::acos(axis.dot(up)));
		q.normalize();
		tf::quaternionTFToMsg(q, pose.orientation);
		*/

		tf::Transform new_tf;
		tf::poseMsgToTF(pose, new_tf);
		if(has_surface_transform) {
			interpolateTransforms(plane_tf, new_tf, 0.1, new_tf);
		}
		plane_tf = new_tf;

		static tf::TransformBroadcaster tf_broadcaster;
		tf_broadcaster.sendTransform(tf::StampedTransform(plane_tf, ros::Time::now(), camera_frame, surface_frame));
		has_surface_transform = true;

	}
}

void handle_segment(const std_msgs::Float32MultiArray coefficient_msg) {
	const std::vector<float> coefficients = coefficient_msg.data;
	if(coefficients.size()==7) {
		static tf::TransformBroadcaster tf_broadcaster;
		static tf::Transformer tf_transformer;
		tf_transformer.setUsingDedicatedThread(true);

		double bottle_height = 0.3;

		geometry_msgs::PoseStamped pose;
		pose.header.frame_id = camera_frame;

		pose.pose.position.x = coefficients[0];
		pose.pose.position.y = coefficients[1];
		pose.pose.position.z = coefficients[2];
		pose.pose.orientation.w = 1.0;

		tf::Transform new_tf;

		double x = coefficients[3];
		double y = coefficients[4];
		double z = coefficients[5];

		tf::Vector3 axis(x, y, z);
		tf::Vector3 up(0.0, 0.0, 1.0);
		tf::Vector3 norm=axis.cross(up).normalized();
		tf::Quaternion q(norm, -1.0*std::acos(axis.dot(up)));
		q.normalize();
		tf::quaternionTFToMsg(q, pose.pose.orientation);

		/*
		tf::Transform tf_pose;
		tf::poseMsgToTF(pose.pose, tf_pose);

		//tf_pose.setRotation(tf::Quaternion(0, 0, 0, 1));
		tf::poseTFToMsg(tf_pose, pose.pose);
		*/


		// update bottle pose
		pose.header.frame_id = surface_frame;
		pose.pose.position.z = 0.5*bottle_height;
		pose.pose.orientation.w = 1.0;
		pose.pose.orientation.x = 0.0;
		pose.pose.orientation.y = 0.0;
		pose.pose.orientation.z = 0.0;

		tf::Transform tf_pose;
		tf::poseMsgToTF(pose.pose, new_tf);

		if(has_cylinder_transform) {
			interpolateTransforms(cyl_tf, new_tf, 0.05, new_tf);
		}
		cyl_tf = new_tf;
		tf_broadcaster.sendTransform(tf::StampedTransform(cyl_tf, ros::Time::now(), surface_frame, cylinder_topic));
		has_cylinder_transform = true;

		visualization_msgs::Marker cyl;
		cyl.header.frame_id = "/cylinder";
		cyl.header.stamp = ros::Time();
		cyl.ns = "";
		cyl.id = 0;
		cyl.type = visualization_msgs::Marker::CYLINDER;
		cyl.action = visualization_msgs::Marker::ADD;
		cyl.scale.x = 2 * coefficients[6];
		cyl.scale.y = 2 * coefficients[6];
		cyl.scale.z = 0.3;
		cyl.color.a = 0.2;
		cyl.color.g = 1.0;
		cyl.pose.orientation.w = 1.0;
		cyl_pub.publish (cyl);
		ROS_INFO_STREAM("Publishing cylinder at: " << pose);
	}
	else {
		ROS_ERROR_STREAM("Recieved cylinder coefficients with wrong size!");
	}
}

int main(int argc, char** argv) {
		// Initialize ROS
		ros::init (argc, argv, "segment_handling");
	ros::NodeHandle nh;

	// Create a ROS subscriber for the input point cloud
	ros::Subscriber cyl_sub = nh.subscribe ("/cylinder_coefficients", 1, handle_segment);
	//ros::Subscriber plane_sub = nh.subscribe ("/plane_coefficients", 1, handle_plane);

	cyl_pub = nh.advertise<visualization_msgs::Marker> ("cylinders", 1);

	// Spin
	ros::spin ();
}

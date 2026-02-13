#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <tf/transform_broadcaster.h>
#include <cmath>

class FakeSimulatorNode
{
public:
    FakeSimulatorNode()
    {
        ros::NodeHandle nh;
        ros::NodeHandle nh_private("~");

        // Parameters
        nh_private.param("init_x", init_x_, 0.0);
        nh_private.param("init_y", init_y_, 0.0);
        nh_private.param("init_z", init_z_, 0.5);
        nh_private.param("init_yaw", init_yaw_, 0.0);
        nh_private.param("publish_rate", publish_rate_, 50.0);
        nh_private.param("odom_frame", odom_frame_, std::string("odom"));
        nh_private.param("base_frame", base_frame_, std::string("base_link"));
        nh_private.param("world_frame", world_frame_, std::string("world"));
        
        // Get topic names
        std::string pos_cmd_topic, cmd_vel_topic, odom_topic;
        nh_private.param("position_cmd_topic", pos_cmd_topic, std::string("/planning/pos_cmd"));
        nh_private.param("cmd_vel_topic", cmd_vel_topic, std::string("/cmd_vel"));
        nh_private.param("odom_topic", odom_topic, std::string("/odom"));

        // Initialize state
        x_ = init_x_;
        y_ = init_y_;
        z_ = init_z_;
        yaw_ = init_yaw_;
        vx_ = 0.0;
        vy_ = 0.0;
        vz_ = 0.0;
        vyaw_ = 0.0;
        
        last_time_ = ros::Time::now();
        last_pos_cmd_time_ = ros::Time(0);  // Initialize to zero to indicate no command received yet
        use_position_cmd_ = false;
        use_cmd_vel_ = false;
        cmd_vel_only_yaw_ = false;
        cmd_vel_yaw_ = 0.0;

        // Subscribers
        pos_cmd_sub_ = nh.subscribe(pos_cmd_topic, 10, &FakeSimulatorNode::positionCmdCallback, this);
        cmd_vel_sub_ = nh.subscribe(cmd_vel_topic, 10, &FakeSimulatorNode::cmdVelCallback, this);

        // Publishers
        odom_pub_ = nh.advertise<nav_msgs::Odometry>(odom_topic, 10);
        tf_broadcaster_ = new tf::TransformBroadcaster();

        // Timer for simulation loop
        timer_ = nh.createTimer(ros::Duration(1.0 / publish_rate_), &FakeSimulatorNode::timerCallback, this);

        ROS_INFO("Fake simulator initialized at (%.2f, %.2f, %.2f) with yaw %.2f", 
                 init_x_, init_y_, init_z_, init_yaw_);
    }

    ~FakeSimulatorNode()
    {
        delete tf_broadcaster_;
    }

private:
    void positionCmdCallback(const quadrotor_msgs::PositionCommand::ConstPtr& msg)
    {
        // PositionCommand is in "world" frame, but since world and odom are fixed and aligned
        // (static transform is identity: world -> odom), we can directly use the position values
        // as base_link position in odom frame. This creates a fixed relationship: 
        // base_link position in odom = PositionCommand position in world
        target_x_ = msg->position.x;
        target_y_ = msg->position.y;
        target_z_ = msg->position.z;
        target_vx_ = msg->velocity.x;
        target_vy_ = msg->velocity.y;
        target_vz_ = msg->velocity.z;
        target_yaw_ = msg->yaw;
        target_vyaw_ = msg->yaw_dot;
        
        last_pos_cmd_time_ = ros::Time::now();
        use_position_cmd_ = true;
        // Don't clear use_cmd_vel_, we need it to check if cmd_vel has only yaw
        
        // Immediately update position if not using cmd_vel yaw only
        // This ensures camera position is updated as soon as PositionCommand arrives
        if (!(use_cmd_vel_ && cmd_vel_only_yaw_))
        {
            // Direct assignment: base_link in odom = PositionCommand in world (they are aligned)
            x_ = target_x_;
            y_ = target_y_;
            z_ = target_z_;
            yaw_ = target_yaw_;
            vx_ = target_vx_;
            vy_ = target_vy_;
            vz_ = target_vz_;
            vyaw_ = target_vyaw_;
            
            // Publish immediately to ensure camera gets correct position
            ros::Time current_time = ros::Time::now();
            publishTransformAndOdometry(current_time);
        }
        
        ROS_DEBUG_THROTTLE(1.0, "PositionCommand received: pos=(%.2f, %.2f, %.2f), yaw=%.2f", 
                          target_x_, target_y_, target_z_, target_yaw_);
    }

    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        // Use cmd_vel
        target_vx_ = msg->linear.x;
        target_vy_ = msg->linear.y;
        target_vz_ = msg->linear.z;
        cmd_vel_yaw_ = msg->angular.z;  // Store separately to avoid being overwritten
        
        // Check if only yaw command (linear velocities are zero)
        const double vel_threshold = 1e-3;
        cmd_vel_only_yaw_ = (fabs(target_vx_) < vel_threshold && 
                             fabs(target_vy_) < vel_threshold && 
                             fabs(target_vz_) < vel_threshold);
        
        use_cmd_vel_ = true;
        // Don't disable position_cmd, we might need it if cmd_vel has linear velocity
    }

    void timerCallback(const ros::TimerEvent& event)
    {
        ros::Time current_time = ros::Time::now();
        double dt = (current_time - last_time_).toSec();
        
        if (dt <= 0.0 || dt > 0.1)  // Skip invalid or too large dt
        {
            last_time_ = current_time;
            return;
        }

        // Logic:
        // 1. If PositionCommand exists:
        //    - If cmd_vel has only yaw: use PositionCommand position, integrate yaw from cmd_vel
        //    - Otherwise: use all values from PositionCommand directly
        // 2. If only cmd_vel with only yaw: integrate yaw, keep position
        // 3. If cmd_vel has linear velocity but no PositionCommand: wait (keep current state)
        
        if (use_position_cmd_)
        {
            // Use position from PositionCommand directly (these are in world frame, 
            // and since world->odom is identity transform, we can use directly)
            x_ = target_x_;
            y_ = target_y_;
            z_ = target_z_;
            
            // Update velocities for odom message
            vx_ = target_vx_;
            vy_ = target_vy_;
            vz_ = target_vz_;
            
            // For yaw: if cmd_vel has only yaw, integrate it; otherwise use PositionCommand yaw
            if (use_cmd_vel_ && cmd_vel_only_yaw_)
            {
                // cmd_vel only has yaw: integrate yaw from cmd_vel, position from PositionCommand
                vyaw_ = cmd_vel_yaw_;  // from cmd_vel
                yaw_ += vyaw_ * dt;
                
                // Normalize yaw to [-pi, pi]
                while (yaw_ > M_PI) yaw_ -= 2.0 * M_PI;
                while (yaw_ < -M_PI) yaw_ += 2.0 * M_PI;
            }
            else
            {
                // Use yaw directly from PositionCommand
                yaw_ = target_yaw_;
                vyaw_ = target_vyaw_;
            }
            
            // Debug: log position updates
            static int debug_counter = 0;
            if (debug_counter++ % 50 == 0)  // Print every 50 iterations (~1 second at 50Hz)
            {
                ROS_INFO("Simulator state: pos=(%.3f, %.3f, %.3f), yaw=%.3f, vel=(%.3f, %.3f, %.3f)", 
                        x_, y_, z_, yaw_, vx_, vy_, vz_);
            }
        }
        else if (use_cmd_vel_)
        {
            if (cmd_vel_only_yaw_)
            {
                // Only yaw command: integrate yaw, keep position unchanged
                vyaw_ = cmd_vel_yaw_;
                yaw_ += vyaw_ * dt;
                
                // Normalize yaw to [-pi, pi]
                while (yaw_ > M_PI) yaw_ -= 2.0 * M_PI;
                while (yaw_ < -M_PI) yaw_ += 2.0 * M_PI;
                
                // Linear velocities remain zero, position unchanged
                vx_ = 0.0;
                vy_ = 0.0;
                vz_ = 0.0;
            }
            else
            {
                // cmd_vel has linear velocity but no PositionCommand: wait for PositionCommand
                // Keep current position and yaw unchanged
                vx_ = 0.0;
                vy_ = 0.0;
                vz_ = 0.0;
                vyaw_ = 0.0;
            }
        }
        else
        {
            // No command received, keep current state (velocities decay to zero)
            vx_ *= 0.9;
            vy_ *= 0.9;
            vz_ *= 0.9;
            vyaw_ *= 0.9;
            
            // Still integrate with decaying velocity
            double cos_yaw = cos(yaw_);
            double sin_yaw = sin(yaw_);
            double vx_world = vx_ * cos_yaw - vy_ * sin_yaw;
            double vy_world = vx_ * sin_yaw + vy_ * cos_yaw;
            
            x_ += vx_world * dt;
            y_ += vy_world * dt;
            z_ += vz_ * dt;
            yaw_ += vyaw_ * dt;
            
            while (yaw_ > M_PI) yaw_ -= 2.0 * M_PI;
            while (yaw_ < -M_PI) yaw_ += 2.0 * M_PI;
        }

        // Publish transform and odometry together to ensure consistency
        publishTransformAndOdometry(current_time);

        last_time_ = current_time;
    }

    void publishTransformAndOdometry(const ros::Time& current_time)
    {
        // Create quaternion once to ensure consistency between TF and odom
        tf::Quaternion q = tf::createQuaternionFromYaw(yaw_);
        
        // Publish TF transform (odom -> base_link)
        // Since world and odom are fixed and aligned, base_link position in odom frame
        // directly equals PositionCommand position in world frame
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(x_, y_, z_));
        transform.setRotation(q);
        tf_broadcaster_->sendTransform(
            tf::StampedTransform(transform, current_time, odom_frame_, base_frame_));

        // Publish odometry message with the same values
        // The odom message represents base_link pose in odom frame
        // Since world and odom are aligned, this equals PositionCommand pose in world frame
        nav_msgs::Odometry odom;
        odom.header.stamp = current_time;
        odom.header.frame_id = odom_frame_;
        odom.child_frame_id = base_frame_;

        // Position: base_link in odom = PositionCommand in world (fixed relationship)
        odom.pose.pose.position.x = x_;
        odom.pose.pose.position.y = y_;
        odom.pose.pose.position.z = z_;

        // Orientation (same quaternion as TF)
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        // Velocity (in body frame)
        odom.twist.twist.linear.x = vx_;
        odom.twist.twist.linear.y = vy_;
        odom.twist.twist.linear.z = vz_;
        odom.twist.twist.angular.z = vyaw_;

        // Covariance (simple diagonal)
        for (int i = 0; i < 36; i++)
        {
            odom.pose.covariance[i] = 0.0;
            odom.twist.covariance[i] = 0.0;
        }
        odom.pose.covariance[0] = 0.01;  // x
        odom.pose.covariance[7] = 0.01;  // y
        odom.pose.covariance[14] = 0.01; // z
        odom.pose.covariance[21] = 0.01; // roll
        odom.pose.covariance[28] = 0.01; // pitch
        odom.pose.covariance[35] = 0.01; // yaw

        odom_pub_.publish(odom);
    }

    // State variables
    double x_, y_, z_, yaw_;
    double vx_, vy_, vz_, vyaw_;
    
    // Target commands
    double target_x_, target_y_, target_z_, target_yaw_;
    double target_vx_, target_vy_, target_vz_, target_vyaw_;
    double cmd_vel_yaw_;  // Store yaw from cmd_vel separately
    
    // Initial conditions
    double init_x_, init_y_, init_z_, init_yaw_;
    
    // Flags
    bool use_position_cmd_;
    bool use_cmd_vel_;
    bool cmd_vel_only_yaw_;
    
    // Timing
    ros::Time last_time_;
    ros::Time last_pos_cmd_time_;
    double publish_rate_;
    
    // Frames
    std::string odom_frame_;
    std::string base_frame_;
    std::string world_frame_;

    // ROS
    ros::Subscriber pos_cmd_sub_;
    ros::Subscriber cmd_vel_sub_;
    ros::Publisher odom_pub_;
    tf::TransformBroadcaster* tf_broadcaster_;
    ros::Timer timer_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fake_simulator_node");
    
    FakeSimulatorNode node;
    
    ros::spin();
    
    return 0;
}


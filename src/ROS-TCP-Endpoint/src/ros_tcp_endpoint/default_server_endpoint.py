#!/usr/bin/env python

import rospy

from ros_tcp_endpoint import TcpServer
from ros_tcp_endpoint.server import SysCommands


def register_configured_publishers(tcp_server):
    publishers = rospy.get_param("~publishers", [])
    syscommands = SysCommands(tcp_server)

    for publisher in publishers:
        topic = publisher.get("topic", "")
        message_name = publisher.get("message", "")
        queue_size = publisher.get("queue_size", 10)
        latch = publisher.get("latch", False)

        if not topic or not message_name:
            rospy.logwarn(
                "Skipping configured publisher with missing topic or message: {}".format(
                    publisher
                )
            )
            continue

        syscommands.publish(topic, message_name, queue_size=queue_size, latch=latch)


def main(args=None):
    # Start the Server Endpoint
    rospy.init_node("unity_endpoint", anonymous=True)
    tcp_server = TcpServer(rospy.get_name())
    register_configured_publishers(tcp_server)
    tcp_server.start()
    rospy.spin()


if __name__ == "__main__":
    main()

// Main.cpp의 SendHandTelemetry 함수를 아래 코드로 완전히 교체
#include <sstream>

void SendHandTelemetry(
        const OVRFW::ovrApplFrameIn& in,
        const XrHandTrackingAimStateFB& aimStateL,
        const XrHandTrackingAimStateFB& aimStateR) {

    if (!tcpReady_) {
        tcpReady_ = tcpSender_.Init(tcpIp_.c_str(), tcpPort_);
        if (tcpReady_) {
            ALOG("TCP sender reconnected: %s:%d", tcpIp_.c_str(), tcpPort_);
        }
        return;
    }

    // 오른손만 전송 (Tesollo Hand 제어용)
    if (handTrackedR_) {
        std::ostringstream ss;
        ss << "HAND_DATA,";

        // MediaPipe 21개 랜드마크와 1:1 매핑되는 OpenXR 관절 순서
        int mediapipe_mapping[21] = {
            XR_HAND_JOINT_WRIST_EXT,              // 0: Wrist
            XR_HAND_JOINT_THUMB_METACARPAL_EXT,   // 1: Thumb CMC
            XR_HAND_JOINT_THUMB_PROXIMAL_EXT,     // 2: Thumb MCP  
            XR_HAND_JOINT_THUMB_DISTAL_EXT,       // 3: Thumb IP
            XR_HAND_JOINT_THUMB_TIP_EXT,          // 4: Thumb Tip
            XR_HAND_JOINT_INDEX_METACARPAL_EXT,   // 5: Index MCP base
            XR_HAND_JOINT_INDEX_PROXIMAL_EXT,     // 6: Index MCP
            XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT, // 7: Index PIP
            XR_HAND_JOINT_INDEX_DISTAL_EXT,       // 8: Index DIP
            XR_HAND_JOINT_INDEX_TIP_EXT,          // 9: Index Tip
            XR_HAND_JOINT_MIDDLE_METACARPAL_EXT,  // 10: Middle MCP base
            XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT,    // 11: Middle MCP
            XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT,// 12: Middle PIP
            XR_HAND_JOINT_MIDDLE_DISTAL_EXT,      // 13: Middle DIP
            XR_HAND_JOINT_MIDDLE_TIP_EXT,         // 14: Middle Tip
            XR_HAND_JOINT_RING_METACARPAL_EXT,    // 15: Ring MCP base
            XR_HAND_JOINT_RING_PROXIMAL_EXT,      // 16: Ring MCP
            XR_HAND_JOINT_RING_INTERMEDIATE_EXT,  // 17: Ring PIP
            XR_HAND_JOINT_RING_DISTAL_EXT,        // 18: Ring DIP
            XR_HAND_JOINT_RING_TIP_EXT,           // 19: Ring Tip
            XR_HAND_JOINT_LITTLE_PROXIMAL_EXT     // 20: Pinky MCP (Little metacarpal 대신)
        };

        // 21개 관절의 x,y,z 좌표를 CSV로 직렬화
        for (int i = 0; i < 21; i++) {
            if (jointLocationsR_[mediapipe_mapping[i]].locationFlags & 
                (XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
                
                const XrPosef& pose = jointLocationsR_[mediapipe_mapping[i]].pose;
                ss << pose.position.x << "," << pose.position.y << "," << pose.position.z;
            } else {
                ss << "0,0,0"; // Invalid joint는 원점으로
            }
            
            if (i < 20) ss << ",";
        }

        // Pinch 상태 추가
        float pinch = (aimStateR.status & XR_HAND_TRACKING_AIM_INDEX_PINCHING_BIT_FB) ? 1.0f : 0.0f;
        ss << "," << pinch << "\n";

        std::string data = ss.str();
        bool ok = tcpSender_.Send(data.c_str(), data.length());
        
        if (!ok) {
            tcpReady_ = false;
            ALOG("TCP send failed, will reconnect");
        }
    }
}

# Ballet Leap Analyzer App
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ballet Leap Analyzer", layout="centered")
st.title("Ballet Leap Analyzer 🩰")

st.markdown("""
**Instructions:**
1. Upload a video (MP4 or MOV) of your ballet leap.
2. Wait for the analysis to complete.
3. View your hip trajectory, jump height, angles, and feedback below.
""")

uploaded_file = st.file_uploader(
	"Upload a ballet leap video (MP4 or MOV)",
	type=["mp4", "mov"]
)

def analyze_video(video_path):
	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
	cap = cv2.VideoCapture(video_path)
	hip_ys = []
	frames = []
	takeoff_angle = None
	landing_angle = None
	left_knee_angles = []
	right_knee_angles = []
	frame_count = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = pose.process(frame_rgb)
		if results.pose_landmarks:
			lm = results.pose_landmarks.landmark
			# Hip midpoint (average of left and right hip)
			hip_y = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
			hip_ys.append(hip_y)
			frames.append(frame_count)
			# Knee angles (for feedback)
			left_angle = get_leg_angle(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)
			right_angle = get_leg_angle(lm, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
			left_knee_angles.append(left_angle)
			right_knee_angles.append(right_angle)
		frame_count += 1
	cap.release()
	pose.close()
	hip_ys = np.array(hip_ys)
	# Calculate peak jump (lowest y, since y=0 is top)
	if len(hip_ys) == 0:
		return None, None, None, None, None, None
	peak_idx = np.argmin(hip_ys)
	peak_height = hip_ys[0] - hip_ys[peak_idx]  # relative units
	# Takeoff: first 10% of frames, Landing: last 10%
	n = len(hip_ys)
	takeoff_idx = max(0, int(n * 0.1))
	landing_idx = min(n-1, int(n * 0.9))
	takeoff_angle = (left_knee_angles[takeoff_idx] + right_knee_angles[takeoff_idx]) / 2
	landing_angle = (left_knee_angles[landing_idx] + right_knee_angles[landing_idx]) / 2
	return frames, hip_ys, peak_height, takeoff_angle, landing_angle, (left_knee_angles, right_knee_angles)

def get_leg_angle(lm, hip, knee, ankle):
	# Returns angle at the knee (in degrees)
	a = np.array([lm[hip].x, lm[hip].y])
	b = np.array([lm[knee].x, lm[knee].y])
	c = np.array([lm[ankle].x, lm[ankle].y])
	ba = a - b
	bc = c - b
	cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
	angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
	return np.degrees(angle)

def give_feedback(peak_height, takeoff_angle, landing_angle, left_knee_angles, right_knee_angles):
	feedback = []
	if peak_height < 0.05:
		feedback.append("Try to jump higher for more elevation.")
	if takeoff_angle < 160:
		feedback.append("Try extending legs earlier during takeoff.")
	if landing_angle < 160:
		feedback.append("Try to land with straighter legs to reduce landing force.")
	if np.min(left_knee_angles) < 120 or np.min(right_knee_angles) < 120:
		feedback.append("Avoid excessive knee bend during leap.")
	if not feedback:
		feedback.append("Great leap! Keep practicing for even better results.")
	return feedback

if uploaded_file is not None:
	tfile = tempfile.NamedTemporaryFile(delete=False)
	tfile.write(uploaded_file.read())
	st.video(tfile.name)
	with st.spinner("Analyzing video, please wait..."):
		result = analyze_video(tfile.name)
	if result[0] is None:
		st.error("Could not detect a leap in the video. Please try another video.")
	else:
		frames, hip_ys, peak_height, takeoff_angle, landing_angle, (left_knee_angles, right_knee_angles) = result
		# Plot hip trajectory
		fig, ax = plt.subplots()
		ax.plot(frames, hip_ys, label="Hip Y Trajectory")
		ax.invert_yaxis()
		ax.set_xlabel("Frame")
		ax.set_ylabel("Hip Height (relative)")
		ax.set_title("Hip Trajectory During Leap")
		st.subheader("Hip Trajectory")
		st.pyplot(fig)
		st.subheader("Jump Analysis")
		st.write(f"- **Peak Jump Height:** {peak_height:.3f} (relative units)")
		st.write(f"- **Takeoff Angle:** {takeoff_angle:.1f}°")
		st.write(f"- **Landing Angle:** {landing_angle:.1f}°")
		st.subheader("Feedback")
		feedback = give_feedback(peak_height, takeoff_angle, landing_angle, left_knee_angles, right_knee_angles)
		for f in feedback:
			st.write(f"- {f}")
		# Optional: Placeholder for future result tracking
		st.markdown("---")
		st.subheader("(Future) Progress Tracking")
		st.info("In a future version, your results will be stored here to track improvement over time.")
else:
	st.info("Please upload a video to begin analysis.")

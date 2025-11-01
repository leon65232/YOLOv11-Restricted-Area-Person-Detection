# save as app.py and run with: streamlit run app.py

import os
import urllib.request
import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
import time

# ---------------------------
# Detect device (GPU if available)
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# UI header
st.title("👀 YOLOv11 Person Detection (Streamlit + WebRTC)")
st.write(f"⚡ This is being run on **{device.upper()}**")

# ---------------------------
# Model download / load
# ---------------------------
MODEL_PATH = "yolov11n.pt"
MODEL_URL = "https://huggingface.co/Ultralytics/YOLOv11/resolve/main/yolov11n.pt?download=true"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading YOLOv11 model..."):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.stop()

# Load model and move to device
model = YOLO(MODEL_PATH).to(device)

# ---------------------------
# Helper: point-in-polygon (ray-casting)
# ---------------------------
def point_in_polygon(x, y, polygon_pts):
    inside = False
    n = len(polygon_pts)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_pts[i]
        xj, yj = polygon_pts[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

# ---------------------------
# Video transformer (stores last frame and restricted area)
# ---------------------------
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.last_frame = None           # store last BGR frame for capture
        self.restricted_shape = None     # ("rect", (x1,y1,x2,y2)) or ("poly", [(x,y), ...])
        self.person_in_restricted = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img.copy()
        person_found = False

        try:
            results = model.predict(img, device=device, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = model.names[cls]
                    if name == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        if self.restricted_shape is not None:
                            if self.restricted_shape[0] == "rect":
                                rx1, ry1, rx2, ry2 = self.restricted_shape[1]
                                if (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2):
                                    person_found = True
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(img, f"INSIDE {conf:.2f}", (x1, y2 + 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                else:
                                    cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            elif self.restricted_shape[0] == "poly":
                                poly = self.restricted_shape[1]
                                if point_in_polygon(cx, cy, poly):
                                    person_found = True
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(img, f"INSIDE {conf:.2f}", (x1, y2 + 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                else:
                                    cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # draw restricted shape on live feed
            if self.restricted_shape is not None:
                if self.restricted_shape[0] == "rect":
                    rx1, ry1, rx2, ry2 = map(int, self.restricted_shape[1])
                    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
                    label = "Person in restricted area" if person_found else "No person in restricted area"
                    cv2.putText(img, label, (rx1, ry2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7 if person_found else 0.6,
                                (0, 0, 255) if person_found else (255, 0, 0), 2)
                elif self.restricted_shape[0] == "poly":
                    poly = [(int(x), int(y)) for (x, y) in self.restricted_shape[1]]
                    for i in range(len(poly)):
                        cv2.line(img, poly[i], poly[(i + 1) % len(poly)], (255, 0, 0), 3)
                    label = "Person in restricted polygon" if person_found else "No person in restricted polygon"
                    cv2.putText(img, label, (poly[0][0], poly[0][1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7 if person_found else 0.6,
                                (0, 0, 255) if person_found else (255, 0, 0), 2)

        except Exception:
            pass

        self.person_in_restricted = bool(person_found)
        return img

# start webrtc streamer
webrtc_ctx = webrtc_streamer(
    key="person-detection",
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# PLACE THE STATUS RIGHT UNDER THE WEBRTC COMPONENT
status_placeholder = st.empty()
status_placeholder.info("Start the camera (use the Start button above) and draw a restricted shape to enable live checks.")

st.write("---")

# ---------------------------
# Capture frame UI (first-frame capture + canvas)
# ---------------------------
col1, col2 = st.columns([1, 1])
with col1:
    captured = st.button("📸 Capture current frame")

with col2:
    st.write("Instructions:")
    st.markdown(
        "- Click **Capture current frame** to grab an image from the stream.  \n"
        "- Use **rect** or **polygon** drawing modes to define the restricted area.  \n"
        "- For polygon: left-click to add points, right-click to close.  \n"
        "- After creating a polygon, click **Use current polygon as restricted area**."
    )

# choose drawing mode
drawing_mode = st.selectbox("Drawing mode", ["rect", "polygon"], index=1)

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "restricted_shape" not in st.session_state:
    st.session_state.restricted_shape = None

# capture exactly one frame when user clicks
if captured:
    if webrtc_ctx and webrtc_ctx.video_transformer:
        frame = webrtc_ctx.video_transformer.last_frame
        if frame is None:
            st.warning("No frame available yet. Make sure your camera is running.")
        else:
            st.session_state.captured_image = frame.copy()
            st.success("Captured frame — draw on the canvas below.")
    else:
        st.warning("Camera not running or transformer not ready.")

# Utility: robustly extract polygon points from canvas object
def extract_polygon_points(obj):
    pts = obj.get("points") or obj.get("path") or obj.get("polylinePoints") or None
    if pts:
        if isinstance(pts, list) and len(pts) > 0 and isinstance(pts[0], list) and len(pts[0]) >= 3 and isinstance(pts[0][0], str):
            out = []
            for item in pts:
                cmd = item[0]
                if cmd in ("M", "L", "m", "l") and len(item) >= 3:
                    x = float(item[1]); y = float(item[2])
                    out.append((x, y))
            return out
        if isinstance(pts, list) and isinstance(pts[0], dict):
            return [(float(p.get("x", 0)), float(p.get("y", 0))) for p in pts]
    path = obj.get("path")
    if path and isinstance(path, list):
        out = []
        for cmd in path:
            if isinstance(cmd, (list, tuple)) and len(cmd) >= 3:
                sym = cmd[0]
                if isinstance(sym, str) and sym.upper() in ("M", "L"):
                    out.append((float(cmd[1]), float(cmd[2])))
        if len(out) > 0:
            return out
    return None

# show canvas to draw restricted shape
if st.session_state.captured_image is not None:
    frame = st.session_state.captured_image
    captured_bgr = frame.copy()
    captured_rgb = cv2.cvtColor(captured_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(captured_rgb)

    st.subheader("Captured frame — draw a restricted shape")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=pil_img,
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode=drawing_mode,
        key="canvas",
        display_toolbar=True,
        update_streamlit=True,
    )

    restricted_rect = None
    restricted_poly = None
    if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
        objs = canvas_result.json_data["objects"]
        for o in objs:
            t = o.get("type", "").lower()
            if t == "rect" and drawing_mode == "rect":
                left = o.get("left", 0)
                top = o.get("top", 0)
                width = o.get("width", 0)
                height = o.get("height", 0)
                restricted_rect = (int(left), int(top), int(left + width), int(top + height))
                break
            if drawing_mode == "polygon" and t in ("polygon", "poly", "polyline", "path"):
                pts = extract_polygon_points(o)
                if pts and len(pts) >= 3:
                    restricted_poly = pts
                    break

    # If polygon found, offer single-button save (no sliders)
    if restricted_poly is not None:
        st.success("Polygon detected. Click to apply it to the live feed.")
        if st.button("Use current polygon as restricted area"):
            st.session_state.restricted_shape = ("poly", restricted_poly)
            if webrtc_ctx and webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.restricted_shape = ("poly", restricted_poly)
            st.success("Polygon applied to live feed.")

    elif restricted_rect is not None:
        st.success(f"Rectangle detected: {restricted_rect}")
        if st.button("Use rectangle as restricted area"):
            st.session_state.restricted_shape = ("rect", restricted_rect)
            if webrtc_ctx and webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.restricted_shape = ("rect", restricted_rect)
            st.success("Rectangle applied to live feed.")

    else:
        st.info("Draw a rectangle or polygon on the canvas to define the restricted area.")

    # Clear button (remove restricted area)
    if st.session_state.get("restricted_shape", None) is not None:
        if st.button("Clear restricted area"):
            st.session_state.restricted_shape = None
            if webrtc_ctx and webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.restricted_shape = None
            st.info("Restricted area cleared from live feed.")

st.write("---")

# ---------------------------
# Live status UPDATER (uses status_placeholder located under the Start/Stop)
# ---------------------------
if webrtc_ctx:
    if webrtc_ctx.state.playing:
        try:
            while webrtc_ctx.state.playing:
                if webrtc_ctx.video_transformer:
                    flag = bool(getattr(webrtc_ctx.video_transformer, "person_in_restricted", False))
                    if flag:
                        status_placeholder.error("🚨 Person detected in the restricted shape (live).")
                    else:
                        if getattr(webrtc_ctx.video_transformer, "restricted_shape", None) is not None:
                            status_placeholder.success("✅ No person detected inside the restricted shape (live).")
                        else:
                            status_placeholder.info("Draw and save a restricted shape on the captured frame to enable live checks.")
                else:
                    status_placeholder.info("Waiting for video transformer...")
                time.sleep(0.5)
        except Exception:
            pass
else:
    status_placeholder.info("Start the camera to get live updates.")

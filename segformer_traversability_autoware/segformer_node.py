# #!/usr/bin/env python3
# # segformer_node.py
# import cv2, torch, numpy as np, rclpy, os, warnings, time
# from rclpy.node import Node
# from sensor_msgs.msg import Image as RosImage
# #from cv_bridge import CvBridge
# from transformers import SegformerForSemanticSegmentation
# from PIL import Image
# # import torchvision.transforms as transforms

# # ✅ Suppress warnings
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=UserWarning)

# # ✅ CUDA optimizations
# os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
# torch.backends.cudnn.benchmark = True
# torch.set_float32_matmul_precision("high")


# class SegFormerTraversability(Node):
#     def __init__(self):
#         super().__init__("segformer_traversability")

#         # ---- Parameters ----
#         self.seg_infer_size = (512, 512)
#         #self.bridge = CvBridge()
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.fp16 = (self.device == "cuda")

#         # ---- Performance Tracking ----
#         self.frame_count = 0
#         self.inference_times = []
#         self.preprocess_times = []
#         self.postprocess_times = []
#         self.total_times = []
#         self.first_frame_time = None
#         self.startup_time = time.time()

#         # ---- Load SegFormer Model (NO feature_extractor!) ----
#         self.get_logger().info("⏳ Loading SegFormer model...")
#         model_start = time.time()

#         model_dir = "/home/nvidia/segimage/models/segformer-b5-ade"
#         self.model = SegformerForSemanticSegmentation.from_pretrained(
#             model_dir,
#             ignore_mismatched_sizes=True,
#             low_cpu_mem_usage=True
#         ).to(self.device, torch.float16 if self.fp16 else torch.float32).eval()

#         model_load_time = time.time() - model_start
#         self.get_logger().info(f"✅ Model loaded in {model_load_time:.2f}s")

#          # ---- Get model config for labels ----
#         self.id2label = self.model.config.id2label
#         self.num_classes = max(self.id2label.keys()) + 1
#         self.get_logger().info(f"✅ {self.num_classes} classes loaded")


#         # ---- Traversability Color Mapping ----
#         self.trav_colors = {
#             "safe": (0, 255, 0),
#             "risky": (0, 255, 255),
#             "blocked": (0, 0, 255)
#         }

#         safe_classes = {
#             "road", "path", "sidewalk", "earth", "ground", "dirt road",
#             "gravel", "sand", "grass", "field", "land", "runway",
#             "floor", "floor tile", "tile", "carpet", "mat", "rug"
#         }

#         risky_classes = {
#             "stairs", "step", "ramp", "hill", "rock", "mountain",
#             "plant", "vegetation", "bush", "flower",
#             "sofa", "chair", "table", "bed", "desk", "bench",
#             "couch", "armchair", "cabinet", "countertop"
#         }

#         blocked_classes = {
#             "wall", "door", "windowpane", "building", "house", "ceiling", "column",
#             "curtain", "refrigerator", "sink", "cupboard", "stove", "microwave",
#             "television", "fence", "railing", "pillar", "person",
#             "vehicle", "car", "truck", "bus", "van", "motorbike",
#             "tree", "palm", "mountain", "rock", "boulder",
#             "water", "sea", "river", "lake", "pool", "sky",
#             "bridge", "fire hydrant", "signboard", "traffic light"
#         }

#         semantic_colors = {
#             "floor": (200, 200, 200), "road": (140, 140, 140),
#             "grass": (4, 250, 7),     "path": (255, 31, 0),
#             "wall": (120, 120, 120),  "building": (180, 120, 120),
#             "sky": (6, 230, 230),     "tree": (4, 200, 3),
#             "table": (255, 6, 82),    "chair": (204, 70, 3),
#             "sofa": (11, 102, 255),   "car": (0, 102, 200),
#             "ceiling": (120, 120, 80),"door": (8, 255, 51),
#             "stairs": (255, 224, 0),  "mountain": (143, 255, 140),
#             "fence": (255, 184, 6),   "sand": (160, 150, 20),
#             "water": (61, 230, 250),  "person": (150, 5, 61)
#         }

#         # Build lookup tables
#         self.id2semantic = np.zeros((self.num_classes, 3), dtype=np.uint8)
#         self.id2trav = np.zeros((self.num_classes, 3), dtype=np.uint8)
#         default_color = np.array([180, 180, 180], dtype=np.uint8)

#         for i in range(self.num_classes):
#             name = self.id2label[i]
#             self.id2semantic[i] = semantic_colors.get(name, default_color)

#             if name in safe_classes:
#                 self.id2trav[i] = self.trav_colors["safe"]
#             elif name in risky_classes:
#                 self.id2trav[i] = self.trav_colors["risky"]
#             else:
#                 self.id2trav[i] = self.trav_colors["blocked"]

#         # ---- Warmup CUDA kernels ----

#         # ---- ROS2 I/O ----
#         self.create_subscription(RosImage, "/camera/camera/color/image_raw", self.image_callback, 10)
#         self.trav_pub = self.create_publisher(RosImage, "/fusion_segmentation/traversability", 10)
#         self.semantic_pub = self.create_publisher(RosImage, "/fusion_segmentation/semantic", 10)

#         total_startup = time.time() - self.startup_time
#         self.get_logger().info("=" * 70)
#         self.get_logger().info(f"✅ Production node ready [{total_startup:.2f}s total startup]")
#         self.get_logger().info("Publishing: /fusion_segmentation/traversability + semantic")
#         self.get_logger().info("=" * 70)

#     def preprocess_image(self, img):
#         """CLAHE preprocessing for low-light enhancement"""
#         lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         l = clahe.apply(l)
#         return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

#     def segformer_infer(self, img_bgr):
#         H, W = img_bgr.shape[:2]
        
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         resized = cv2.resize(img_rgb, self.seg_infer_size, interpolation=cv2.INTER_LINEAR)
        
#         input_tensor = torch.frombuffer(resized.tobytes(), dtype=torch.uint8
#                                     ).view(self.seg_infer_size[1], self.seg_infer_size[0], 3
#                                             ).permute(2,0,1).float().div_(255.0)
        
#         # ✅ CUDA normalization pipeline
#         input_tensor = input_tensor.unsqueeze(0).to(self.device)  # ← ADD THIS!
#         mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3,1,1)
#         std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3,1,1)
#         input_tensor = (input_tensor - mean) / std
        
#         with torch.no_grad():
#             if self.fp16:
#                 with torch.amp.autocast('cuda', dtype=torch.float16):
#                     outputs = self.model(input_tensor)
#                     logits = outputs.logits
#             else:
#                 outputs = self.model(input_tensor)
#                 logits = outputs.logits
        
#         upsampled = torch.nn.functional.interpolate(
#             logits, size=(H, W), mode="bilinear", align_corners=False
#         )
#         # ✅ 100% NUMPY-FREE: Pure Torch → indexed color lookup
#         seg = upsampled.argmax(dim=1)[0]  # [H,W] torch.long
#         seg_cpu = seg.to('cpu')  # Move to CPU for indexing
#         return seg_cpu  # Return torch tensor, not numpy!
        
#     def colorize(self, seg_torch):
#         """Generate semantic and traversability maps - NUMPY SAFE"""
#         # ✅ FIX: Ensure uint8 + clamp
#         seg_np = seg_torch.numpy()
#         seg_clamped = np.clip(seg_np, 0, self.num_classes - 1)
        
#         semantic_map = self.id2semantic[seg_clamped]
#         trav_map = self.id2trav[seg_clamped]

#         # Sky/ceiling detection → mark as blocked
#         h = trav_map.shape[0]
#         if np.mean(trav_map[:h // 5, :, 2]) < 128:
#             trav_map[:h // 6, :] = self.trav_colors["blocked"]

#         return semantic_map, trav_map

#     def image_callback(self, msg):
#         try:
#             frame_start = time.time()

#             # Track first frame
#             if self.first_frame_time is None:
#                 self.first_frame_time = time.time()
#                 time_to_first = self.first_frame_time - self.startup_time
#                 self.get_logger().info(f"🎯 First frame received [{time_to_first:.2f}s from startup]")

#             self.frame_count += 1

#             # MANUAL conversion (bypass cv_bridge NumPy conflict)
#             preprocess_start = time.time()
#             height = msg.height
#             width = msg.width
#             if msg.encoding == "rgb8":
#                 img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
#             elif msg.encoding == "bgr8":
#                 img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
#             else:  # mono16/32FC1 → grayscale
#                 img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width))
#                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
#             img = self.preprocess_image(img)
#             preprocess_time = time.time() - preprocess_start

#             # Inference + publish (unchanged)
#             inference_start = time.time()
#             seg_np = self.segformer_infer(img)
#             inference_time = time.time() - inference_start

#             postprocess_start = time.time()
#             semantic_map, trav_map = self.colorize(seg_np)
#             postprocess_time = time.time() - postprocess_start

#             # MANUAL publish (bypass cv_bridge)
#             msg_sem = RosImage()
#             msg_sem.height = semantic_map.shape[0]
#             msg_sem.width = semantic_map.shape[1]
#             msg_sem.encoding = 'bgr8'
#             msg_sem.is_bigendian = False
#             msg_sem.step = semantic_map.strides[0]
#             msg_sem.data = semantic_map.tobytes()
#             msg_sem.header = msg.header
#             self.semantic_pub.publish(msg_sem)

#             msg_trav = RosImage()
#             msg_trav.height = trav_map.shape[0]
#             msg_trav.width = trav_map.shape[1]
#             msg_trav.encoding = 'bgr8'
#             msg_trav.is_bigendian = False
#             msg_trav.step = trav_map.strides[0]
#             msg_trav.data = trav_map.tobytes()
#             msg_trav.header = msg.header
#             self.trav_pub.publish(msg_trav)

#             # Timing stats
#             total_time = time.time() - frame_start
#             self.preprocess_times.append(preprocess_time)
#             self.inference_times.append(inference_time)
#             self.postprocess_times.append(postprocess_time)
#             self.total_times.append(total_time)

#             if self.frame_count % 10 == 0:
#                 avg_total = np.mean(self.total_times[-10:])
#                 avg_inference = np.mean(self.inference_times[-10:])
#                 fps = 1.0 / avg_total if avg_total > 0 else 0
#                 self.get_logger().info(
#                     f"Frame {self.frame_count:4d} | "
#                     f"Preproc: {avg_preprocess*1000:5.1f}ms | "
#                     f"Inference: {avg_inference*1000:5.1f}ms | "
#                     f"Total: {avg_total*1000:5.1f}ms ({fps:.1f} FPS)"
#                 )

#         except Exception as e:
#             self.get_logger().error(f"Error: {e}")


# def main():
#     rclpy.init()
#     node = SegFormerTraversability()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         # Print final statistics
#         if node.total_times:
#             avg_total = np.mean(node.total_times)
#             avg_inference = np.mean(node.inference_times)
#             avg_preprocess = np.mean(node.preprocess_times)
#             avg_postprocess = np.mean(node.postprocess_times)
#             fps = 1.0 / avg_total if avg_total > 0 else 0

#             print("\n" + "=" * 70)
#             print("📊 FINAL PERFORMANCE STATISTICS")
#             print("=" * 70)
#             print(f"Total Frames Processed: {node.frame_count}")
#             print(f"Average Preprocessing:  {avg_preprocess*1000:6.2f}ms")
#             print(f"Average Inference:      {avg_inference*1000:6.2f}ms")
#             print(f"Average Postprocessing: {avg_postprocess*1000:6.2f}ms")
#             print(f"Average Total Time:     {avg_total*1000:6.2f}ms")
#             print(f"Average FPS:            {fps:6.2f}")
#             print("=" * 70)

#         try:
#             node.destroy_node()
#         except:
#             pass
#         try:
#             if rclpy.ok():
#                 rclpy.shutdown()
#         except:
#             pass


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# segformer_node.py
import cv2, torch, numpy as np, rclpy, os, warnings, time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# CUDA optimizations
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class SegFormerTraversability(Node):
    def __init__(self):
        super().__init__("segformer_traversability")

        # ---- Parameters ----
        self.seg_infer_size = (512, 512)
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = (self.device == "cuda")

        # ---- Performance Tracking ----
        self.frame_count = 0
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        self.total_times = []
        self.first_frame_time = None
        self.startup_time = time.time()

        # ---- Load SegFormer Model ----
        self.get_logger().info("⏳ Loading SegFormer model...")
        model_start = time.time()

        model_dir = "/home/nvidia/segimage/models/segformer-b5-ade"

        self.feature_extractor = SegformerImageProcessor.from_pretrained(
            model_dir, do_resize=False, do_normalize=True
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        ).to(self.device, torch.float16 if self.fp16 else torch.float32).eval()

        model_load_time = time.time() - model_start
        self.get_logger().info(f"✅ Model loaded in {model_load_time:.2f}s")

        id2label = self.model.config.id2label
        self.num_classes = max(id2label.keys()) + 1
        self.get_logger().info(f"✅ {self.num_classes} classes loaded")

        # ---- Traversability Color Mapping ----
        self.trav_colors = {
            "safe":    (0, 255, 0),
            "risky":   (0, 255, 255),
            "blocked": (0, 0, 255)
        }

        safe_classes = {
            "road", "path", "sidewalk", "earth", "ground", "dirt road",
            "gravel", "sand", "grass", "field", "land", "runway",
            "floor", "floor tile", "tile", "carpet", "mat", "rug"
        }

        risky_classes = {
            "stairs", "step", "ramp", "hill", "rock", "mountain",
            "plant", "vegetation", "bush", "flower",
            "sofa", "chair", "table", "bed", "desk", "bench",
            "couch", "armchair", "cabinet", "countertop"
        }

        semantic_colors = {
            "floor":    (200, 200, 200), "road":     (140, 140, 140),
            "grass":    (4, 250, 7),     "path":     (255, 31, 0),
            "wall":     (120, 120, 120), "building": (180, 120, 120),
            "sky":      (6, 230, 230),   "tree":     (4, 200, 3),
            "table":    (255, 6, 82),    "chair":    (204, 70, 3),
            "sofa":     (11, 102, 255),  "car":      (0, 102, 200),
            "ceiling":  (120, 120, 80),  "door":     (8, 255, 51),
            "stairs":   (255, 224, 0),   "mountain": (143, 255, 140),
            "fence":    (255, 184, 6),   "sand":     (160, 150, 20),
            "water":    (61, 230, 250),  "person":   (150, 5, 61)
        }

        # Build lookup tables
        self.id2semantic = np.zeros((self.num_classes, 3), dtype=np.uint8)
        self.id2trav     = np.zeros((self.num_classes, 3), dtype=np.uint8)
        default_color    = np.array([180, 180, 180], dtype=np.uint8)

        for i in range(self.num_classes):
            name = id2label[i]
            self.id2semantic[i] = semantic_colors.get(name, default_color)

            if name in safe_classes:
                self.id2trav[i] = self.trav_colors["safe"]
            elif name in risky_classes:
                self.id2trav[i] = self.trav_colors["risky"]
            else:
                self.id2trav[i] = self.trav_colors["blocked"]

        # ---- Warmup CUDA kernels ----
        self.get_logger().info("⏳ Warming up CUDA kernels...")
        warmup_start = time.time()

        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_resized = cv2.resize(dummy, self.seg_infer_size)
        inputs = self.feature_extractor(images=dummy_resized, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.fp16:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _ = self.model(**inputs).logits
            else:
                _ = self.model(**inputs).logits

        if self.device == "cuda":
            torch.cuda.synchronize()

        warmup_time = time.time() - warmup_start
        self.get_logger().info(f"✅ Warmup completed in {warmup_time:.2f}s")

        # ---- ROS2 I/O ----
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, 10)
        self.trav_pub     = self.create_publisher(Image, "/fusion_segmentation/traversability", 10)
        self.semantic_pub = self.create_publisher(Image, "/fusion_segmentation/semantic", 10)

        total_startup = time.time() - self.startup_time
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"✅ Production node ready [{total_startup:.2f}s total startup]")
        self.get_logger().info("Publishing: /fusion_segmentation/traversability + semantic")
        self.get_logger().info("=" * 70)

    def preprocess_image(self, img):
        """CLAHE preprocessing for low-light enhancement"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def segformer_infer(self, img_bgr):
        """SegFormer inference"""
        H, W = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, self.seg_infer_size, interpolation=cv2.INTER_LINEAR)
        inputs = self.feature_extractor(images=resized, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad():
            if self.fp16:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = self.model(**inputs).logits
            else:
                logits = self.model(**inputs).logits

        upsampled = torch.nn.functional.interpolate(
            logits.float(), size=(H, W), mode="bilinear", align_corners=False
        )
        seg = upsampled.argmax(dim=1)[0]
        return seg.detach().cpu().numpy()

    def colorize(self, seg_np):
        """Generate semantic and traversability maps"""
        seg_clamped  = np.clip(seg_np, 0, self.num_classes - 1)
        semantic_map = self.id2semantic[seg_clamped]
        trav_map     = self.id2trav[seg_clamped].copy()  # copy so we can safely write to it

        # Top strip → mark as blocked if not already sky-colored
        h = trav_map.shape[0]
        if np.mean(trav_map[:h // 5, :, 2]) < 128:
            trav_map[:h // 6, :] = self.trav_colors["blocked"]

        return semantic_map, trav_map

    def image_callback(self, msg):
        try:
            frame_start = time.time()

            if self.first_frame_time is None:
                self.first_frame_time = time.time()
                time_to_first = self.first_frame_time - self.startup_time
                self.get_logger().info(f"🎯 First frame received [{time_to_first:.2f}s from startup]")

            self.frame_count += 1

            # Preprocess
            preprocess_start = time.time()
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = self.preprocess_image(img)
            preprocess_time = time.time() - preprocess_start

            # Inference
            inference_start = time.time()
            seg_np = self.segformer_infer(img)
            inference_time = time.time() - inference_start

            # Colorize
            postprocess_start = time.time()
            semantic_map, trav_map = self.colorize(seg_np)
            postprocess_time = time.time() - postprocess_start

            # Publish
            msg_sem = self.bridge.cv2_to_imgmsg(semantic_map, "bgr8")
            msg_sem.header = msg.header
            self.semantic_pub.publish(msg_sem)

            msg_trav = self.bridge.cv2_to_imgmsg(trav_map, "bgr8")
            msg_trav.header = msg.header
            self.trav_pub.publish(msg_trav)

            # Track timing
            total_time = time.time() - frame_start
            self.preprocess_times.append(preprocess_time)
            self.inference_times.append(inference_time)
            self.postprocess_times.append(postprocess_time)
            self.total_times.append(total_time)

            # Log every 10 frames
            if self.frame_count % 10 == 0:
                avg_preprocess  = np.mean(self.preprocess_times[-10:])
                avg_inference   = np.mean(self.inference_times[-10:])
                avg_postprocess = np.mean(self.postprocess_times[-10:])
                avg_total       = np.mean(self.total_times[-10:])
                fps = 1.0 / avg_total if avg_total > 0 else 0

                self.get_logger().info(
                    f"Frame {self.frame_count:4d} | "
                    f"Preproc: {avg_preprocess*1000:5.1f}ms | "
                    f"Inference: {avg_inference*1000:5.1f}ms | "
                    f"Postproc: {avg_postprocess*1000:5.1f}ms | "
                    f"Total: {avg_total*1000:5.1f}ms ({fps:.1f} FPS)"
                )

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}", throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = SegFormerTraversability()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.total_times:
            avg_total       = np.mean(node.total_times)
            avg_inference   = np.mean(node.inference_times)
            avg_preprocess  = np.mean(node.preprocess_times)
            avg_postprocess = np.mean(node.postprocess_times)
            fps = 1.0 / avg_total if avg_total > 0 else 0

            print("\n" + "=" * 70)
            print("📊 FINAL PERFORMANCE STATISTICS")
            print("=" * 70)
            print(f"Total Frames Processed: {node.frame_count}")
            print(f"Average Preprocessing:  {avg_preprocess*1000:6.2f}ms")
            print(f"Average Inference:      {avg_inference*1000:6.2f}ms")
            print(f"Average Postprocessing: {avg_postprocess*1000:6.2f}ms")
            print(f"Average Total Time:     {avg_total*1000:6.2f}ms")
            print(f"Average FPS:            {fps:6.2f}")
            print("=" * 70)

        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

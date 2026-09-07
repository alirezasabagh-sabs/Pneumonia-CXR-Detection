"""
lung_segmenter.py
===================
این‌جا چیکار می‌کنه:
    کلاس LungSegmenter مسئول سه کار جداست:
      1) حذف حاشیه‌ی سیاه اطراف تصویر (collimation artifact دستگاه X-ray)
      2) اجرای مدل DL پیش‌آموزش‌دیده (ResNet34 از کتابخونه‌ی lungs_segmentation)
         برای پیدا کردن ماسک تقریبی ریه چپ/راست
      3) استفاده از اون ماسک برای برش نهایی تصویر، به یکی از دو روش:
           - "bbox" (پیش‌فرض، توصیه‌شده): فقط مستطیل دور ریه با حاشیه‌ی زیاد
             کراپ می‌شه، هیچ پیکسلی سیاه/ماسک نمی‌شه. طبق تحقیق و شواهد
             (تیم‌های برنده‌ی RSNA Challenge هم همینو انجام دادن، و مطالعات
             ۲۰۲۵-۲۰۲۶ نشون دادن ماسک‌گذاری سخت context بالینی مفید رو حذف
             می‌کنه و حتی می‌تونه باعث یادگیری shortcut از لبه‌ی ماسک بشه).
           - "pixel_mask" (قدیمی/اختیاری، فقط برای مقایسه/ablation): برش
             پیکسلی سخت با feathering، همون روشی که اول پروژه پیاده کردیم.

    چرا دیگه dilation/convex-hull/feathering پیش‌فرض نیستن:
        همه‌شون patch هایی بودن برای مشکلاتی که ریشه‌شون خودِ "ماسک‌گذاری
        پیکسلی سخت" بود. با رفتن سراغ حالت "bbox" (که فقط یه مستطیل شل
        کراپ می‌کنه، نه ماسک‌گذاری دقیق)، اصلاً به این workaround ها نیاز
        نیست: نه لبه‌ی مصنوعی داریم (پس shortcut یادگیری لبه بی‌معنیه)، نه
        خطر بریدن لوب/پاتولوژی (چون margin مستطیل سخاوتمندانه‌ست).

استفاده‌ی رایج:
    segmenter = LungSegmenter()
    mask = segmenter.predict_mask(image_path)
    cropped = segmenter.crop_to_lung_bbox(original_image_array, mask)
"""

import torch
import numpy as np
import cv2
from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference


class LungSegmenter:
    def __init__(self, device=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # استفاده از مدل ResNet34 که گفتی نتیجه عالی داده
        self.model = create_model("resnet34")
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Success: Model loaded using golden logic on {self.device}")

    # ------------------------------------------------------------------
    # 1. حذف حاشیه‌ی سیاه اطراف تصویر (collimation)
    # ------------------------------------------------------------------
    @staticmethod
    def detect_content_bbox(image_np, black_threshold: int = 10):
        """پیدا کردن bounding box ناحیه‌ی واقعی تصویر (غیر از حاشیه‌ی سیاه اطراف)."""
        gray = image_np if image_np.ndim == 2 else cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        mask = gray > black_threshold
        coords = np.argwhere(mask)
        if coords.size == 0:
            h, w = gray.shape
            return 0, 0, w, h
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)

    @staticmethod
    def crop_black_borders(image_np, black_threshold: int = 10):
        """تصویر رو به bounding box محتوای واقعی (بدون حاشیه سیاه) کراپ می‌کنه."""
        x, y, w, h = LungSegmenter.detect_content_bbox(image_np, black_threshold)
        return image_np[y:y + h, x:x + w]

    # ------------------------------------------------------------------
    # 2. Fallback کلاسیک (وقتی مدل DL خطا بده)
    # ------------------------------------------------------------------
    @staticmethod
    def _classical_fallback_mask(image_np):
        """
        اگه مدل DL خطا داد، به‌جای ماسک صفر، یک تخمین کلاسیک برمی‌گردونه.
        نکته: توی CXR، ریه‌ها (پر از هوا) تیره‌ترن؛ پس پیکسل‌های تیره‌تر رو
        foreground می‌گیریم (THRESH_BINARY_INV روی percentile پایین)، نه
        پیکسل‌های روشن‌تر (استخوان/بافت نرم).
        """
        gray = image_np if image_np.ndim == 2 else cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (512, 512))

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        content_pixels = blurred[blurred > 10]
        thresh_val = np.percentile(content_pixels, 35) if len(content_pixels) > 0 else 127

        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        binary = (binary > 0).astype(np.uint8)
        return LungSegmenter.clean_mask(binary.astype(np.float32), binary_threshold=0.5)

    # ------------------------------------------------------------------
    # 3. پیش‌بینی ماسک اصلی (DL + fallback)
    # ------------------------------------------------------------------
    def predict_mask(self, image_path, clean: bool = True, remove_black_borders: bool = True,
                      mask_binary_threshold: float = 0.6, use_convex_hull: bool = False):
        """
        ماسک تقریبی ریه رو برمی‌گردونه. این ماسک لازم نیست پیکسل‌به‌پیکسل کامل
        دقیق باشه -- چون فقط برای پیدا کردن bounding box نهایی استفاده می‌شه
        (نه ماسک‌گذاری مستقیم پیکسل)، یه تخمین معقول کافیه.

        use_convex_hull: پیش‌فرض False. توی تست‌های قبلی، وقتی True بود باعث
        می‌شد شونه/بازو هم به اشتباه جزو ریه حساب بشه. فقط برای موارد نادر
        (پاتولوژی شدید که لوب رو تیکه‌تیکه کنه) به True تغییرش بده.
        """
        import tempfile
        import os

        temp_path = None
        try:
            target_path = image_path
            if remove_black_borders:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"cv2.imread نتونست فایل رو بخونه: {image_path}")
                cropped = self.crop_black_borders(img)

                fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                cv2.imwrite(temp_path, cropped)
                target_path = temp_path

            masks = inference.inference(self.model, target_path, thresh=0.5)
            combined = self.unify_mask(masks, use_convex_hull=use_convex_hull)
            if clean:
                combined = self.clean_mask(combined, binary_threshold=mask_binary_threshold)
            return combined
        except Exception as e:
            print(f"⚠️ DL Inference خطا داد ({e})، در حال استفاده از fallback کلاسیک...")
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("cv2.imread نتونست فایل رو بخونه")
                if remove_black_borders:
                    img = self.crop_black_borders(img)
                return self._classical_fallback_mask(img)
            except Exception as e2:
                print(f"❌ Fallback کلاسیک هم خطا داد: {e2}")
                return np.zeros((512, 512), dtype=np.float32)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

    # ------------------------------------------------------------------
    # 4. پاکسازی نویز ماسک (Open→Close→بزرگ‌ترین دو ناحیه→پر کردن حفره)
    # ------------------------------------------------------------------
    @staticmethod
    def clean_mask(mask_np, binary_threshold: float = 0.6, keep_top_n: int = 2):
        """
        حذف نویز پراکنده از ماسک خام مدل. چون هدف نهایی فقط پیدا کردن یه
        bounding box معقوله (نه ماسک دقیق)، این پاکسازی صرفاً جلوی این رو
        می‌گیره که یه لکه‌ی اشتباه دوردست (مثلاً گوشه‌ی تصویر) باعث بشه
        bounding box بی‌جهت بزرگ بشه.
        """
        from scipy.ndimage import binary_fill_holes

        binary = (mask_np > binary_threshold).astype(np.uint8)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return mask_np

        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)
        keep_ids = {i for i, _ in areas[:keep_top_n]}

        cleaned = np.isin(labels, list(keep_ids)).astype(np.uint8)
        cleaned = binary_fill_holes(cleaned).astype(np.float32)
        return cleaned

    def unify_mask(self, mask_list, use_convex_hull: bool = False):
        """ترکیب لیست ماسک‌ها (هر عضو یک ریه، چپ یا راست) به یک ماتریس واحد."""
        combined = np.zeros((512, 512), dtype=np.float32)
        for m in mask_list:
            m_np = np.array(m).astype(np.float32)

            if m_np.ndim == 3:
                if m_np.shape[0] < 10:
                    m_np = np.max(m_np, axis=0)
                else:
                    m_np = np.max(m_np, axis=2)

            if m_np.shape != (512, 512):
                m_np = cv2.resize(m_np, (512, 512))

            if use_convex_hull:
                m_np = self._apply_convex_hull(m_np)

            combined = np.maximum(combined, m_np)
        return combined

    @staticmethod
    def _apply_convex_hull(single_lung_mask, binary_threshold: float = 0.5, min_area_ratio: float = 0.15):
        """Convex Hull یک ماسک تک‌ریه (فقط اگه use_convex_hull=True فعال بشه)."""
        binary = (single_lung_mask > binary_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return single_lung_mask

        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas)
        significant = [c for c, a in zip(contours, areas) if a >= min_area_ratio * max_area]

        all_points = np.vstack(significant)
        hull = cv2.convexHull(all_points)

        hull_mask = np.zeros_like(binary)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        return hull_mask.astype(np.float32)

    # ------------------------------------------------------------------
    # 5. برش نهایی — روش پیش‌فرض توصیه‌شده: bounding box شل (بدون ماسک‌گذاری پیکسل)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_mask_prior_grid(mask_np, grid_side: int):
        """
        ماسک ریه (معمولاً 512x512) رو به یک grid کوچیک (مثلاً 16x16، هم‌اندازه‌ی
        patch grid ViT) resize می‌کنه و flatten می‌کنه — این چیزیه که به
        AttentionPooling به‌عنوان mask_prior داده می‌شه.

        از INTER_AREA استفاده می‌کنیم (نه nearest/bilinear ساده) چون این
        روش میانگین‌گیری صحیح روی ناحیه‌ی هر patch رو انجام می‌ده (نه صرفاً
        نمونه‌برداری یک نقطه)، که برای یک "درصد پوشش ریه در این patch" دقیق‌تره.

        Returns:
            آرایه‌ی float32 به شکل (grid_side*grid_side,) با مقادیر بین ۰ و ۱
        """
        resized = cv2.resize(mask_np.astype(np.float32), (grid_side, grid_side), interpolation=cv2.INTER_AREA)
        resized = np.clip(resized, 0.0, 1.0)
        return resized.flatten()

    @staticmethod
    def compute_distance_prior_grid(mask_np, grid_side: int, binary_threshold: float = 0.5):
        """
        (جدید) جایگزین/مکمل compute_mask_prior_grid: به‌جای «چند درصد این patch
        پوشش ریه داره» (که نزدیک لبه و عمق ریه رو یکسان امتیاز می‌ده)، اینجا
        فاصله‌ی هر پیکسل ریه از نزدیک‌ترین مرز (Distance Transform) رو حساب
        می‌کنیم:
            - وسط ریه (دور از لبه)  -> فاصله‌ی زیاد -> بعد از نرمالایز نزدیک ۱
            - نزدیک لبه‌ی ریه        -> فاصله‌ی کم   -> مقدار کوچیک (نه صفر)
            - کاملاً بیرون ریه        -> صفر

        چرا این بهتره: در تست‌های عملی (Grad-CAM + occlusion sensitivity)
        دیده شد که مدل با soft mask_prior قدیمی (که فقط coverage رو نشون
        می‌داد) هنوز شدیداً به context بیرون ریه (شونه، دیافراگم) و/یا به نوار
        نزدیک لبه‌ی ریه تکیه می‌کرد. با این نسخه، حتی patch های *داخل* ماسک
        هم اگه نزدیک لبه باشن امتیاز کمتری می‌گیرن، پس مدل بیشتر به‌سمت عمق
        بافت ریه (جایی که کدورت پاتولوژیک واقعی معمولاً دیده می‌شه) سوق داده
        می‌شه، نه به لبه‌ی ماسک.

        Returns:
            آرایه‌ی float32 به شکل (grid_side*grid_side,) با مقادیر بین ۰ و ۱
        """
        binary = (mask_np > binary_threshold).astype(np.uint8)

        if binary.sum() == 0:
            # ماسک خالیه -> برگردوندن صفر یکنواخت (بدون bias، بهتر از crash)
            return np.zeros((grid_side * grid_side,), dtype=np.float32)

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        max_dist = dist.max()
        if max_dist > 0:
            dist = dist / max_dist
        dist = dist.astype(np.float32)

        resized = cv2.resize(dist, (grid_side, grid_side), interpolation=cv2.INTER_AREA)
        resized = np.clip(resized, 0.0, 1.0)
        return resized.flatten()

    @staticmethod
    def crop_to_lung_bbox(image_np, mask_np, margin_ratio: float = 0.15, debug: bool = True):
        """
        روش پیش‌فرض و توصیه‌شده (طبق شواهد تحقیق): مستطیل دور ریه (bounding box
        ماسک) رو با یه حاشیه‌ی سخاوتمندانه (margin_ratio) کراپ می‌کنه، بدون
        این‌که هیچ پیکسلی داخل مستطیل سیاه/ماسک بشه.
        """
        h_mask, w_mask = mask_np.shape
        binary = (mask_np > 0.5).astype(np.uint8)
        coords = np.argwhere(binary > 0)

        if coords.size == 0:
            if debug:
                print("⚠️ [crop_to_lung_bbox] ماسک کاملاً خالیه! کل تصویر برگردونده می‌شه.")
            return image_np

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        raw_area_ratio = binary.sum() / binary.size

        box_h, box_w = (y1 - y0), (x1 - x0)
        margin_y = int(box_h * margin_ratio)
        margin_x = int(box_w * margin_ratio)

        y0m = max(0, y0 - margin_y)
        y1m = min(h_mask - 1, y1 + margin_y)
        x0m = max(0, x0 - margin_x)
        x1m = min(w_mask - 1, x1 + margin_x)

        h_img, w_img = image_np.shape[:2]
        scale_y, scale_x = h_img / h_mask, w_img / w_mask

        iy0, iy1 = int(y0m * scale_y), int(y1m * scale_y)
        ix0, ix1 = int(x0m * scale_x), int(x1m * scale_x)

        if debug:
            kept_ratio = ((iy1 - iy0) * (ix1 - ix0)) / (h_img * w_img)
            print(
                f"🔍 [crop_to_lung_bbox] mask raw area ratio={raw_area_ratio:.2%} | "
                f"bbox (mask-space) y=[{y0},{y1}] x=[{x0},{x1}] از {h_mask}x{w_mask} | "
                f"بعد از margin: y=[{y0m},{y1m}] x=[{x0m},{x1m}] | "
                f"final crop = {iy1-iy0}x{ix1-ix0} از {h_img}x{w_img} اصلی "
                f"({kept_ratio:.1%} از مساحت نگه داشته شد)"
            )

        return image_np[iy0:iy1, ix0:ix1]

    # ------------------------------------------------------------------
    # 6. (اختیاری/legacy) برش پیکسلی سخت با feathering — فقط برای ablation
    # ------------------------------------------------------------------
    @staticmethod
    def crop_lungs_from_original(image_np, mask_np, threshold=0.4, feather_px: int = 15):
        """
        روش قدیمی: ماسک‌گذاری پیکسلی (نه فقط crop مستطیلی). طبق شواهد تحقیق
        دیگه توصیه نمی‌شه (نگاه کن به docstring بالای فایل)، ولی برای مقایسه/
        ablation study نگه داشته شده. feather_px مرز رو نرم می‌کنه تا حداقل
        لبه‌ی مصنوعی تیز نداشته باشیم.
        """
        h, w = mask_np.shape
        image_res = cv2.resize(image_np, (w, h)) if image_np.shape[:2] != (h, w) else image_np

        binary_mask = (mask_np > threshold).astype(np.float32)

        if feather_px > 0:
            ksize = feather_px * 2 + 1
            soft_mask = cv2.GaussianBlur(binary_mask, (ksize, ksize), 0)
            soft_mask = np.clip(soft_mask, 0.0, 1.0)
        else:
            soft_mask = binary_mask

        if len(image_res.shape) == 3:
            alpha = np.stack([soft_mask] * 3, axis=-1)
        else:
            alpha = soft_mask

        cropped = (image_res.astype(np.float32) * alpha).astype(np.uint8)
        return cropped

    @staticmethod
    def dilate_mask(mask_np, pixel_radius: int = 12):
        """
        (Legacy/اختیاری، دیگه پیش‌فرض صدا زده نمی‌شه) دیلیت ماسک. با رفتن
        سراغ crop_to_lung_bbox، دیگه نیازی به این نیست چون bbox خودش margin
        کافی داره. فقط اگه خواستی از crop_lungs_from_original قدیمی (pixel
        mask) استفاده کنی و بازم dilation جدا خواستی، نگه داشته شده.
        """
        binary = (mask_np > 0.5).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pixel_radius * 2 + 1, pixel_radius * 2 + 1)
        )
        dilated = cv2.dilate(binary, kernel, iterations=1)
        return (dilated > 0).astype(np.float32)


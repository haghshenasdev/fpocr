# OCR فارسی آفلاین برای Flutter Windows

این بسته برای Flutter/Windows ساخته شده و از PP-OCRv5 ONNX استفاده می‌کند:

- Detection: `PP-OCRv5_mobile_det.onnx`
- Recognition: `arabic_PP-OCRv5_mobile_rec.onnx`
- Dictionary: `ppocrv5_arabic_dict.txt`
- Runtime: `flutter_onnxruntime`
- تصویر: `image`
- بدون Python
- بدون OpenCV
- بدون Tesseract
- بدون اینترنت هنگام اجرا

مدل Arabic/Persian خود PaddleOCR از فارسی نیز پشتیبانی می‌کند. دیکشنری باید دقیقاً متعلق به همین مدل باشد.

## 1. فایل‌ها را کجا کپی کنم؟

محتویات `lib/` را در پروژه خود قرار دهید:

```text
lib/
  persian_ocr.dart
  src/
    persian_ocr.dart
    persian_ocr_models.dart
    persian_ocr_normalizer.dart
```

و این پوشه را بسازید:

```text
assets/models/
```

سپس سه فایل مدل را در آن قرار دهید.

## 2. dependency

محتویات `pubspec_ocr_additions.yaml` را با `pubspec.yaml` پروژه خود ادغام کنید.

بعد:

```powershell
flutter clean
flutter pub get
```

## 3. مدل‌ها

به علت حجم فایل‌های ONNX، مدل‌های باینری داخل این ZIP قرار نگرفته‌اند. فایل `download_models.ps1` لینک و SHA256 مورد انتظار را دارد و در Windows آنها را داخل `assets/models` قرار می‌دهد.

پس از دانلود باید این سه فایل وجود داشته باشند:

```text
assets/models/PP-OCRv5_mobile_det.onnx
assets/models/arabic_PP-OCRv5_mobile_rec.onnx
assets/models/ppocrv5_arabic_dict.txt
```

## 4. استفاده

```dart
import 'dart:io';
import 'package:your_app/persian_ocr.dart';

final ocr = PersianOcr();
await ocr.initialize();

final result = await ocr.recognizeBytes(
  await File(imagePath).readAsBytes(),
);

print(result.text);
print(result.elapsed);
```

برای dispose هنگام خروج صفحه/برنامه:

```dart
await ocr.dispose();
```

## 5. نکته مهم درباره خروجی فارسی

مدل Arabic/Persian یک dictionary اختصاصی دارد و ترتیب RTL آن باید در post-processing اصلاح شود. این بسته همین کار را انجام می‌دهد و سپس `ي/ى/ك` و اعراب و ارقام را برای استفاده در سیستم دبیرخانه نرمال می‌کند.

## 6. اتصال به RecordForm دبیرخانه

اگر می‌خواهید نتیجه OCR وارد فیلدهای `onvan`، `saheb_name`، `date` و ... شود، OCR را جدا از parser نگه دارید:

```text
Image
  ↓
PersianOcr
  ↓
PersianOcrResult.text
  ↓
FieldExtractor / Regex
  ↓
RecordForm
```

مدل OCR نباید با Qwen یا LLM ترکیب شود؛ برای سرعت Windows CPU بهتر است استخراج فیلدها بعد از OCR با regex و قواعد ساده انجام شود.

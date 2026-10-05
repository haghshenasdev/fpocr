import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'src/persian_ocr_models.dart';
import 'src/persian_ocr_normalizer.dart';

class PersianOcr {
  static const String detAsset =
      'assets/models/PP-OCRv5_mobile_det.onnx';

  static const String recAsset =
      'assets/models/arabic_PP-OCRv5_mobile_rec.onnx';

  static const String dictAsset =
      'assets/models/ppocrv5_arabic_dict.txt';

  final OnnxRuntime _ort = OnnxRuntime();

  OrtSession? _detSession;
  OrtSession? _recSession;

  List<String>? _dictionary;

  bool _initialized = false;

  // ============================================================
  // تنظیمات
  // ============================================================

  int maxSide = 1600;

  double detThreshold = 0.30;

  double boxThreshold = 0.55;

  int maxLines = 120;

  /*
   * حداکثر عرض نسبی یک خط قبل از تقسیم شدن.
   *
   * این مقدار را عمداً بالا نگه داشته‌ایم تا
   * خطوط معمولی وارد chunking نشوند.
   */
  double longLineRatio = 7.0;

  /*
   * مقدار overlap بین دو قطعه.
   */
  double chunkOverlapRatio = 0.18;

  bool get isInitialized => _initialized;

  // ============================================================
  // INITIALIZE
  // ============================================================

  Future<void> initialize() async {
    if (_initialized) return;

    _detSession =
        await _ort.createSessionFromAsset(detAsset);

    _recSession =
        await _ort.createSessionFromAsset(recAsset);

    final raw =
        await rootBundle.loadString(dictAsset);

    final dictionary = raw
        .replaceAll('\r', '')
        .split('\n')
        .where((e) => e.isNotEmpty)
        .toList();

    /*
     * PP-OCRv5 Arabic dictionary:
     *
     * 747 characters
     *
     * class 0 = blank
     * class 1..747 = dictionary
     * class 748 = space
     *
     * total = 749
     */
    if (dictionary.length != 747) {
      throw StateError(
        'Invalid PP-OCRv5 Arabic dictionary. '
        'Expected 747 characters, '
        'found ${dictionary.length}.',
      );
    }

    _dictionary =
        List<String>.unmodifiable(dictionary);

    _initialized = true;
  }

  // ============================================================
  // PUBLIC
  // ============================================================

  Future<PersianOcrResult> recognizeBytes(
    Uint8List bytes,
  ) async {
    await initialize();

    final started = DateTime.now();

    final image =
        img.decodeImage(bytes);

    if (image == null) {
      throw const FormatException(
        'Cannot decode input image.',
      );
    }

    return _recognizeImage(
      image,
      started,
    );
  }

  Future<PersianOcrResult> recognizeFile(
    String path,
  ) async {
    throw UnsupportedError(
      'Use recognizeBytes(await File(path).readAsBytes()) '
      'so this service stays platform-neutral.',
    );
  }

  // ============================================================
  // MAIN
  // ============================================================

  Future<PersianOcrResult> _recognizeImage(
    img.Image source,
    DateTime started,
  ) async {
    final scale = math.min(
      1.0,
      maxSide /
          math.max(
            source.width,
            source.height,
          ),
    );

    final image = scale < 0.999
        ? img.copyResize(
            source,
            width: math.max(
              1,
              (source.width * scale).round(),
            ),
            height: math.max(
              1,
              (source.height * scale).round(),
            ),
            interpolation:
                img.Interpolation.linear,
          )
        : source;

    final boxes =
        await _detect(image);

    final lines =
        <PersianOcrLine>[];

    /*
     * مهم:
     *
     * ترتیب قبلی detector را حفظ می‌کنیم.
     */
    for (final box
        in boxes.take(maxLines)) {
      final crop =
          _cropBox(
        image,
        box,
      );

      if (crop == null) {
        continue;
      }

      if (crop.width < 4 ||
          crop.height < 4) {
        continue;
      }

      /*
       * تنها تغییر اصلی:
       *
       * اگر خط بلند باشد، _recognizeLongCrop
       * خودش آن را تقسیم می‌کند.
       */
      final recognized =
          await _recognizeLongCrop(
        crop,
      );

      if (recognized.text.isEmpty) {
        continue;
      }

      lines.add(
        PersianOcrLine(
          text:
              PersianOcrNormalizer.normalize(
            recognized.text,
          ),
          confidence:
              recognized.confidence,
          left: box.left,
          top: box.top,
          right: box.right,
          bottom: box.bottom,
        ),
      );
    }

    /*
     * همان sort قبلی
     */
    lines.sort((a, b) {
      final dy =
          a.top.compareTo(
        b.top,
      );

      if (dy.abs() > 6) {
        return dy;
      }

      return a.left.compareTo(
        b.left,
      );
    });

    final text = lines
        .map((e) => e.text)
        .where(
          (e) => e.trim().isNotEmpty,
        )
        .join('\n');

    return PersianOcrResult(
      text: text,
      lines: lines,
      elapsed:
          DateTime.now().difference(started),
    );
  }

  // ============================================================
  // DETECTION
  // ============================================================

  Future<List<_Box>> _detect(
    img.Image image,
  ) async {
    final session =
        _detSession!;

    final prep =
        _prepareDetector(image);

    final inputName =
        session.inputNames.first;

    final input =
        await OrtValue.fromList(
      prep.data,
      prep.shape,
    );

    final outputs =
        await session.run({
      inputName: input,
    });

    final output =
        outputs.values.first;

    final flat =
        (await output.asFlattenedList())
            .map(
              (e) => (e as num).toDouble(),
            )
            .toList(
              growable: false,
            );

    if (flat.isEmpty) {
      return const [];
    }

    final ratio =
        prep.mapWidth /
            prep.mapHeight;

    var outH =
        math.sqrt(
          flat.length / ratio,
        ).round();

    var outW =
        (outH * ratio).round();

    if (outW * outH !=
        flat.length) {
      outW = prep.mapWidth;

      outH = math.max(
        1,
        flat.length ~/
            math.max(
              1,
              outW,
            ),
      );
    }

    final boxes =
        _postProcessDb(
      flat,
      outW,
      outH,
      image.width,
      image.height,
    );

    return boxes;
  }

  // ============================================================
  // RECOGNIZE LONG LINE
  // ============================================================

  Future<_Recognized> _recognizeLongCrop(
    img.Image crop,
  ) async {
    /*
     * مدل:
     *
     * 48 x 320
     *
     * ابتدا بررسی می‌کنیم که آیا خط
     * بدون chunk شدن داخل مدل جا می‌شود یا نه.
     */

    const targetHeight = 48;
    const targetWidth = 320;

    final naturalWidth =
        crop.width *
            targetHeight /
            math.max(
              1,
              crop.height,
            );

    /*
     * اکثر خطوط معمولی از همین مسیر قبلی
     * عبور می‌کنند.
     *
     * این مهم است چون نمی‌خواهیم رفتار
     * OCR قبلی خراب شود.
     */
    if (naturalWidth <=
        targetWidth * 0.95) {
      return _recognizeCrop(
        crop,
      );
    }

    /*
     * اگر خط خیلی بلند باشد:
     *
     * عرضی از تصویر اصلی که بعد از resize
     * تقریباً به 320 می‌رسد.
     */
    final chunkWidth =
        math.max(
      32,
      (crop.height *
              targetWidth /
              targetHeight)
          .round(),
    );

    /*
     * کمی overlap.
     */
    final overlap =
        math.max(
      8,
      (chunkWidth *
              chunkOverlapRatio)
          .round(),
    );

    final step =
        math.max(
      16,
      chunkWidth - overlap,
    );

    final parts =
        <_RecognizedPart>[];

    var startX = 0;

    while (startX <
        crop.width) {
      final endX =
          math.min(
        crop.width,
        startX + chunkWidth,
      );

      final width =
          endX - startX;

      if (width < 8) {
        break;
      }

      final part =
          img.copyCrop(
        crop,
        x: startX,
        y: 0,
        width: width,
        height: crop.height,
      );

      final result =
          await _recognizeCrop(
        part,
      );

      if (result.text
          .trim()
          .isNotEmpty) {
        parts.add(
          _RecognizedPart(
            result.text.trim(),
            result.confidence,
            startX,
            endX,
          ),
        );
      }

      if (endX >=
          crop.width) {
        break;
      }

      startX += step;
    }

    if (parts.isEmpty) {
      /*
       * fallback:
       *
       * اگر chunking نتیجه نداد،
       * همان کل crop را یک بار امتحان می‌کنیم.
       *
       * این باعث می‌شود نسخه جدید
       * از نسخه قبلی ضعیف‌تر نشود.
       */
      return _recognizeCrop(
        crop,
      );
    }

    /*
     * اگر فقط یک قطعه داشتیم،
     * همان را برگردان.
     */
    if (parts.length == 1) {
      return _Recognized(
        parts.first.text,
        parts.first.confidence,
      );
    }

    /*
     * اتصال قطعات.
     */
    var combined =
        parts.first.text;

    var confidenceSum =
        parts.first.confidence;

    var confidenceCount = 1;

    for (var i = 1;
        i < parts.length;
        i++) {
      final current =
          parts[i];

      combined =
          _mergeChunkText(
        combined,
        current.text,
      );

      confidenceSum +=
          current.confidence;

      confidenceCount++;
    }

    return _Recognized(
      combined,
      confidenceCount == 0
          ? 0
          : confidenceSum /
              confidenceCount,
    );
  }

  // ============================================================
  // ORIGINAL RECOGNITION
  // ============================================================

  Future<_Recognized> _recognizeCrop(
    img.Image crop,
  ) async {
    final session =
        _recSession!;

    final prepared =
        _prepareRecognition(
      crop,
    );

    final inputName =
        session.inputNames.first;

    final input =
        await OrtValue.fromList(
      prepared.data,
      prepared.shape,
    );

    final outputs =
        await session.run({
      inputName: input,
    });

    final output =
        outputs.values.first;

    final flat =
        (await output.asFlattenedList())
            .map(
              (e) => (e as num).toDouble(),
            )
            .toList(
              growable: false,
            );

    final classes =
        _dictionary!;

    /*
     * دقیقاً همان decoder قبلی:
     *
     * 747 dictionary
     * + blank
     * + space
     *
     * = 749
     */
    const int classCount = 749;

    if (flat.length %
            classCount !=
        0) {
      throw StateError(
        'Invalid PP-OCRv5 Arabic output shape. '
        'flat=${flat.length}, '
        'classes=$classCount',
      );
    }

    final int timeSteps =
        flat.length ~/
            classCount;

    final best =
        <String>[];

    var confidenceSum =
        0.0;

    var confidenceCount =
        0;

    var previous = -1;

    for (var t = 0;
        t < timeSteps;
        t++) {
      var maxValue =
          -double.infinity;

      var maxIndex = 0;

      final int base =
          t * classCount;

      for (var c = 0;
          c < classCount;
          c++) {
        final value =
            flat[base + c];

        if (value >
            maxValue) {
          maxValue = value;
          maxIndex = c;
        }
      }

      /*
       * blank
       */
      if (maxIndex == 0) {
        previous = maxIndex;
        continue;
      }

      /*
       * CTC duplicate
       */
      if (maxIndex ==
          previous) {
        continue;
      }

      previous = maxIndex;

      /*
       * class 748 = space
       */
      if (maxIndex == 748) {
        best.add(' ');
        continue;
      }

      final int dictionaryIndex =
          maxIndex - 1;

      if (dictionaryIndex < 0 ||
          dictionaryIndex >=
              classes.length) {
        continue;
      }

      /*
       * همان confidence قبلی:
       * فقط برای میانگین confidence،
       * نه برای حذف کاراکتر.
       */
      final double probability =
          _softmaxLocal(
        flat,
        base,
        classCount,
        maxIndex,
      );

      best.add(
        classes[
            dictionaryIndex],
      );

      confidenceSum +=
          probability;

      confidenceCount++;
    }

    final raw =
        best.join();

    /*
     * مهم:
     *
     * این قسمت را دست نزده‌ایم.
     * چون گفتی نسخه قبلی در همین حالت
     * برای متن‌های بزرگ بهتر کار می‌کرد.
     */
    final logical =
        _reverseArabicVisualOrder(
      raw,
    );

    return _Recognized(
      PersianOcrNormalizer.normalize(
        logical,
      ),
      confidenceCount == 0
          ? 0
          : confidenceSum /
              confidenceCount,
    );
  }

  // ============================================================
  // MERGE CHUNKS
  // ============================================================

  String _mergeChunkText(
    String left,
    String right,
  ) {
    final a =
        left.trimRight();

    final b =
        right.trimLeft();

    if (a.isEmpty) {
      return b;
    }

    if (b.isEmpty) {
      return a;
    }

    /*
     * چون chunkها overlap دارند،
     * ممکن است چند حرف انتهای اول
     * در ابتدای دوم تکرار شوند.
     *
     * حداکثر 20 کاراکتر را بررسی می‌کنیم.
     */
    final maxOverlap =
        math.min(
      20,
      math.min(
        a.length,
        b.length,
      ),
    );

    for (var len =
            maxOverlap;
        len >= 2;
        len--) {
      final suffix =
          a.substring(
        a.length - len,
      );

      final prefix =
          b.substring(
        0,
        len,
      );

      if (_normalizeCompare(
            suffix,
          ) ==
          _normalizeCompare(
            prefix,
          )) {
        return a +
            b.substring(len);
      }
    }

    /*
     * اگر overlap قابل تشخیص نبود،
     * بدون حذف چیزی کنار هم می‌گذاریم.
     *
     * فاصله فقط اگر هیچ‌کدام
     * فاصله نداشته باشند.
     */
    if (a.endsWith(' ') ||
        b.startsWith(' ')) {
      return '$a$b';
    }

    return '$a $b';
  }

  String _normalizeCompare(
    String value,
  ) {
    return value
        .replaceAll(
          RegExp(r'\s+'),
          '',
        )
        .replaceAll(
          'ي',
          'ی',
        )
        .replaceAll(
          'ى',
          'ی',
        )
        .replaceAll(
          'ك',
          'ک',
        );
  }

  // ============================================================
  // RTL
  // ============================================================

  String _reverseArabicVisualOrder(
    String input,
  ) {
    if (input.isEmpty) {
      return input;
    }

    final parts =
        <String>[];

    final buffer =
        StringBuffer();

    bool isLatinNumeric(
      String c,
    ) {
      return RegExp(
        r'[a-zA-Z0-9 :*./%+\-]',
      ).hasMatch(c);
    }

    for (final rune
        in input.runes) {
      final c =
          String.fromCharCode(
        rune,
      );

      if (!isLatinNumeric(c)) {
        if (buffer.isNotEmpty) {
          parts.add(
            buffer.toString(),
          );

          buffer.clear();
        }

        parts.add(c);
      } else {
        buffer.write(c);
      }
    }

    if (buffer.isNotEmpty) {
      parts.add(
        buffer.toString(),
      );
    }

    return parts.reversed.join();
  }

  // ============================================================
  // SOFTMAX
  // ============================================================

  double _softmaxLocal(
    List<double> values,
    int base,
    int length,
    int target,
  ) {
    var maxValue =
        -double.infinity;

    for (var i = 0;
        i < length;
        i++) {
      maxValue =
          math.max(
        maxValue,
        values[base + i],
      );
    }

    var sum = 0.0;

    var targetExp =
        0.0;

    for (var i = 0;
        i < length;
        i++) {
      final e =
          math.exp(
        values[base + i] -
            maxValue,
      );

      sum += e;

      if (i == target) {
        targetExp = e;
      }
    }

    return sum == 0
        ? 0
        : targetExp / sum;
  }

  // ============================================================
  // DETECTOR PREPARE
  // ============================================================

  _DetectorInput _prepareDetector(
    img.Image image,
  ) {
    var h =
        image.height;

    var w =
        image.width;

    final ratio =
        math.min(
      960 /
          math.max(
            h,
            w,
          ),
      1.0,
    );

    w = math.max(
      32,
      ((w * ratio) / 32)
              .round() *
          32,
    );

    h = math.max(
      32,
      ((h * ratio) / 32)
              .round() *
          32,
    );

    final resized =
        img.copyResize(
      image,
      width: w,
      height: h,
      interpolation:
          img.Interpolation.linear,
    );

    final data =
        Float32List(
      1 * 3 * h * w,
    );

    var offsetR = 0;

    var offsetG =
        h * w;

    var offsetB =
        2 * h * w;

    for (var y = 0;
        y < h;
        y++) {
      for (var x = 0;
          x < w;
          x++) {
        final p =
            resized.getPixel(
          x,
          y,
        );

        data[offsetR++] =
            (p.r.toDouble() /
                        255.0 -
                    0.485) /
                0.229;

        data[offsetG++] =
            (p.g.toDouble() /
                        255.0 -
                    0.456) /
                0.224;

        data[offsetB++] =
            (p.b.toDouble() /
                        255.0 -
                    0.406) /
                0.225;
      }
    }

    return _DetectorInput(
      data,
      [1, 3, h, w],
      w,
      h,
    );
  }

  // ============================================================
  // RECOGNITION PREPARE
  // ============================================================

  _RecognitionInput _prepareRecognition(
    img.Image image,
  ) {
    const targetH = 48;
    const targetW = 320;

    final ratio =
        targetH /
            image.height;

    var width =
        (image.width *
                ratio)
            .round();

    width =
        math.max(
      1,
      math.min(
        targetW,
        width,
      ),
    );

    final resized =
        img.copyResize(
      image,
      width: width,
      height: targetH,
      interpolation:
          img.Interpolation.linear,
    );

    final data =
        Float32List(
      3 *
          targetH *
          targetW,
    );

    var r = 0;

    var g =
        targetH *
            targetW;

    var b =
        2 *
            targetH *
            targetW;

    for (var y = 0;
        y < targetH;
        y++) {
      for (var x = 0;
          x < targetW;
          x++) {
        final p =
            x < width
                ? resized.getPixel(
                    x,
                    y,
                  )
                : img.ColorRgb8(
                    255,
                    255,
                    255,
                  );

        data[r++] =
            (p.r.toDouble() /
                        255.0 -
                    0.5) /
                0.5;

        data[g++] =
            (p.g.toDouble() /
                        255.0 -
                    0.5) /
                0.5;

        data[b++] =
            (p.b.toDouble() /
                        255.0 -
                    0.5) /
                0.5;
      }
    }

    return _RecognitionInput(
      data,
      [1, 3, targetH, targetW],
    );
  }

  // ============================================================
  // CROP
  // ============================================================

  img.Image? _cropBox(
    img.Image image,
    _Box box,
  ) {
    final padX =
        ((box.right -
                    box.left) *
                0.03)
            .round();

    final padY =
        ((box.bottom -
                    box.top) *
                0.35)
            .round();

    final left =
        math.max(
      0,
      box.left - padX,
    );

    final top =
        math.max(
      0,
      box.top - padY,
    );

    final right =
        math.min(
      image.width,
      box.right + padX,
    );

    final bottom =
        math.min(
      image.height,
      box.bottom + padY,
    );

    if (right <= left ||
        bottom <= top) {
      return null;
    }

    return img.copyCrop(
      image,
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    );
  }

  // ============================================================
  // DB POST PROCESS
  // ============================================================

  List<_Box> _postProcessDb(
    List<double> map,
    int mapW,
    int mapH,
    int imageW,
    int imageH,
  ) {
    if (map.isEmpty) {
      return const [];
    }

    final sx =
        imageW / mapW;

    final sy =
        imageH / mapH;

    final visited =
        Uint8List(
      mapW * mapH,
    );

    final boxes =
        <_Box>[];

    for (var y = 0;
        y < mapH;
        y++) {
      for (var x = 0;
          x < mapW;
          x++) {
        final idx =
            y * mapW + x;

        if (visited[idx] != 0 ||
            map[idx] <
                detThreshold) {
          continue;
        }

        final queue =
            <int>[idx];

        visited[idx] = 1;

        var head = 0;

        var minX = x;
        var maxX = x;

        var minY = y;
        var maxY = y;

        var sum = 0.0;

        var count = 0;

        while (
            head <
                queue.length) {
          final q =
              queue[head++];

          final qx =
              q % mapW;

          final qy =
              q ~/ mapW;

          final score =
              map[q];

          sum += score;

          count++;

          minX =
              math.min(
            minX,
            qx,
          );

          maxX =
              math.max(
            maxX,
            qx,
          );

          minY =
              math.min(
            minY,
            qy,
          );

          maxY =
              math.max(
            maxY,
            qy,
          );

          for (final d in const [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
          ]) {
            final nx =
                qx + d[0];

            final ny =
                qy + d[1];

            if (nx < 0 ||
                ny < 0 ||
                nx >= mapW ||
                ny >= mapH) {
              continue;
            }

            final ni =
                ny * mapW + nx;

            if (visited[ni] !=
                    0 ||
                map[ni] <
                    detThreshold) {
              continue;
            }

            visited[ni] = 1;

            queue.add(ni);
          }
        }

        final bw =
            maxX - minX + 1;

        final bh =
            maxY - minY + 1;

        final area =
            bw * bh;

        if (area < 20 ||
            bw < 3 ||
            bh < 2) {
          continue;
        }

        final double mean =
            count == 0
                ? 0.0
                : sum / count;

        if (mean <
            boxThreshold) {
          continue;
        }

        boxes.add(
          _Box(
            (minX * sx)
                .round(),
            (minY * sy)
                .round(),
            ((maxX + 1) * sx)
                .round(),
            ((maxY + 1) * sy)
                .round(),
            mean,
          ),
        );
      }
    }

    boxes.sort(
      (a, b) {
        final rowA =
            a.top ~/
                math.max(
                  4,
                  (a.height *
                          0.5)
                      .round(),
                );

        final rowB =
            b.top ~/
                math.max(
                  4,
                  (b.height *
                          0.5)
                      .round(),
                );

        final row =
            rowA.compareTo(
          rowB,
        );

        if (row != 0) {
          return row;
        }

        return a.left.compareTo(
          b.left,
        );
      },
    );

    return _mergeNearby(
      boxes,
    );
  }

  // ============================================================
  // MERGE DETECTION BOXES
  // ============================================================

  List<_Box> _mergeNearby(
    List<_Box> boxes,
  ) {
    if (boxes.length < 2) {
      return boxes;
    }

    final result =
        <_Box>[];

    for (final box in boxes) {
      if (result.isEmpty) {
        result.add(box);
        continue;
      }

      final last =
          result.last;

      final overlapY =
          math.min(
                last.bottom,
                box.bottom,
              ) -
              math.max(
                last.top,
                box.top,
              );

      final minH =
          math.min(
        last.height,
        box.height,
      );

      final gap =
          box.left -
              last.right;

      if (overlapY >
              minH * 0.45 &&
          gap >= 0 &&
          gap <
              minH * 1.8) {
        result[
                result.length -
                    1] =
            _Box(
          math.min(
            last.left,
            box.left,
          ),
          math.min(
            last.top,
            box.top,
          ),
          math.max(
            last.right,
            box.right,
          ),
          math.max(
            last.bottom,
            box.bottom,
          ),
          (last.confidence +
                  box.confidence) /
              2,
        );
      } else {
        result.add(box);
      }
    }

    return result;
  }

  // ============================================================
  // DISPOSE
  // ============================================================

  Future<void> dispose() async {
    await _detSession?.close();
    await _recSession?.close();

    _detSession = null;
    _recSession = null;

    _initialized = false;
  }
}

// ==================================================================
// DATA CLASSES
// ==================================================================

class _DetectorInput {
  final Float32List data;
  final List<int> shape;
  final int mapWidth;
  final int mapHeight;

  _DetectorInput(
    this.data,
    this.shape,
    this.mapWidth,
    this.mapHeight,
  );
}

class _RecognitionInput {
  final Float32List data;
  final List<int> shape;

  _RecognitionInput(
    this.data,
    this.shape,
  );
}

class _Recognized {
  final String text;
  final double confidence;

  const _Recognized(
    this.text,
    this.confidence,
  );
}

/*
 * برای نگه داشتن محل chunk در صورت نیاز
 * و امکان توسعه بعدی.
 */
class _RecognizedPart {
  final String text;
  final double confidence;
  final int startX;
  final int endX;

  const _RecognizedPart(
    this.text,
    this.confidence,
    this.startX,
    this.endX,
  );
}

class _Box {
  final int left;
  final int top;
  final int right;
  final int bottom;
  final double confidence;

  const _Box(
    this.left,
    this.top,
    this.right,
    this.bottom,
    this.confidence,
  );

  int get width =>
      right - left;

  int get height =>
      bottom - top;
}
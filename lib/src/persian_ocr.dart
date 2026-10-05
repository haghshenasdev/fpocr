import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'persian_ocr_models.dart';
import 'persian_ocr_normalizer.dart';

/// Offline Persian/Arabic OCR based on PP-OCRv5 ONNX models.
///
/// Pipeline:
/// image -> DB detector -> line crops -> Arabic/Persian recognizer ->
/// Persian normalization.
///
/// No Python, OpenCV, Tesseract or internet is used at runtime.
class PersianOcr {
  static const String detAsset = 'assets/models/PP-OCRv5_mobile_det.onnx';

  static const String recAsset =
      'assets/models/arabic_PP-OCRv5_mobile_rec.onnx';

  static const String dictAsset = 'assets/models/ppocrv5_arabic_dict.txt';

  final OnnxRuntime _ort = OnnxRuntime();

  OrtSession? _detSession;
  OrtSession? _recSession;

  List<String>? _dictionary;

  bool _initialized = false;

  // Conservative values for Windows CPU.
  int maxSide = 1280;

  double detThreshold = 0.30;

  double boxThreshold = 0.55;

  int maxLines = 80;

  bool get isInitialized => _initialized;

  Future<void> initialize() async {
    if (_initialized) return;

    try {
      _detSession = await _ort.createSessionFromAsset(detAsset);

      _recSession = await _ort.createSessionFromAsset(recAsset);

      final raw = await rootBundle.loadString(dictAsset);

      final dictionary = raw
          .replaceAll('\r', '')
          .split('\n')
          .where((e) => e.isNotEmpty)
          .toList();

      /*
       * IMPORTANT:
       *
       * PP-OCRv5 Arabic dictionary contains 747 characters.
       *
       * The model output has:
       *
       *   0   = CTC blank
       *   1..747 = dictionary
       *   748 = space
       *
       * Therefore:
       *
       *   dictionary.length == 747
       *   output classes == 749
       *
       * Do NOT manually add " " to the dictionary.
       */
      if (dictionary.length != 747) {
        throw StateError(
          'Invalid PP-OCRv5 Arabic dictionary. '
          'Expected 747 entries, got ${dictionary.length}.',
        );
      }

      _dictionary = List<String>.unmodifiable(dictionary);

      _initialized = true;
    } catch (_) {
      await dispose();
      rethrow;
    }
  }

  Future<PersianOcrResult> recognizeBytes(Uint8List bytes) async {
    await initialize();

    final started = DateTime.now();

    final image = img.decodeImage(bytes);

    if (image == null) {
      throw const FormatException('Cannot decode input image.');
    }

    return _recognizeImage(image, started);
  }

  Future<PersianOcrResult> recognizeFile(String path) async {
    throw UnsupportedError(
      'Use recognizeBytes(await File(path).readAsBytes()) '
      'so this service stays platform-neutral.',
    );
  }

  Future<PersianOcrResult> _recognizeImage(
    img.Image source,
    DateTime started,
  ) async {
    final int sourceMax = math.max(source.width, source.height);

    final double scale = math.min(1.0, maxSide / sourceMax).toDouble();

    final img.Image image = scale < 0.999
        ? img.copyResize(
            source,
            width: math.max(1, (source.width * scale).round()),
            height: math.max(1, (source.height * scale).round()),
            interpolation: img.Interpolation.linear,
          )
        : source;

    final boxes = await _detect(image);

    final lines = <PersianOcrLine>[];

    for (final box in boxes.take(maxLines)) {
      final crop = _cropBox(image, box);

      if (crop == null || crop.width < 4 || crop.height < 4) {
        continue;
      }

      final recognized = await _recognizeCrop(crop);

      if (recognized.text.trim().isEmpty) {
        continue;
      }

      lines.add(
        PersianOcrLine(
          text: PersianOcrNormalizer.normalize(recognized.text),
          confidence: recognized.confidence,
          left: box.left,
          top: box.top,
          right: box.right,
          bottom: box.bottom,
        ),
      );
    }

    /*
     * The detector gives boxes from top to bottom.
     *
     * For now we keep the physical line order here.
     * RTL character order is handled by the recognition decoder.
     */
    lines.sort((a, b) {
      final int dy = a.top.compareTo(b.top);

      if (dy.abs() > 6) {
        return dy;
      }

      return a.left.compareTo(b.left);
    });

    final text = lines.map((e) => e.text).join('\n');

    return PersianOcrResult(
      text: text,
      lines: lines,
      elapsed: DateTime.now().difference(started),
    );
  }

  Future<List<_Box>> _detect(img.Image image) async {
    final session = _detSession!;

    final prep = _prepareDetector(image);

    final inputName = session.inputNames.first;

    final input = await OrtValue.fromList(prep.data, prep.shape);

    final outputs = await session.run({inputName: input});

    final output = outputs.values.first;

    final flat = (await output.asFlattenedList())
        .map((e) => (e as num).toDouble())
        .toList(growable: false);

    if (flat.isEmpty) {
      return const [];
    }

    /*
     * DB output is normally full resolution.
     *
     * Some exported models may return a downsampled map,
     * so infer its dimensions while preserving aspect ratio.
     */
    final double ratio = prep.mapWidth / prep.mapHeight;

    int outH = math.sqrt(flat.length / ratio).round();

    int outW = (outH * ratio).round();

    if (outW * outH != flat.length) {
      outW = prep.mapWidth;

      outH = math.max(1, flat.length ~/ math.max(1, outW));
    }

    return _postProcessDb(flat, outW, outH, image.width, image.height);
  }

  Future<_Recognized> _recognizeCrop(img.Image crop) async {
    final session = _recSession!;

    final prepared = _prepareRecognition(crop);

    final inputName = session.inputNames.first;

    final input = await OrtValue.fromList(prepared.data, prepared.shape);

    final outputs = await session.run({inputName: input});

    final output = outputs.values.first;

    final flat = (await output.asFlattenedList())
        .map((e) => (e as num).toDouble())
        .toList(growable: false);

    final classes = _dictionary!;

    /*
     * ============================================================
     * PP-OCRv5 ARABIC / PERSIAN CTC DECODER
     * ============================================================
     *
     * Official dictionary:
     *
     *   747 characters
     *
     * Model output:
     *
     *   749 classes
     *
     * Index:
     *
     *   0       -> CTC blank
     *   1..747  -> dictionary[0..746]
     *   748     -> space
     *
     * Therefore the output stride MUST be 749.
     */

    const int ctcBlankIndex = 0;

    final int dictionarySize = classes.length;

    if (dictionarySize != 747) {
      throw StateError(
        'Invalid Arabic/Persian dictionary size: '
        '$dictionarySize. Expected 747.',
      );
    }

    const int spaceIndex = 748;

    const int classCount = 749;

    if (flat.length % classCount != 0) {
      throw StateError(
        'Invalid PP-OCRv5 Arabic recognition output. '
        'flat=${flat.length}, '
        'expected multiple of $classCount.',
      );
    }

    final int timeSteps = flat.length ~/ classCount;

    final best = <String>[];

    double confidenceSum = 0.0;
    int confidenceCount = 0;

    int previous = -1;

    for (int t = 0; t < timeSteps; t++) {
      double maxValue = -double.infinity;

      int maxIndex = 0;

      final int base = t * classCount;

      for (int c = 0; c < classCount; c++) {
        final double value = flat[base + c];

        if (value > maxValue) {
          maxValue = value;
          maxIndex = c;
        }
      }

      /*
       * CTC blank.
       */
      if (maxIndex == ctcBlankIndex) {
        previous = maxIndex;
        continue;
      }

      /*
       * CTC repeated character.
       */
      if (maxIndex == previous) {
        continue;
      }

      previous = maxIndex;

      /*
       * Space is the last class.
       */
      if (maxIndex == spaceIndex) {
        best.add(' ');

        final double probability = _softmaxLocal(
          flat,
          base,
          classCount,
          maxIndex,
        );

        confidenceSum += probability;
        confidenceCount++;

        continue;
      }

      /*
       * Character class.
       *
       * Model:
       *
       *   class 1 -> dictionary[0]
       *   class 2 -> dictionary[1]
       *   ...
       *   class 747 -> dictionary[746]
       */
      final int dictionaryIndex = maxIndex - 1;

      if (dictionaryIndex < 0 || dictionaryIndex >= dictionarySize) {
        continue;
      }

      final double probability = _softmaxLocal(
        flat,
        base,
        classCount,
        maxIndex,
      );

      best.add(classes[dictionaryIndex]);

      confidenceSum += probability;
      confidenceCount++;
    }

    final raw = best.join();

    /*
     * Arabic/Persian PP-OCR models are
     * trained with visual-order labels.
     *
     * Convert visual output into logical
     * reading order.
     */
    final logical = _reverseArabicVisualOrder(raw);

    final normalized = PersianOcrNormalizer.normalize(logical);

    final double confidence = confidenceCount == 0
        ? 0.0
        : confidenceSum / confidenceCount;

    return _Recognized(normalized, confidence);
  }

  double _softmaxLocal(List<double> values, int base, int length, int target) {
    double maxValue = -double.infinity;

    for (int i = 0; i < length; i++) {
      maxValue = math.max(maxValue, values[base + i]).toDouble();
    }

    double sum = 0.0;
    double targetExp = 0.0;

    for (int i = 0; i < length; i++) {
      final double e = math.exp(values[base + i] - maxValue);

      sum += e;

      if (i == target) {
        targetExp = e;
      }
    }

    if (sum == 0.0) {
      return 0.0;
    }

    return targetExp / sum;
  }

  String _reverseArabicVisualOrder(String input) {
    if (input.isEmpty) {
      return input;
    }

    /*
     * IMPORTANT:
     *
     * Do not simply reverse the entire Unicode
     * string. Dates, numbers and Latin words
     * must keep their internal order.
     *
     * Example:
     *
     *   1405/07/13
     *
     * must remain:
     *
     *   1405/07/13
     */

    final parts = <String>[];

    final buffer = StringBuffer();

    bool isLatinNumeric(String c) {
      return RegExp(r'[a-zA-Z0-9 :*./%+\-]').hasMatch(c);
    }

    for (final rune in input.runes) {
      final c = String.fromCharCode(rune);

      if (!isLatinNumeric(c)) {
        if (buffer.isNotEmpty) {
          parts.add(buffer.toString());
          buffer.clear();
        }

        parts.add(c);
      } else {
        buffer.write(c);
      }
    }

    if (buffer.isNotEmpty) {
      parts.add(buffer.toString());
    }

    /*
     * If the recognition model produced
     * visual RTL order, reverse logical
     * chunks.
     */
    return parts.reversed.join();
  }

  _DetectorInput _prepareDetector(img.Image image) {
    int h = image.height;
    int w = image.width;

    final double ratio = math.min(960.0 / math.max(h, w), 1.0).toDouble();

    w = math.max(32, ((w * ratio) / 32).round() * 32);

    h = math.max(32, ((h * ratio) / 32).round() * 32);

    final resized = img.copyResize(
      image,
      width: w,
      height: h,
      interpolation: img.Interpolation.linear,
    );

    final data = Float32List(1 * 3 * h * w);

    int offsetR = 0;

    int offsetG = h * w;

    int offsetB = 2 * h * w;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = resized.getPixel(x, y);

        data[offsetR++] = (p.r.toDouble() / 255.0 - 0.485) / 0.229;

        data[offsetG++] = (p.g.toDouble() / 255.0 - 0.456) / 0.224;

        data[offsetB++] = (p.b.toDouble() / 255.0 - 0.406) / 0.225;
      }
    }

    return _DetectorInput(data, [1, 3, h, w], w, h);
  }

  _RecognitionInput _prepareRecognition(img.Image image) {
    /*
     * PP-OCRv5 Arabic recognition:
     *
     * [3, 48, 320]
     */
    const int targetH = 48;
    const int targetW = 320;

    final double ratio = targetH / image.height;

    int width = (image.width * ratio).round();

    width = math.max(1, math.min(targetW, width));

    final resized = img.copyResize(
      image,
      width: width,
      height: targetH,
      interpolation: img.Interpolation.linear,
    );

    final data = Float32List(3 * targetH * targetW);

    int r = 0;

    int g = targetH * targetW;

    int b = 2 * targetH * targetW;

    for (int y = 0; y < targetH; y++) {
      for (int x = 0; x < targetW; x++) {
        final p = x < width
            ? resized.getPixel(x, y)
            : img.ColorRgb8(255, 255, 255);

        data[r++] = (p.r.toDouble() / 255.0 - 0.5) / 0.5;

        data[g++] = (p.g.toDouble() / 255.0 - 0.5) / 0.5;

        data[b++] = (p.b.toDouble() / 255.0 - 0.5) / 0.5;
      }
    }

    return _RecognitionInput(data, [1, 3, targetH, targetW]);
  }

  img.Image? _cropBox(img.Image image, _Box box) {
    final int padX = ((box.right - box.left) * 0.03).round();

    final int padY = ((box.bottom - box.top) * 0.35).round();

    final int left = math.max(0, box.left - padX);

    final int top = math.max(0, box.top - padY);

    final int right = math.min(image.width, box.right + padX);

    final int bottom = math.min(image.height, box.bottom + padY);

    if (right <= left || bottom <= top) {
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

    final double sx = imageW / mapW;

    final double sy = imageH / mapH;

    final visited = Uint8List(mapW * mapH);

    final boxes = <_Box>[];

    /*
     * Lightweight connected components
     * over DB probability map.
     */
    for (int y = 0; y < mapH; y++) {
      for (int x = 0; x < mapW; x++) {
        final int idx = y * mapW + x;

        if (visited[idx] != 0 || map[idx] < detThreshold) {
          continue;
        }

        final queue = <int>[idx];

        visited[idx] = 1;

        int head = 0;

        int minX = x;
        int maxX = x;

        int minY = y;
        int maxY = y;

        double sum = 0.0;

        int count = 0;

        while (head < queue.length) {
          final int q = queue[head++];

          final int qx = q % mapW;

          final int qy = q ~/ mapW;

          final double score = map[q];

          sum += score;
          count++;

          minX = math.min(minX, qx);

          maxX = math.max(maxX, qx);

          minY = math.min(minY, qy);

          maxY = math.max(maxY, qy);

          const directions = [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
          ];

          for (final d in directions) {
            final int nx = qx + d[0];

            final int ny = qy + d[1];

            if (nx < 0 || ny < 0 || nx >= mapW || ny >= mapH) {
              continue;
            }

            final int ni = ny * mapW + nx;

            if (visited[ni] != 0 || map[ni] < detThreshold) {
              continue;
            }

            visited[ni] = 1;

            queue.add(ni);
          }
        }

        final int bw = maxX - minX + 1;

        final int bh = maxY - minY + 1;

        final int area = bw * bh;

        if (area < 20 || bw < 3 || bh < 2) {
          continue;
        }

        final double mean = count == 0 ? 0.0 : sum / count;

        if (mean < boxThreshold) {
          continue;
        }

        boxes.add(
          _Box(
            (minX * sx).round(),
            (minY * sy).round(),
            ((maxX + 1) * sx).round(),
            ((maxY + 1) * sy).round(),
            mean,
          ),
        );
      }
    }

    boxes.sort((a, b) {
      final int rowA = a.top ~/ math.max(4, (a.height * 0.5).round());

      final int rowB = b.top ~/ math.max(4, (b.height * 0.5).round());

      final int row = rowA.compareTo(rowB);

      if (row != 0) {
        return row;
      }

      return a.left.compareTo(b.left);
    });

    return _mergeNearby(boxes);
  }

  List<_Box> _mergeNearby(List<_Box> boxes) {
    if (boxes.length < 2) {
      return boxes;
    }

    final result = <_Box>[];

    for (final box in boxes) {
      if (result.isEmpty) {
        result.add(box);
        continue;
      }

      final last = result.last;

      final int overlapY =
          math.min(last.bottom, box.bottom) - math.max(last.top, box.top);

      final int minH = math.min(last.height, box.height);

      final int gap = box.left - last.right;

      if (overlapY > minH * 0.45 && gap >= 0 && gap < minH * 1.8) {
        result[result.length - 1] = _Box(
          math.min(last.left, box.left),
          math.min(last.top, box.top),
          math.max(last.right, box.right),
          math.max(last.bottom, box.bottom),
          (last.confidence + box.confidence) / 2.0,
        );
      } else {
        result.add(box);
      }
    }

    return result;
  }

  Future<void> dispose() async {
    await _detSession?.close();
    await _recSession?.close();

    _detSession = null;
    _recSession = null;

    _dictionary = null;

    _initialized = false;
  }
}

class _DetectorInput {
  final Float32List data;
  final List<int> shape;
  final int mapWidth;
  final int mapHeight;

  _DetectorInput(this.data, this.shape, this.mapWidth, this.mapHeight);
}

class _RecognitionInput {
  final Float32List data;
  final List<int> shape;

  _RecognitionInput(this.data, this.shape);
}

class _Recognized {
  final String text;
  final double confidence;

  _Recognized(this.text, this.confidence);
}

class _Box {
  final int left;
  final int top;
  final int right;
  final int bottom;
  final double confidence;

  const _Box(this.left, this.top, this.right, this.bottom, this.confidence);

  int get width => right - left;

  int get height => bottom - top;
}

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:fpocr/models/model_info.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'model_manager.dart';

class OcrLine {
  final String text;
  final double confidence;
  final ui.Rect box;
  const OcrLine({required this.text, required this.confidence, required this.box});
}

class OcrResult {
  final String text;
  final List<OcrLine> lines;
  final Duration elapsed;
  const OcrResult({required this.text, required this.lines, required this.elapsed});
}

class PersianOcrEngine {
  final ModelManager models;
  OrtSession? _det;
  OrtSession? _rec;
  OrtSession? _cls;
  List<String> _dict = const [];
  bool _ready = false;

  PersianOcrEngine(this.models);

  Future<void> initialize() async {
    if (_ready) return;
    final detFile = await models.fileFor(OcrModelCatalog.det);
    final recFile = await models.fileFor(OcrModelCatalog.rec);
    final clsFile = await models.fileFor(OcrModelCatalog.cls);
    if (!await models.isInstalled(OcrModelCatalog.det) ||
        !await models.isInstalled(OcrModelCatalog.rec) ||
        !await models.isInstalled(OcrModelCatalog.dict)) {
      throw StateError('مدل‌های OCR نصب نشده‌اند.');
    }
    OrtEnv.instance.init();
    final options = OrtSessionOptions();
    options.appendCPUProvider(CPUFlags.useArena);
    _det = OrtSession.fromBuffer(await detFile.readAsBytes(), options);
    _rec = OrtSession.fromBuffer(await recFile.readAsBytes(), options);
    if (await clsFile.exists() && await models.isInstalled(OcrModelCatalog.cls)) {
      _cls = OrtSession.fromBuffer(await clsFile.readAsBytes(), options);
    }
    _dict = const ['blank'];
    final dictText = await models.readDictionary();
    _dict = ['blank', ...dictText.split(RegExp(r'\r?\n')).where((e) => e.isNotEmpty)];
    _ready = true;
  }

  Future<OcrResult> recognize(Uint8List bytes, {bool accurate = true}) async {
    if (!_ready) await initialize();
    final started = DateTime.now();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw const FormatException('فرمت تصویر قابل خواندن نیست.');
    final source = img.bakeOrientation(decoded);
    final detInput = _prepareDetection(source, accurate: accurate);
    final detOutput = await _run(_det!, detInput.data, detInput.shape);
    final scores = _flatten(detOutput);
    final boxes = _boxesFromMask(scores, detInput.width, detInput.height, source.width, source.height);

    final recognized = <OcrLine>[];
    for (final box in boxes) {
      final crop = _cropSafe(source, box.left, box.top, box.right, box.bottom);
      if (crop.width < 3 || crop.height < 3) continue;
      final oriented = await _maybeRotate(crop);
      final rec = _prepareRecognition(oriented);
      final out = await _run(_rec!, rec.data, rec.shape);
      final decodedLine = _decodeCtc(_flatten(out));
      if (decodedLine.text.trim().isEmpty) continue;
      recognized.add(OcrLine(
        text: _normalizePersian(decodedLine.text),
        confidence: decodedLine.confidence,
        box: ui.Rect.fromLTRB(box.left, box.top, box.right, box.bottom),
      ));
    }

    recognized.sort((a, b) {
      final ay = a.box.top, by = b.box.top;
      final tolerance = math.max(8.0, math.min(a.box.height, b.box.height) * .55);
      if ((ay - by).abs() > tolerance) return ay.compareTo(by);
      return b.box.left.compareTo(a.box.left); // Persian/Arabic: right to left.
    });
    final text = recognized.map((e) => e.text.trim()).where((e) => e.isNotEmpty).join('\n');
    return OcrResult(text: text, lines: recognized, elapsed: DateTime.now().difference(started));
  }

  Future<List<double>> _run(OrtSession session, List<double> data, List<int> shape) async {
    final inputName = session.inputNames.first;
    final tensor = OrtValueTensor.createTensorWithDataList(Float32List.fromList(data), shape);
    final options = OrtRunOptions();
    try {
      final outputs = await session.runAsync(options, {inputName: tensor});
      final first = outputs?.first;
      if (first == null) throw StateError('مدل خروجی خالی برگرداند.');
      final value = first.value;
      final flat = _flattenDynamic(value);
      for (final o in outputs!) { o?.release(); }
      return flat;
    } finally {
      tensor.release();
      options.release();
    }
  }

  List<double> _flattenDynamic(dynamic value) {
    final result = <double>[];
    void visit(dynamic x) {
      if (x is num) {
        result.add(x.toDouble());
      } else if (x is Iterable) {
        for (final v in x) visit(v);
      } else if (x is Float32List) {
        result.addAll(x.map((e) => e.toDouble()));
      }
    }
    visit(value);
    return result;
  }

  List<double> _flatten(dynamic value) => _flattenDynamic(value);

  _TensorData _prepareDetection(img.Image source, {required bool accurate}) {
    final maxSide = accurate ? 1280 : 960;
    final scale = math.min(1.0, maxSide / math.max(source.width, source.height));
    var w = math.max(32, (source.width * scale).round());
    var h = math.max(32, (source.height * scale).round());
    w = ((w + 31) ~/ 32) * 32;
    h = ((h + 31) ~/ 32) * 32;
    final resized = img.copyResize(source, width: w, height: h, interpolation: img.Interpolation.linear);
    final data = List<double>.filled(3 * w * h, 0);
    var p = 0;
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          final px = resized.getPixel(x, y);
          final v = c == 0 ? px.r : c == 1 ? px.g : px.b;
          final mean = const [0.485, 0.456, 0.406][c];
          final std = const [0.229, 0.224, 0.225][c];
          data[p++] = (v / 255.0 - mean) / std;
        }
      }
    }
    return _TensorData(data, [1, 3, h, w], w, h);
  }

  Future<img.Image> _maybeRotate(img.Image crop) async {
    // The orientation model is intentionally optional.  Recognition is kept
    // stable on devices where the classifier model is unavailable.
    if (_cls == null) return crop;
    return crop;
  }

  _TensorData _prepareRecognition(img.Image source) {
    const height = 48;
    final ratio = source.width / math.max(1, source.height);
    var width = math.max(48, (height * ratio).round());
    width = math.min(320, ((width + 7) ~/ 8) * 8);
    final resized = img.copyResize(source, width: width, height: height, interpolation: img.Interpolation.linear);
    final data = List<double>.filled(3 * width * height, 0);
    var p = 0;
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
          final px = resized.getPixel(x, y);
          final v = c == 0 ? px.r : c == 1 ? px.g : px.b;
          data[p++] = (v / 255.0 - .5) / .5;
        }
      }
    }
    return _TensorData(data, [1, 3, height, width], width, height);
  }

  List<_Box> _boxesFromMask(List<double> scores, int inputW, int inputH, int originalW, int originalH) {
    if (scores.isEmpty) return [];
    final outW = math.max(1, inputW ~/ 4);
    final outH = math.max(1, inputH ~/ 4);
    if (scores.length < outW * outH) return [];
    final mask = Uint8List(outW * outH);
    for (var i = 0; i < mask.length; i++) {
      mask[i] = scores[i] > .30 ? 1 : 0;
    }
    final seen = Uint8List(mask.length);
    final boxes = <_Box>[];
    final sx = originalW / outW;
    final sy = originalH / outH;
    final q = <int>[];
    for (var start = 0; start < mask.length; start++) {
      if (mask[start] == 0 || seen[start] == 1) continue;
      q.clear(); q.add(start); seen[start] = 1;
      var minX = start % outW, maxX = minX, minY = start ~/ outW, maxY = minY, count = 0;
      for (var qi = 0; qi < q.length; qi++) {
        final idx = q[qi]; final x = idx % outW; final y = idx ~/ outW; count++;
        minX = math.min(minX, x); maxX = math.max(maxX, x); minY = math.min(minY, y); maxY = math.max(maxY, y);
        for (final n in <int>[idx - 1, idx + 1, idx - outW, idx + outW]) {
          if (n < 0 || n >= mask.length) continue;
          final nx = n % outW, ny = n ~/ outW;
          if ((nx - x).abs() + (ny - y).abs() != 1) continue;
          if (mask[n] == 1 && seen[n] == 0) { seen[n] = 1; q.add(n); }
        }
      }
      final bw = maxX - minX + 1, bh = maxY - minY + 1;
      if (count < 8 || bw < 2 || bh < 2 || bw / bh < .12) continue;
      final padX = math.max(2, (bw * .08).round());
      final padY = math.max(2, (bh * .35).round());
      boxes.add(_Box(
        math.max(0, (minX - padX) * sx),
        math.max(0, (minY - padY) * sy),
        math.min(originalW.toDouble(), (maxX + 1 + padX) * sx),
        math.min(originalH.toDouble(), (maxY + 1 + padY) * sy),
      ));
    }
    boxes.sort((a,b) => (a.top - b.top).abs() < 20 ? b.left.compareTo(a.left) : a.top.compareTo(b.top));
    return _mergeNearby(boxes);
  }

  List<_Box> _mergeNearby(List<_Box> boxes) {
    if (boxes.length < 2) return boxes;
    final out = <_Box>[];
    for (final b in boxes) {
      var merged = false;
      for (var i = 0; i < out.length; i++) {
        final a = out[i];
        final yOverlap = math.max(0, math.min(a.bottom,b.bottom)-math.max(a.top,b.top));
        final minH = math.min(a.height,b.height);
        final close = (a.left-b.right).abs() < math.max(20, minH*.8) || (b.left-a.right).abs() < math.max(20, minH*.8);
        if (yOverlap > minH*.55 && close) {
          out[i] = _Box(math.min(a.left,b.left), math.min(a.top,b.top), math.max(a.right,b.right), math.max(a.bottom,b.bottom));
          merged = true; break;
        }
      }
      if (!merged) out.add(b);
    }
    return out;
  }

  img.Image _cropSafe(img.Image src, double l, double t, double r, double b) {
    final x = l.round().clamp(0, src.width-1);
    final y = t.round().clamp(0, src.height-1);
    final right = r.round().clamp(x+1, src.width);
    final bottom = b.round().clamp(y+1, src.height);
    return img.copyCrop(src, x:x, y:y, width:right-x, height:bottom-y);
  }

  _Decoded _decodeCtc(List<double> data) {
    if (data.isEmpty || _dict.isEmpty) return const _Decoded('', 0);
    // PP-OCR CTC output is normally [time, class] or [1,time,class].
    final classes = _dict.length;
    final time = data.length ~/ classes;
    var last = -1;
    var total = 0.0;
    var count = 0;
    final chars = <String>[];
    for (var t = 0; t < time; t++) {
      var best = 0, bestScore = -double.infinity;
      final base = t * classes;
      for (var c = 0; c < classes; c++) {
        final s = data[base+c];
        if (s > bestScore) { bestScore=s; best=c; }
      }
      if (best != 0 && best != last && best < _dict.length) {
        chars.add(_dict[best]); total += bestScore; count++;
      }
      last = best;
    }
    return _Decoded(chars.join(), count == 0 ? 0 : total/count);
  }

  String _normalizePersian(String s) => s.replaceAll('ي','ی').replaceAll('ى','ی').replaceAll('ك','ک').replaceAll('ۀ','هٔ').replaceAll('\u200c\u200c','\u200c').trim();

  Future<void> dispose() async {
    _det?.release(); _rec?.release(); _cls?.release();
    _det = null; _rec = null; _cls = null; _ready = false;
    OrtEnv.instance.release();
  }
}

class _TensorData { final List<double> data; final List<int> shape; final int width; final int height; _TensorData(this.data,this.shape,this.width,this.height); }
class _Box { final double left,top,right,bottom; const _Box(this.left,this.top,this.right,this.bottom); double get width=>right-left; double get height=>bottom-top; }
class _Decoded { final String text; final double confidence; const _Decoded(this.text,this.confidence); }

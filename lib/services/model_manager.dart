import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import '../models/model_info.dart';

class ModelStatus {
  final ModelInfo model;
  final bool installed;
  final int bytes;
  const ModelStatus({required this.model, required this.installed, required this.bytes});
}

class ModelManager {
  Directory? _dir;

  Future<Directory> get directory async {
    _dir ??= Directory('${(await getApplicationSupportDirectory()).path}/ocr_models/ppocrv5');
    if (!await _dir!.exists()) await _dir!.create(recursive: true);
    return _dir!;
  }

  Future<File> fileFor(ModelInfo model) async => File('${(await directory).path}/${model.fileName}');

  Future<bool> isInstalled(ModelInfo model) async {
    final f = await fileFor(model);
    if (!await f.exists() || await f.length() == 0) return false;
    if (model.sha256 == null) return true;
    return await _sha256(f) == model.sha256;
  }

  Future<List<ModelStatus>> status() async => [
    for (final m in OcrModelCatalog.all)
      ModelStatus(model: m, installed: await isInstalled(m), bytes: await _existingSize(m)),
  ];

  Future<int> _existingSize(ModelInfo model) async {
    final f = await fileFor(model);
    return await f.exists() ? f.length() : 0;
  }

  Future<String> _sha256(File file) async {
    final digest = await sha256.bind(file.openRead()).first;
    return digest.toString();
  }

  Future<File> ensure(ModelInfo model, {void Function(int received, int? total)? onProgress}) async {
    final target = await fileFor(model);
    if (await isInstalled(model)) return target;
    final temp = File('${target.path}.download');
    if (await temp.exists()) await temp.delete();
    final client = http.Client();
    try {
      final request = http.Request('GET', Uri.parse(model.url));
      request.headers['User-Agent'] = 'PersianOCR-Flutter/1.7';
      final response = await client.send(request);
      if (response.statusCode != 200) {
        throw HttpException('HTTP ${response.statusCode} while downloading ${model.fileName}');
      }
      final sink = temp.openWrite();
      var received = 0;
      await for (final chunk in response.stream) {
        received += chunk.length;
        sink.add(chunk);
        onProgress?.call(received, response.contentLength);
      }
      await sink.flush();
      await sink.close();
      if (model.sha256 != null) {
        final got = await _sha256(temp);
        if (got.toLowerCase() != model.sha256!.toLowerCase()) {
          await temp.delete();
          throw const FormatException('SHA-256 مدل با مقدار رسمی مطابقت ندارد.');
        }
      }
      if (await target.exists()) await target.delete();
      await temp.rename(target.path);
      return target;
    } finally {
      client.close();
    }
  }

  Future<void> ensureAll({void Function(ModelInfo model, int received, int? total)? onProgress}) async {
    for (final m in OcrModelCatalog.all) {
      await ensure(m, onProgress: (r, t) => onProgress?.call(m, r, t));
    }
  }

  Future<String> readDictionary() async {
    final f = await fileFor(OcrModelCatalog.dict);
    return const Utf8Decoder(allowMalformed: true).convert(await f.readAsBytes());
  }
}

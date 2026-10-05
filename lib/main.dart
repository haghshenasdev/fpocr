
import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import 'persian_ocr.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const PersianOcrApp());
}

class PersianOcrApp extends StatelessWidget {
  const PersianOcrApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'OCR فارسی آفلاین',
      theme: ThemeData(
        useMaterial3: true,
        fontFamily: 'Segoe UI',
        colorSchemeSeed: Colors.indigo,
      ),
      home: const PersianOcrPage(),
    );
  }
}

class PersianOcrPage extends StatefulWidget {
  const PersianOcrPage({super.key});

  @override
  State<PersianOcrPage> createState() => _PersianOcrPageState();
}

class _PersianOcrPageState extends State<PersianOcrPage> {
  final PersianOcr _ocr = PersianOcr();
  final TextEditingController _textController = TextEditingController();

  Uint8List? _imageBytes;
  String? _fileName;
  String _status = 'آماده';
  String _elapsed = '';
  bool _busy = false;
  bool _initialized = false;

  @override
  void dispose() {
    _textController.dispose();
    _ocr.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    if (_initialized) return;

    setState(() {
      _busy = true;
      _status = 'در حال بارگذاری مدل‌های ONNX...';
    });

    try {
      await _ocr.initialize();

      if (!mounted) return;
      setState(() {
        _initialized = true;
        _busy = false;
        _status = 'مدل OCR آماده است';
      });
    } catch (e, st) {
      debugPrint('OCR initialize error: $e');
      debugPrintStack(stackTrace: st);

      if (!mounted) return;
      setState(() {
        _busy = false;
        _status = 'خطا در بارگذاری مدل: $e';
      });
    }
  }

  Future<void> _pickImage() async {
    if (_busy) return;

    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const [
          'jpg',
          'jpeg',
          'png',
          'bmp',
          'webp',
        ],
        withData: true,
      );

      if (result == null || result.files.isEmpty) return;

      final file = result.files.single;

      Uint8List? bytes = file.bytes;

      // On Windows, depending on the FilePicker version/configuration,
      // bytes may be null. Fall back to reading the selected path.
      if (bytes == null && file.path != null) {
        bytes = await File(file.path!).readAsBytes();
      }

      if (bytes == null || bytes.isEmpty) {
        throw StateError('فایل تصویر قابل خواندن نیست.');
      }

      if (!mounted) return;

      setState(() {
        _imageBytes = bytes;
        _fileName = file.name;
        _textController.clear();
        _elapsed = '';
        _status = 'تصویر انتخاب شد';
      });
    } catch (e, st) {
      debugPrint('Pick image error: $e');
      debugPrintStack(stackTrace: st);

      if (!mounted) return;
      setState(() {
        _status = 'خطا در انتخاب تصویر: $e';
      });
    }
  }

  Future<void> _runOcr() async {
    final bytes = _imageBytes;
    if (bytes == null || bytes.isEmpty || _busy) return;

    setState(() {
      _busy = true;
      _status = 'در حال تشخیص متن...';
      _elapsed = '';
    });

    try {
      if (!_initialized) {
        await _ocr.initialize();
        _initialized = true;
      }

      final result = await _ocr.recognizeBytes(bytes);

      if (!mounted) return;

      setState(() {
        _textController.text = result.text;
        _elapsed =
            '${result.elapsed.inMilliseconds} ms - ${result.lines.length} خط';
        _status = result.text.trim().isEmpty
            ? 'متنی پیدا نشد'
            : 'OCR با موفقیت انجام شد';
        _busy = false;
      });
    } catch (e, st) {
      debugPrint('OCR error: $e');
      debugPrintStack(stackTrace: st);

      if (!mounted) return;

      setState(() {
        _busy = false;
        _status = 'خطای OCR: $e';
      });
    }
  }

  void _clear() {
    if (_busy) return;

    setState(() {
      _imageBytes = null;
      _fileName = null;
      _textController.clear();
      _elapsed = '';
      _status = 'آماده';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('OCR فارسی آفلاین - ONNX'),
          actions: [
            IconButton(
              tooltip: 'پاک کردن',
              onPressed: _busy ? null : _clear,
              icon: const Icon(Icons.clear_all),
            ),
          ],
        ),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              _buildToolbar(),
              const SizedBox(height: 12),
              _buildStatus(),
              const SizedBox(height: 12),
              Expanded(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      flex: 5,
                      child: _buildImagePanel(),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      flex: 5,
                      child: _buildTextPanel(),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildToolbar() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Wrap(
          spacing: 10,
          runSpacing: 10,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            FilledButton.icon(
              onPressed: _busy ? null : _initialize,
              icon: const Icon(Icons.memory),
              label: const Text('بارگذاری مدل'),
            ),
            FilledButton.icon(
              onPressed: _busy ? null : _pickImage,
              icon: const Icon(Icons.image_outlined),
              label: const Text('انتخاب تصویر'),
            ),
            FilledButton.icon(
              onPressed:
                  (_busy || _imageBytes == null) ? null : _runOcr,
              icon: _busy
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.document_scanner_outlined),
              label: const Text('اجرای OCR'),
            ),
            if (_fileName != null)
              Chip(
                avatar: const Icon(Icons.image, size: 18),
                label: Text(
                  _fileName!,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatus() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(
          horizontal: 14,
          vertical: 10,
        ),
        child: Row(
          children: [
            Icon(
              _busy
                  ? Icons.sync
                  : _status.startsWith('خطا')
                      ? Icons.error_outline
                      : Icons.info_outline,
            ),
            const SizedBox(width: 8),
            Expanded(child: Text(_status)),
            if (_elapsed.isNotEmpty)
              Text(
                _elapsed,
                style: Theme.of(context).textTheme.bodySmall,
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildImagePanel() {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: _imageBytes == null
          ? const Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.image_outlined, size: 64),
                  SizedBox(height: 12),
                  Text('برای شروع یک تصویر انتخاب کنید'),
                ],
              ),
            )
          : InteractiveViewer(
              minScale: 0.2,
              maxScale: 5,
              child: Center(
                child: Image.memory(
                  _imageBytes!,
                  fit: BoxFit.contain,
                  filterQuality: FilterQuality.medium,
                ),
              ),
            ),
    );
  }

  Widget _buildTextPanel() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                const Icon(Icons.text_fields),
                const SizedBox(width: 8),
                const Text(
                  'متن تشخیص داده شده',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                IconButton(
                  tooltip: 'کپی متن',
                  onPressed: _textController.text.isEmpty
                      ? null
                      : () async {
                          // Clipboard is intentionally kept out of the
                          // OCR engine; this button can be connected to
                          // your application's clipboard helper.
                        },
                  icon: const Icon(Icons.copy),
                ),
              ],
            ),
            const Divider(),
            Expanded(
              child: TextField(
                controller: _textController,
                expands: true,
                maxLines: null,
                minLines: null,
                textDirection: TextDirection.rtl,
                textAlign: TextAlign.right,
                decoration: const InputDecoration(
                  border: InputBorder.none,
                  hintText: 'خروجی OCR اینجا نمایش داده می‌شود...',
                  alignLabelWithHint: true,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

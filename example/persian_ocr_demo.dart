import 'dart:io';

import 'package:flutter/material.dart';
import '../lib/persian_ocr.dart';

void main() => runApp(const OcrDemo());

class OcrDemo extends StatefulWidget {
  const OcrDemo({super.key});

  @override
  State<OcrDemo> createState() => _OcrDemoState();
}

class _OcrDemoState extends State<OcrDemo> {
  final _ocr = PersianOcr();
  String _status = 'آماده';
  String _text = '';

  Future<void> run(String path) async {
    setState(() => _status = 'در حال OCR...');
    try {
      final result = await _ocr.recognizeBytes(await File(path).readAsBytes());
      if (!mounted) return;
      setState(() {
        _status = 'زمان: ${result.elapsed.inMilliseconds} ms';
        _text = result.text;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _status = 'خطا: $e');
    }
  }

  @override
  void dispose() {
    _ocr.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Persian ONNX OCR')),
        body: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(_status),
              const SizedBox(height: 16),
              SelectableText(_text, textDirection: TextDirection.rtl),
            ],
          ),
        ),
      ),
    );
  }
}

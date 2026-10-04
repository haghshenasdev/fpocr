import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pdf_render/pdf_render.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:image/image.dart' as img;
import 'models/model_info.dart';
import 'services/model_manager.dart';
import 'services/ocr_engine.dart';
import 'services/export_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const PersianOcrApp());
}

class PersianOcrApp extends StatelessWidget {
  const PersianOcrApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'OCR فارسی',
      locale: const Locale('fa'),
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo, brightness: Brightness.dark),
      home: const OcrHomePage(),
    );
  }
}

class OcrHomePage extends StatefulWidget {
  const OcrHomePage({super.key});
  @override
  State<OcrHomePage> createState() => _OcrHomePageState();
}

class _OcrHomePageState extends State<OcrHomePage> {
  final _models = ModelManager();
  late final _engine = PersianOcrEngine(_models);
  final _picker = ImagePicker();
  final _result = TextEditingController();
  final _logs = <String>[];
  Uint8List? _imageBytes;
  String? _fileName;
  bool _busy = false;
  bool _accurate = true;
  double _downloadProgress = 0;
  String _status = 'یک تصویر یا PDF انتخاب کنید.';
  String _export = 'txt';
  int _pdfPageCount = 0;
  File? _selectedFile;
  SharedPreferences? _prefs;

  @override
  void initState() {
    super.initState();
    _loadPrefs();
  }

  Future<void> _loadPrefs() async {
    _prefs = await SharedPreferences.getInstance();
    setState(() {
      _accurate = _prefs?.getBool('accurate') ?? true;
      _export = _prefs?.getString('export') ?? 'txt';
    });
    final states = await _models.status();
    if (states.any((s) => !s.installed)) {
      setState(() => _status = 'مدل‌های OCR آماده نیستند؛ از «مدیریت مدل‌ها» دانلودشان کنید.');
    }
  }

  void _log(String text) {
    if (!mounted) return;
    setState(() {
      _logs.insert(0, '[${TimeOfDay.now().format(context)}] $text');
      if (_logs.length > 80) _logs.removeLast();
    });
  }

  Future<void> _pickImage() async {
    final file = await _picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
    if (file == null) return;
    await _setImage(await file.readAsBytes(), file.name);
  }

  Future<void> _takePhoto() async {
    final file = await _picker.pickImage(source: ImageSource.camera, imageQuality: 100);
    if (file == null) return;
    await _setImage(await file.readAsBytes(), file.name);
  }

  Future<void> _pickFile() async {
    final picked = await FilePicker.platform.pickFiles(type: FileType.custom, allowedExtensions: ['png','jpg','jpeg','bmp','webp','pdf']);
    if (picked == null || picked.files.single.path == null) return;
    final f = File(picked.files.single.path!);
    final ext = picked.files.single.extension?.toLowerCase() ?? '';
    if (ext == 'pdf') {
      await _setPdf(f);
    } else {
      await _setImage(await f.readAsBytes(), picked.files.single.name, file: f);
    }
  }

  Future<void> _setImage(Uint8List bytes, String name, {File? file}) async {
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      _showError('تصویر قابل خواندن نیست.');
      return;
    }
    setState(() {
      _imageBytes = bytes;
      _fileName = name;
      _selectedFile = file;
      _pdfPageCount = 0;
      _result.clear();
      _status = 'تصویر آماده است.';
    });
    _log('تصویر انتخاب شد: $name (${decoded.width}×${decoded.height})');
  }

  Future<void> _setPdf(File file) async {
    try {
      final doc = await PdfDocument.openFile(file.path);
      final pageCount = doc.pageCount;
      final page = await doc.getPage(1);
      final rendered = await page.render(fullWidth: page.width * 1.5, fullHeight: page.height * 1.5);
      await doc.dispose();
      final image = img.Image(width: rendered.width, height: rendered.height);
      for (var y = 0; y < rendered.height; y++) {
        for (var x = 0; x < rendered.width; x++) {
          final i = (y * rendered.width + x) * 4;
          image.setPixelRgba(x, y, rendered.pixels[i], rendered.pixels[i+1], rendered.pixels[i+2], rendered.pixels[i+3]);
        }
      }
      setState(() {
        _imageBytes = Uint8List.fromList(img.encodePng(image));
        _fileName = file.path.split(Platform.pathSeparator).last;
        _selectedFile = file;
        _pdfPageCount = pageCount;
        _result.clear();
        _status = 'صفحه اول PDF آماده است.';
      });
      _log('PDF انتخاب شد: $_fileName ($_pdfPageCount صفحه)');
    } catch (e) {
      _showError('خطا در باز کردن PDF: $e');
    }
  }

  Future<void> _runOcr() async {
    if (_imageBytes == null) {
      _showError('ابتدا یک تصویر یا PDF انتخاب کنید.');
      return;
    }
    setState(() { _busy = true; _status = 'در حال آماده‌سازی موتور OCR…'; });
    try {
      await _engine.initialize();
      setState(() => _status = 'در حال تشخیص متن…');
      final result = await _engine.recognize(_imageBytes!, accurate: _accurate);
      setState(() {
        _result.text = result.text;
        _status = 'OCR کامل شد؛ ${result.lines.length} خط در ${result.elapsed.inMilliseconds}ms';
      });
      _log('OCR کامل شد: ${result.lines.length} خط، ${result.text.length} نویسه.');
    } catch (e, st) {
      _log('OCR ERROR: $e');
      debugPrint('$e\n$st');
      _showError('خطا در OCR: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _downloadAllModels() async {
    Navigator.of(context).pop();
    setState(() { _busy = true; _downloadProgress = 0; _status = 'در حال دانلود مدل‌ها…'; });
    var index = 0;
    try {
      for (final model in OcrModelCatalog.all) {
        index++;
        _log('شروع دانلود ${model.fileName}');
        await _models.ensure(model, onProgress: (received, total) {
          if (!mounted) return;
          setState(() {
            final current = total == null || total == 0 ? 0.0 : received / total;
            _downloadProgress = ((index - 1) + current) / OcrModelCatalog.all.length;
            _status = 'دانلود ${model.title}: ${(current * 100).toStringAsFixed(0)}٪';
          });
        });
        _log('دانلود و بررسی ${model.fileName} کامل شد.');
      }
      await _engine.initialize();
      setState(() { _status = 'همه مدل‌ها آماده و قابل استفاده آفلاین هستند.'; _downloadProgress = 1; });
    } catch (e) {
      _showError('دانلود مدل‌ها ناموفق بود: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _showModels() async {
    final states = await _models.status();
    if (!mounted) return;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => Directionality(
        textDirection: TextDirection.rtl,
        child: StatefulBuilder(builder: (context, setSheetState) {
          return Padding(
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 30),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              const Text('مدیریت مدل‌های OCR', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              const Text('مدل‌ها در حافظه برنامه ذخیره می‌شوند و پس از دانلود، OCR بدون اینترنت اجرا می‌شود.'),
              const SizedBox(height: 12),
              ...states.map((s) => ListTile(
                leading: Icon(s.installed ? Icons.check_circle : Icons.cloud_download, color: s.installed ? Colors.green : Colors.orange),
                title: Text(s.model.title),
                subtitle: Text(s.installed ? 'آماده • ${(s.bytes / 1024 / 1024).toStringAsFixed(1)} MB' : 'دانلود نشده'),
              )),
              const SizedBox(height: 8),
              FilledButton.icon(onPressed: _busy ? null : _downloadAllModels, icon: const Icon(Icons.download), label: const Text('دانلود / به‌روزرسانی مدل‌ها')),
            ]),
          );
        }),
      ),
    );
  }

  Future<void> _save() async {
    final text = _result.text;
    if (text.trim().isEmpty) { _showError('متنی برای ذخیره وجود ندارد.'); return; }
    try {
      final file = _export == 'pdf' ? await ExportService.savePdf(text) : await ExportService.saveText(text, extension: _export);
      _log('خروجی ذخیره شد: ${file.path}');
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('ذخیره شد: ${file.path}')));
    } catch (e) { _showError('خطا در ذخیره خروجی: $e'); }
  }

  void _copy() {
    Clipboard.setData(ClipboardData(text: _result.text));
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('متن کپی شد.')));
  }

  void _showError(String message) {
    if (!mounted) return;
    setState(() => _status = message);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message), backgroundColor: Colors.red.shade800));
  }

  @override
  void dispose() {
    _engine.dispose();
    _result.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('برنامه OCR فارسی'),
          actions: [IconButton(onPressed: _showModels, tooltip: 'مدیریت مدل‌ها', icon: const Icon(Icons.model_training))],
        ),
        body: LayoutBuilder(builder: (context, c) {
          final wide = c.maxWidth >= 900;
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(children: [
              _statusCard(),
              const SizedBox(height: 12),
              if (wide) Row(crossAxisAlignment: CrossAxisAlignment.start, children: [Expanded(child: _previewCard()), const SizedBox(width: 12), Expanded(child: _resultCard())])
              else ...[_previewCard(), const SizedBox(height: 12), _resultCard()],
              const SizedBox(height: 12),
              _logCard(),
            ]),
          );
        }),
      ),
    );
  }

  Widget _statusCard() => Card(child: Padding(padding: const EdgeInsets.all(14), child: Column(children: [
    Row(children: [const Icon(Icons.info_outline), const SizedBox(width: 10), Expanded(child: Text(_status))]),
    if (_busy && _downloadProgress > 0) ...[const SizedBox(height: 10), LinearProgressIndicator(value: _downloadProgress)],
  ])));

  Widget _previewCard() => Card(child: Padding(padding: const EdgeInsets.all(14), child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
    const Text('پیش‌نمایش سند', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
    const SizedBox(height: 10),
    Container(height: 280, decoration: BoxDecoration(color: Colors.black26, borderRadius: BorderRadius.circular(12)), child: _imageBytes == null ? const Center(child: Text('پیش‌نمایش اینجا نمایش داده می‌شود')) : ClipRRect(borderRadius: BorderRadius.circular(12), child: Image.memory(_imageBytes!, fit: BoxFit.contain))),
    const SizedBox(height: 12),
    Wrap(spacing: 8, runSpacing: 8, children: [
      FilledButton.icon(onPressed: _busy ? null : _pickImage, icon: const Icon(Icons.photo), label: const Text('انتخاب تصویر')),
      OutlinedButton.icon(onPressed: _busy ? null : _takePhoto, icon: const Icon(Icons.camera_alt), label: const Text('دوربین')),
      OutlinedButton.icon(onPressed: _busy ? null : _pickFile, icon: const Icon(Icons.picture_as_pdf), label: const Text('تصویر / PDF')),
    ]),
    const SizedBox(height: 10),
    Row(children: [Expanded(child: SwitchListTile(dense: true, contentPadding: EdgeInsets.zero, title: const Text('حالت دقیق'), value: _accurate, onChanged: _busy ? null : (v) { setState(() => _accurate=v); _prefs?.setBool('accurate', v); })), const SizedBox(width: 8), DropdownButton<String>(value: _export, items: const [DropdownMenuItem(value:'txt',child:Text('TXT')),DropdownMenuItem(value:'pdf',child:Text('PDF'))], onChanged: (v){if(v!=null){setState(()=>_export=v);_prefs?.setString('export',v);}})]),
    const SizedBox(height: 8),
    FilledButton.icon(onPressed: _busy ? null : _runOcr, icon: _busy ? const SizedBox(width:18,height:18,child:CircularProgressIndicator(strokeWidth:2)) : const Icon(Icons.document_scanner), label: Text(_busy ? 'در حال پردازش…' : 'اجرای OCR')),
  ])));

  Widget _resultCard() => Card(child: Padding(padding: const EdgeInsets.all(14), child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
    Row(children: [const Expanded(child: Text('نتیجه OCR', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold))), IconButton(onPressed: _copy, icon: const Icon(Icons.copy)), IconButton(onPressed: _save, icon: const Icon(Icons.save))]),
    const SizedBox(height: 8),
    TextField(controller: _result, maxLines: 18, textDirection: TextDirection.rtl, decoration: const InputDecoration(border: OutlineInputBorder(), hintText: 'متن تشخیص‌داده‌شده اینجا نمایش داده می‌شود.')),
    const SizedBox(height: 8),
    Text('${_result.text.split(RegExp(r'\s+')).where((e)=>e.isNotEmpty).length} کلمه • ${_result.text.length} نویسه'),
  ])));

  Widget _logCard() => Card(child: ExpansionTile(title: const Text('گزارش فعالیت'), children: [SizedBox(height: 180, child: ListView.builder(itemCount: _logs.length, itemBuilder: (_, i) => Padding(padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 3), child: Text(_logs[i], style: const TextStyle(fontSize: 12)))))]));
}

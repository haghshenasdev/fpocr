import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:pdf/pdf.dart';

class ExportService {
  static Future<File> saveText(String text, {String extension = 'txt'}) async {
    final dir = await getApplicationDocumentsDirectory();
    final name = 'persian_ocr_${DateTime.now().millisecondsSinceEpoch}.$extension';
    final file = File('${dir.path}/$name');
    await file.writeAsString(text, flush: true);
    return file;
  }

  static Future<File> savePdf(String text) async {
    final fontData = await rootBundle.load('assets/fonts/NotoNaskhArabic-Regular.ttf');
    final font = pw.Font.ttf(fontData);
    final doc = pw.Document();
    final lines = (text.isEmpty ? [''] : text.split('\n'));
    doc.addPage(pw.MultiPage(
      pageFormat: PdfPageFormat.a4,
      textDirection: pw.TextDirection.rtl,
      build: (_) => [
        for (final line in lines) pw.Directionality(
          textDirection: pw.TextDirection.rtl,
          child: pw.Text(line, style: pw.TextStyle(font: font, fontSize: 12)),
        ),
      ],
    ));
    final bytes = await doc.save();
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/persian_ocr_${DateTime.now().millisecondsSinceEpoch}.pdf');
    await file.writeAsBytes(Uint8List.fromList(bytes), flush: true);
    return file;
  }
}

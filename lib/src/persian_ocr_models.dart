class PersianOcrLine {
  final String text;
  final double confidence;
  final int left;
  final int top;
  final int right;
  final int bottom;

  const PersianOcrLine({
    required this.text,
    required this.confidence,
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });
}

class PersianOcrResult {
  final String text;
  final List<PersianOcrLine> lines;
  final Duration elapsed;

  const PersianOcrResult({
    required this.text,
    required this.lines,
    required this.elapsed,
  });
}

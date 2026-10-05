class PersianOcrNormalizer {
  static String normalize(String input) {
    var s = input;

    // Arabic presentation forms / common Arabic variants -> Persian forms.
    s = s
        .replaceAll('\u0640', '') // tatweel
        .replaceAll('\u064A', '\u06CC') // ي -> ی
        .replaceAll('\u0649', '\u06CC') // ى -> ی
        .replaceAll('\u06D2', '\u06D2')
        .replaceAll('\u0643', '\u06A9') // ك -> ک
        .replaceAll('\u0629', '\u0647') // ة -> ه (useful for Persian office OCR)
        .replaceAll('\u0624', '\u0648')
        .replaceAll('\u0626', '\u06CC')
        .replaceAll('\u0671', '\u0627')
        .replaceAll('\u0670', '')
        .replaceAll('\u064B', '')
        .replaceAll('\u064C', '')
        .replaceAll('\u064D', '')
        .replaceAll('\u064E', '')
        .replaceAll('\u064F', '')
        .replaceAll('\u0650', '')
        .replaceAll('\u0651', '')
        .replaceAll('\u0652', '')
        .replaceAll('\u0653', '')
        .replaceAll('\u0654', '')
        .replaceAll('\u0655', '')
        .replaceAll('\u0656', '')
        .replaceAll('\u0657', '')
        .replaceAll('\u0658', '')
        .replaceAll('\u0659', '')
        .replaceAll('\u065A', '')
        .replaceAll('\u065B', '')
        .replaceAll('\u065C', '')
        .replaceAll('\u065D', '')
        .replaceAll('\u065E', '')
        .replaceAll('\u065F', '');

    // Arabic/Persian digits: keep Persian digits for Iranian office workflows.
    const arabic = '٠١٢٣٤٥٦٧٨٩';
    const persian = '۰۱۲۳۴۵۶۷۸۹';
    for (var i = 0; i < arabic.length; i++) {
      s = s.replaceAll(arabic[i], persian[i]);
    }

    s = s.replaceAll(RegExp(r'[ \t\r\n]+'), ' ').trim();
    return s;
  }
}
